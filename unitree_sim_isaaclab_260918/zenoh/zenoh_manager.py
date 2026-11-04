import dexcomm
from dexbot_utils import HandType
from dexcomm.codecs import JointStateCodec, JointCmdCodec, DictDataCodec, JointModeCodec, BMSStateCodec, EStopStateCodec, BasicDataCodec, JointMotionCodec, WrenchStateCodec, WristButtonStateCodec
import json
import time
import torch
from functools import partial
import threading

# Mapping of a ZenohManager part (e.g. 'right_arm') to the dexcontrol 0.5
# motion-plugin topic suffix. dexcontrol derives these from each component's
# control_pub_topic by replacing 'control/' with 'motion/target/' (or
# 'motion/status/'). Arms use 'arm/{side}'; torso/head have no side suffix.
#
#   control/arm/right -> motion/target/arm/right / motion/status/arm/right
#   control/torso     -> motion/target/torso     / motion/status/torso
#   control/head      -> motion/target/head      / motion/status/head
MOTION_TARGET_TOPIC_SUFFIX = {
    'left_arm': 'arm/left',
    'right_arm': 'arm/right',
    'torso': 'torso',
    'head': 'head',
}

# Convergence thresholds for declaring a motion 'finished'. A gravity-loaded PD
# joint controller never settles exactly on the target (steady-state sag), so we
# declare completion when the arm is both close to the goal AND has stopped
# moving. This mirrors how a real motion controller decides "done".
_MOTION_POS_TOL = 0.05   # rad — position band around the goal
_MOTION_VEL_TOL = 0.05   # rad/s — joint speed below which the arm is "stopped"
# How long (seconds) to keep re-publishing a finished motion's terminal status.
# Covers the Zenoh multicast-discovery race where the client's lazily-created
# motion/status subscriber declaration reaches this publisher only after the
# motion has already converged.
_MOTION_FINISHED_GRACE_S = 5.0

class ZenohManager:
    def __init__(self, env, robot_type, enable_zenoh_hand=None):
        self.env = env
        self.robot_type = robot_type
        self.enable_zenoh_hand = enable_zenoh_hand
        self.dexcomm_node = None
        self.publishers = {}
        self.subscribers = {}
        self.joint_map = {}
        self.last_commands = {}
        # target_publishers[part]: re-publishes the held joint targets (last command per part)
        # on /state/target/... so clients can anchor streamed motions at the true setpoint
        # instead of the gravity-sagged measurement.
        self.target_publishers = {}
        # Wrist F/T emulation: /state/wrench/{side} + /state/wrist_button/{side} (see _publish_wrist_wrench)
        self.wrench_publishers = {}
        self.button_publishers = {}
        self._wrist_body_idx = None
        self._wrench_warned = False
        self._wrench_tick = 0
        # command_seq[part]: incremented on every received command message for
        # that part (control/* streaming or motion/target). Lets consumers tell a
        # *fresh* command apart from the buffered last one even if values repeat.
        self.command_seq = {}
        self._lock = threading.Lock()

        # --- Motion-plugin server state (dexcontrol 0.5) ---
        # motion_status_publishers[part]: publishes JSON status events to
        #   motion/status/<suffix>. Keys: the motion-target parts above.
        self.motion_status_publishers = {}
        # active_motions[part]: dict of motion_id -> target position list for
        # motions that have been accepted and are waiting for convergence.
        self.active_motions = {part: {} for part in MOTION_TARGET_TOPIC_SUFFIX}
        # recently_finished[part]: dict motion_id -> finish timestamp (monotonic).
        # The client creates its motion/status subscriber lazily (right before it
        # publishes the target), so with Zenoh multicast scouting its subscriber
        # declaration may not have propagated to this publisher by the time the
        # motion converges. We keep finished motions for a short window and
        # re-publish 'finished' each loop so a late client still unblocks.
        self.recently_finished = {part: {} for part in MOTION_TARGET_TOPIC_SUFFIX}

        self._setup_joint_mapping()
        self._setup_zenoh()

    def _setup_joint_mapping(self):
        if self.robot_type == "vega_1p":
            self.joint_map = {
                'left_arm': ['L_arm_j1', 'L_arm_j2', 'L_arm_j3', 'L_arm_j4', 'L_arm_j5', 'L_arm_j6', 'L_arm_j7'],
                'right_arm': ['R_arm_j1', 'R_arm_j2', 'R_arm_j3', 'R_arm_j4', 'R_arm_j5', 'R_arm_j6', 'R_arm_j7'],
                'torso': ['torso_j1', 'torso_j2', 'torso_j3'],
                'head': ['head_j1', 'head_j2', 'head_j3'],
                'chassis_steer': ['L_wheel_j1', 'R_wheel_j1'],
                'chassis_drive': ['L_wheel_j2', 'R_wheel_j2'],
            }
            if self.enable_zenoh_hand == "gripper":
                self.joint_map['left_hand'] = ['L_finger_joint']
                self.joint_map['right_hand'] = ['R_finger_joint']
            elif self.enable_zenoh_hand == "vertical":
                self.joint_map['left_hand'] = ['L_Slider_1', 'L_Slider_2']
                self.joint_map['right_hand'] = ['R_Slider_1', 'R_Slider_2']
            elif self.enable_zenoh_hand == "xhand":
                self.joint_map['left_hand'] = [
                    'left_hand_thumb_bend_joint', 'left_hand_thumb_rota_joint1', 'left_hand_thumb_rota_joint2',
                    'left_hand_index_bend_joint', 'left_hand_index_joint1', 'left_hand_index_joint2',
                    'left_hand_mid_joint1', 'left_hand_mid_joint2',
                    'left_hand_ring_joint1', 'left_hand_ring_joint2',
                    'left_hand_pinky_joint1', 'left_hand_pinky_joint2'
                ]
                self.joint_map['right_hand'] = [
                    'right_hand_thumb_bend_joint', 'right_hand_thumb_rota_joint1', 'right_hand_thumb_rota_joint2',
                    'right_hand_index_bend_joint', 'right_hand_index_joint1', 'right_hand_index_joint2',
                    'right_hand_mid_joint1', 'right_hand_mid_joint2',
                    'right_hand_ring_joint1', 'right_hand_ring_joint2',
                    'right_hand_pinky_joint1', 'right_hand_pinky_joint2'
                ]
        
        self.all_joint_names = self.env.scene["robot"].data.joint_names
        self.joint_to_index = {name: i for i, name in enumerate(self.all_joint_names)}

        self.source_indices = {}
        for part, names in self.joint_map.items():
            try:
                self.source_indices[part] = [self.joint_to_index[name] for name in names]
            except KeyError as e:
                print(f"[ZenohManager] Joint name {e} not found in robot model. Available joints: {self.all_joint_names}")
                raise

    def _setup_zenoh(self):
        try:
            self.dexcomm_node = dexcomm.Node(name="isaac_sim_zenoh_bridge")
            time.sleep(0.2)
            print("[ZenohManager] DexComm Node initialized successfully.")

            # service
            self._hand_querier = self.dexcomm_node.create_service(
                "/info/hand_type", self._handle_hand_type,
                request_decoder=None,
                response_encoder=DictDataCodec.encode,
            )
            self._left_arm_mode_querier = self.dexcomm_node.create_service(
                "/mode/arm/left", self._handle_mode_arm,
                request_decoder=JointModeCodec.decode,
                response_encoder=DictDataCodec.encode,
            )
            self._right_arm_mode_querier = self.dexcomm_node.create_service(
                "/mode/arm/right", self._handle_mode_arm,
                request_decoder=JointModeCodec.decode,
                response_encoder=DictDataCodec.encode,
            )
            self._head_mode_querier = self.dexcomm_node.create_service(
                "/mode/head", self._handle_mode_head,
                request_decoder=JointModeCodec.decode,
                response_encoder=DictDataCodec.encode,
            )
            self._estop_querier = self.dexcomm_node.create_service(
                "/system/estop", self._handle_estop,
                request_decoder=DictDataCodec.decode,
                response_encoder=DictDataCodec.encode,
            )

            # Publishers: 시뮬레이션 상태를 Zenoh 토픽으로 전송합니다.
            self.publishers['left_arm'] = self.dexcomm_node.create_publisher('/state/arm/left', encoder=JointStateCodec.encode)
            self.publishers['right_arm'] = self.dexcomm_node.create_publisher('/state/arm/right', encoder=JointStateCodec.encode)
            self.publishers['head'] = self.dexcomm_node.create_publisher('/state/head', encoder=JointStateCodec.encode)
            self.publishers['torso'] = self.dexcomm_node.create_publisher('/state/torso', encoder=JointStateCodec.encode)
            self.publishers['chassis_drive'] = self.dexcomm_node.create_publisher('/state/chassis/drive', encoder=JointStateCodec.encode)
            self.publishers['chassis_steer'] = self.dexcomm_node.create_publisher('/state/chassis/steer', encoder=JointStateCodec.encode)
            for _side in ('left', 'right'):
                self.wrench_publishers[_side] = self.dexcomm_node.create_publisher(f'/state/wrench/{_side}', encoder=WrenchStateCodec.encode)
                self.button_publishers[_side] = self.dexcomm_node.create_publisher(f'/state/wrist_button/{_side}', encoder=WristButtonStateCodec.encode)
            self.target_publishers['left_arm'] = self.dexcomm_node.create_publisher('/state/target/arm/left', encoder=JointStateCodec.encode)
            self.target_publishers['right_arm'] = self.dexcomm_node.create_publisher('/state/target/arm/right', encoder=JointStateCodec.encode)
            self.target_publishers['torso'] = self.dexcomm_node.create_publisher('/state/target/torso', encoder=JointStateCodec.encode)
            
            # Auxiliary status publishers
            self.publishers['battery'] = self.dexcomm_node.create_publisher('/state/bms', encoder=BMSStateCodec.encode)
            self.publishers['estop'] = self.dexcomm_node.create_publisher('/state/estop', encoder=EStopStateCodec.encode)
            self.publishers['heartbeat'] = self.dexcomm_node.create_publisher('/heartbeat', encoder=BasicDataCodec.encode)

            if self.enable_zenoh_hand in ("gripper", "vertical"):
                self.publishers['left_hand'] = self.dexcomm_node.create_publisher('/state/custom_ee/left', encoder=JointStateCodec.encode)
                self.publishers['right_hand'] = self.dexcomm_node.create_publisher('/state/custom_ee/right', encoder=JointStateCodec.encode)
            elif self.enable_zenoh_hand == "xhand":
                self.publishers['left_hand'] = self.dexcomm_node.create_publisher('/state/xhand/left', encoder=JointStateCodec.encode)
                self.publishers['right_hand'] = self.dexcomm_node.create_publisher('/state/xhand/right', encoder=JointStateCodec.encode)

            # Subscribers: Zenoh 토픽을 수신하여 시뮬레이션으로 전송합니다.
            self.subscribers['left_arm'] = self.dexcomm_node.create_subscriber(
                '/control/arm/left', partial(self._command_callback, part='left_arm'), decoder=JointCmdCodec.decode
            )
            self.subscribers['right_arm'] = self.dexcomm_node.create_subscriber(
                '/control/arm/right', partial(self._command_callback, part='right_arm'), decoder=JointCmdCodec.decode
            )
            self.subscribers['head'] = self.dexcomm_node.create_subscriber(
                '/control/head', partial(self._command_callback, part='head'), decoder=JointCmdCodec.decode
            )
            self.subscribers['torso'] = self.dexcomm_node.create_subscriber(
                '/control/torso', partial(self._command_callback, part='torso'), decoder=JointCmdCodec.decode
            )
            self.subscribers['chassis_drive'] = self.dexcomm_node.create_subscriber(
                '/control/chassis/drive', partial(self._command_callback, part='chassis_drive'), decoder=JointCmdCodec.decode
            )
            self.subscribers['chassis_steer'] = self.dexcomm_node.create_subscriber(
                '/control/chassis/steer', partial(self._command_callback, part='chassis_steer'), decoder=JointCmdCodec.decode
            )
            if self.enable_zenoh_hand in ("gripper", "vertical"):
                self.subscribers['left_hand'] = self.dexcomm_node.create_subscriber(
                    '/control/custom_ee/left', partial(self._command_callback, part='left_hand'), decoder=JointCmdCodec.decode
                )
                self.subscribers['right_hand'] = self.dexcomm_node.create_subscriber(
                    '/control/custom_ee/right', partial(self._command_callback, part='right_hand'), decoder=JointCmdCodec.decode
                )
            elif self.enable_zenoh_hand == "xhand":
                self.subscribers['left_hand'] = self.dexcomm_node.create_subscriber(
                    '/control/xhand/left', partial(self._command_callback, part='left_hand'), decoder=JointCmdCodec.decode
                )
                self.subscribers['right_hand'] = self.dexcomm_node.create_subscriber(
                    '/control/xhand/right', partial(self._command_callback, part='right_hand'), decoder=JointCmdCodec.decode
                )

            # --- Motion-plugin server (dexcontrol 0.5) ---
            # dexcontrol 0.5 move_to_joint_pos() publishes tracked targets to
            # motion/target/<suffix> (JointMotionCodec) and waits for JSON
            # status events on motion/status/<suffix>. We act as the plugin:
            # subscribe to targets, record the goal, and emit status.
            for part, suffix in MOTION_TARGET_TOPIC_SUFFIX.items():
                target_topic = f'motion/target/{suffix}'
                status_topic = f'motion/status/{suffix}'

                self.motion_status_publishers[part] = self.dexcomm_node.create_publisher(
                    status_topic, encoder=None  # we publish raw JSON bytes
                )
                self.subscribers[f'motion_target_{part}'] = self.dexcomm_node.create_subscriber(
                    target_topic,
                    partial(self._handle_motion_target, part=part),
                    decoder=JointMotionCodec.decode,
                )

            print("[ZenohManager] Zenoh publishers and subscribers created.")
        except Exception as e:
            print(f"[ZenohManager] DexComm (Zenoh) initialization failed: {e}")
            self.dexcomm_node = None

    def _command_callback(self, msg: dict, *, part: str):
        """Zenoh 토픽을 수신하여 action_provider에게 전달하여 시뮬레이션 로봇으로 전송"""
        # dict: {pos: [], vel: []}
        with self._lock:
            self.last_commands[part] = msg
            self.command_seq[part] = self.command_seq.get(part, 0) + 1

    def _publish_motion_status(self, part: str, motion_id: int, state: str, message: str = ""):
        """Publish a JSON status event for a tracked motion (dexcontrol 0.5).

        The client (ManagedJointComponent._on_motion_status) json.loads the raw
        bytes, so we must publish already-encoded JSON. encoder=None means the
        publisher takes bytes as-is.
        """
        publisher = self.motion_status_publishers.get(part)
        if publisher is None:
            return
        try:
            payload = json.dumps({"motion_id": int(motion_id), "state": state, "message": message}).encode("utf-8")
            publisher.publish(payload)
        except Exception as e:
            print(f"[ZenohManager] Failed to publish motion status for {part}: {e}")

    def _handle_motion_target(self, msg: dict, *, part: str):
        """Handle a motion/target/<suffix> message (JointMotionCodec decoded).

        Decoded fields (others added by the codec, e.g. timestamp_ns/mode, are
        ignored): 'pos' (list[float]), 'motion_id' (int, optional),
        'cancel_motion_id' (int, optional).
        """
        # Cancel request: drop the named motion and emit a terminal state so the
        # client's MotionHandle unblocks. No position update.
        cancel_id = msg.get("cancel_motion_id")
        if cancel_id is not None:
            with self._lock:
                existed = cancel_id in self.active_motions[part]
                self.active_motions[part].pop(cancel_id, None)
                # Stop re-publishing a stale 'finished' for a now-cancelled motion.
                self.recently_finished[part].pop(cancel_id, None)
            if existed:
                self._publish_motion_status(part, int(cancel_id), "cancelled", "cancelled by client")
            return

        pos = msg.get("pos")
        if pos is None:
            return

        # Drive the simulation with the target (same buffer the streaming
        # control/ topic used). This is what ZenohActionProvider.get_action reads.
        with self._lock:
            self.last_commands[part] = {'pos': list(pos), 'vel': []}
            self.command_seq[part] = self.command_seq.get(part, 0) + 1

        motion_id = msg.get("motion_id")
        if motion_id is None:
            # Fire-and-forget target (no handle): nothing to track or report.
            return

        motion_id = int(motion_id)
        # Supersede any earlier in-flight tracked motion for this part, mirroring
        # the client side which clears _active_handles before sending a new target.
        with self._lock:
            for old_id in list(self.active_motions[part].keys()):
                self._publish_motion_status(part, old_id, "cancelled", "superseded by new target")
            self.active_motions[part][motion_id] = list(pos)

        # Immediately acknowledge so the client flips _status_received (avoids the
        # 'plugin not running' path) and enters the working state.
        self._publish_motion_status(part, motion_id, "accepted")
        self._publish_motion_status(part, motion_id, "working")


    def clear_commands(self):
        """명령 버퍼 초기화"""
        with self._lock:
            self.last_commands.clear()
            print("[ZenohManager] Command buffer cleared.")

    def get_commands(self):
        with self._lock:
            return self.last_commands.copy()

    def get_command_seq(self):
        """Copy of per-part receive counters (see command_seq)."""
        with self._lock:
            return dict(self.command_seq)

    # Wrist link whose incoming joint (L/R_arm_j7) carries the wrist + gripper load, like the real F/T sensor.
    WRIST_BODIES = {'left': 'L_arm_l8', 'right': 'R_arm_l8'}

    def _publish_wrist_wrench(self):
        """Publish /state/wrench/{side} (+ /state/wrist_button/{side}) like the real robot's wrist F/T sensor.

        Source: ``Articulation.data.body_incoming_joint_wrench_b`` -- PhysX's joint reaction wrench of the
        wrist link (transmitted through L/R_arm_j7 into L/R_arm_l8), [fx, fy, fz, tx, ty, tz] in the
        parent link frame. Like a physical sensor it includes the gripper's own weight and inertia, so
        clients tare it before use (lotte_chilsung_ik_pink/read_force.py). Buttons are always released.
        Published every step; buttons every 20th step. Added 2026-09-23 for 10.left_mirror_and_push_force.py.
        """
        if not self.wrench_publishers:
            return
        try:
            robot = self.env.scene["robot"]
            if self._wrist_body_idx is None:
                idx = {}
                for side, body in self.WRIST_BODIES.items():
                    ids, _names = robot.find_bodies(body)
                    if ids:
                        idx[side] = ids[0]
                self._wrist_body_idx = idx
                print(f"[ZenohManager] wrist wrench source bodies: {idx} (body_incoming_joint_wrench_b)")
            wr = robot.data.body_incoming_joint_wrench_b[0]
            self._wrench_tick += 1
            for side, bi in self._wrist_body_idx.items():
                self.wrench_publishers[side].publish({'wrench': [float(v) for v in wr[bi].tolist()]})
                if self._wrench_tick % 20 == 1:
                    self.button_publishers[side].publish({'blue_button': False, 'green_button': False})
        except Exception as e:
            if not self._wrench_warned:
                self._wrench_warned = True
                print(f"[ZenohManager] wrist wrench unavailable ({e}); /state/wrench/* will not be published")

    def publish_joint_states(self):
        """시뮬레이션 로봇의 상태 정보를 Zenoh 토픽으로 publish"""
        if not self.dexcomm_node:
            return

        joint_pos = self.env.scene["robot"].data.joint_pos[0]
        joint_vel = self.env.scene["robot"].data.joint_vel[0]
        joint_effort = self.env.scene["robot"].data.applied_torque[0]

        for part, indices in self.source_indices.items():
            pos = joint_pos[indices].tolist()
            vel = joint_vel[indices].tolist()
            torque = joint_effort[indices].tolist()
            
            state_dict = {'pos': pos, 'vel': vel, 'torque': torque}

            if part in self.publishers:
                try:
                    self.publishers[part].publish(state_dict)
                except Exception as e:
                    print(f"[ZenohManager] Failed to publish {part} state: {e}")

        self._publish_wrist_wrench()

        # Held targets: last received command per part (only parts that have been commanded).
        with self._lock:
            held = {part: cmd.get('pos') for part, cmd in self.last_commands.items() if part in self.target_publishers}
        for part, pos in held.items():
            if not pos:
                continue
            try:
                self.target_publishers[part].publish({'pos': list(pos), 'vel': []})
            except Exception as e:
                print(f"[ZenohManager] Failed to publish {part} target: {e}")

        # Publish auxiliary status (battery, estop, heartbeat)
        try:
            current_time_ms = int(time.time() * 1000)
            
            # Battery state
            if 'battery' in self.publishers:
                battery_state = {
                    'percentage': 100,
                    'voltage': 24.0,
                    'current': 1.0,
                    'temperature': 25.0,
                    'is_charging': False
                }
                self.publishers['battery'].publish(battery_state)

            # EStop state
            if 'estop' in self.publishers:
                estop_state = {
                    'left_base_estop_enabled': False,
                    'right_base_estop_enabled': False,
                    'torso_estop_enabled': False,
                    'remote_estop_enabled': False,
                    'software_estop_enabled': False
                }
                self.publishers['estop'].publish(estop_state)

            # Heartbeat
            if 'heartbeat' in self.publishers:
                self.publishers['heartbeat'].publish(current_time_ms)
                
        except Exception as e:
            print(f"[ZenohManager] Failed to publish auxiliary status: {e}")

        # Check tracked motions for convergence and emit 'finished'.
        # Called every sim loop; joint_pos/joint_vel are already read above.
        try:
            self._update_active_motions(joint_pos, joint_vel)
        except Exception as e:
            print(f"[ZenohManager] Failed to update active motions: {e}")

    def _update_active_motions(self, joint_pos, joint_vel) -> None:
        """Drive tracked motions to completion and report status.

        Called once per sim loop from publish_joint_states().

        'working' is RE-EMITTED every loop while a motion is in flight. This is
        not redundant: the client creates its motion/status subscriber lazily,
        microseconds before publishing the target, so with Zenoh multicast
        scouting the simulator's initial 'accepted'/'working' burst can be lost
        to a declaration-propagation race. Re-emitting 'working' guarantees the
        client catches at least one status event within its wait window,
        turning a spurious 'plugin not running' (PluginNotAvailableError) into a
        normal 'still in flight' wait.

        'finished' is emitted (and the motion dropped from active tracking)
        when the arm has converged: within the position band AND stopped. A
        gravity-loaded PD controller never settles exactly on the goal
        (observed ~0.038 rad steady-state sag), so exact-match would never fire.
        """
        with self._lock:
            for part, motions in self.active_motions.items():
                if not motions:
                    continue
                indices = self.source_indices.get(part)
                if indices is None:
                    continue
                current_pos = [float(joint_pos[i]) for i in indices]
                current_vel = [float(joint_vel[i]) for i in indices]
                for motion_id, target in list(motions.items()):
                    n = min(len(target), len(current_pos))
                    max_err = max(abs(target[i] - current_pos[i]) for i in range(n)) if n else 0.0
                    max_vel = max(abs(current_vel[i]) for i in range(n)) if n else 0.0
                    if max_err <= _MOTION_POS_TOL and max_vel <= _MOTION_VEL_TOL:
                        self._publish_motion_status(part, motion_id, "finished")
                        motions.pop(motion_id, None)
                        # Remember it so a client whose subscriber declaration
                        # arrives late still catches the terminal event.
                        self.recently_finished[part][motion_id] = time.monotonic()
                    else:
                        # Re-emit progress so a client that missed the initial
                        # burst (late subscriber declaration) still sees activity.
                        self._publish_motion_status(part, motion_id, "working")

            # Re-publish recently-finished motions within the grace window so a
            # client that subscribed AFTER the motion converged still unblocks.
            # This runs every loop regardless of whether there are active motions.
            now = time.monotonic()
            for part, finished in self.recently_finished.items():
                for fid, fin_ts in list(finished.items()):
                    if now - fin_ts > _MOTION_FINISHED_GRACE_S:
                        finished.pop(fid, None)
                    else:
                        self._publish_motion_status(part, fid, "finished")

    def stop(self):
        if self.dexcomm_node:
            try:
                # Some versions of dexcomm may not have a close() method on Node,
                # or it might be named differently (e.g. destroy()).
                if hasattr(self.dexcomm_node, 'close'):
                    self.dexcomm_node.close()
                elif hasattr(self.dexcomm_node, 'destroy'):
                    self.dexcomm_node.destroy()
                else:
                    print(f"[ZenohManager] Warning: Node object has no 'close' or 'destroy' method.")
            except Exception as e:
                print(f"[ZenohManager] Error closing Node: {e}")

    def _handle_hand_type(self, request):
        # if self.enable_gripper:
        #     return {"left": HandType.DexGripper, "right": HandType.DexGripper}
        return {"left": HandType.UNKNOWN, "right": HandType.UNKNOWN}
        # if self._hand_type == "gripper":
        #     return {"left": HandType.DexGripper, "right": HandType.DexGripper}
        # elif self._hand_type == "hand":
        #     return {"left": HandType.HandF5D6_V2, "right": HandType.HandF5D6_V2}
        # else:
        #     return {"left": HandType.UNKNOWN, "right": HandType.UNKNOWN}

    def _handle_mode_arm(self, request):
        return {"mode": [4, 4, 4, 4, 4, 4, 4]}

    def _handle_mode_head(self, request):
        return {"mode": [4, 4, 4]}

    def _handle_estop(self, request):
        return {"success": True}

zenoh_manager_instance = None

def create_zenoh_manager(env, robot_type, enable_zenoh_hand=None):
    global zenoh_manager_instance
    if zenoh_manager_instance is None:
        zenoh_manager_instance = ZenohManager(env, robot_type, enable_zenoh_hand)
    return zenoh_manager_instance

def get_zenoh_manager():
    return zenoh_manager_instance