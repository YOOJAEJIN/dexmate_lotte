from action_provider.action_base import ActionProvider
from typing import Optional
import math
import os
import time
import traceback
import torch
from zenoh.zenoh_manager import get_zenoh_manager

class ZenohActionProvider(ActionProvider):
    def __init__(self, env, args_cli):
        super().__init__("ZenohActionProvider")
        self.env = env
        self.robot_type = args_cli.robot_type
        self.enable_zenoh_hand = getattr(args_cli, "enable_zenoh_hand", None)
        self.zenoh_manager = get_zenoh_manager()
        # Slider gripper lock (vertical hand only), one-way "ratchet": while the slider
        # targets ask for a close, the opening-side joint limit of each slider is kept just
        # behind the finger and re-tightened as the finger advances. The finger can still
        # close (the 23 N drive keeps squeezing the object, effort_limit_sim untouched) but
        # can never be pushed back open by an external force along the slider axis; the
        # hold comes from the joint-limit constraint. Released as soon as the targets are
        # no longer a close (open command, recorded open, ...).
        self.gripper_lock_enabled = (
            self.enable_zenoh_hand == "vertical"
            and not getattr(args_cli, "disable_gripper_lock", False)
        )
        self.gripper_lock_margin = 2e-4        # m: opening-side limit sits this far behind the finger
        self.gripper_lock_ratchet_step = 5e-4  # m: re-tighten once the finger advanced this much
        self._gripper_lock = {}               # side -> state dict, filled in _setup_joint_mapping
        # A wrapper (tools/action_recorder.py SegmentRecorder) that replays / holds actions sets
        # this True and feeds the *final* action tensor to observe_action() itself, so the lock
        # follows what the robot is actually commanded (recorded close included), not just live
        # zenoh hand commands.
        self.gripper_lock_external = False
        self._gripper_lock_error_logged = False
        self._gripper_lock_status_path = os.path.join(
            os.environ.get("PROJECT_ROOT", os.getcwd()), "slider_lock_status.txt")
        self._gripper_lock_status_t = 0.0
        self._setup_joint_mapping()
        
    def _setup_joint_mapping(self):
        self.all_joint_names = self.env.scene["robot"].data.joint_names
        if not self.all_joint_names:
            print("[ZenohActionProvider] Warning: Robot joint names are not available yet.")
            return

        self.joint_to_index = {name: i for i, name in enumerate(self.all_joint_names)}
        
        # Define joints for each part based on vega_1p structure
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

        # Identify Position joints (matching ActionsCfg.arm_head_torso_pos)
        pos_part_names = ['left_arm', 'right_arm', 'head', 'torso', 'chassis_steer']
        if self.enable_zenoh_hand in ["gripper", "vertical", "xhand"]:
            pos_part_names.extend(['left_hand', 'right_hand'])
        
        pos_joint_names = []
        for part in pos_part_names:
            pos_joint_names.extend(self.joint_map[part])
            
        # Isaac Lab sorts matched joints by articulation index for efficiency
        # We must follow this sorted order for the position action tensor slice (indices 0 to 21)
        pos_joints_in_sim = sorted(
            [name for name in pos_joint_names if name in self.joint_to_index],
            key=lambda name: self.joint_to_index[name]
        )
        self.pos_name_to_action_idx = {name: i for i, name in enumerate(pos_joints_in_sim)}
        
        # Identify Velocity joints (matching ActionsCfg.chassis_drive_vel)
        vel_joint_names = self.joint_map['chassis_drive']
        vel_joints_in_sim = sorted(
            [name for name in vel_joint_names if name in self.joint_to_index],
            key=lambda name: self.joint_to_index[name]
        )
        self.vel_name_to_action_idx = {name: i for i, name in enumerate(vel_joints_in_sim)}

        # Pre-calculate mapping for each Zenoh part for faster lookup in get_action
        default_joint_pos = self.env.scene["robot"].data.default_joint_pos[0]
        self.pos_mapping = {}
        for part in pos_part_names:
            indices = []
            valid_names = []
            for name in self.joint_map[part]:
                if name in self.pos_name_to_action_idx:
                    indices.append(self.pos_name_to_action_idx[name])
                    valid_names.append(name)
            if indices:
                self.pos_mapping[part] = {
                    'action_indices': torch.tensor(indices, dtype=torch.long, device=self.env.device),
                    'names': valid_names,
                    # ActionsCfg uses use_default_offset=True: the action term adds
                    # default_joint_pos back to the raw action. Zenoh commands are
                    # absolute joint targets, so subtract the default here to keep the
                    # commanded pose exact (no-op for tasks with zero defaults).
                    'default_pos': default_joint_pos[
                        [self.joint_to_index[name] for name in valid_names]
                    ].to(self.env.device),
                }
            
        self.vel_mapping = {}
        part = 'chassis_drive'
        indices = []
        valid_names = []
        for name in self.joint_map[part]:
            if name in self.vel_name_to_action_idx:
                indices.append(self.vel_name_to_action_idx[name])
                valid_names.append(name)
        if indices:
            self.vel_mapping[part] = {
                'action_indices': torch.tensor(indices, dtype=torch.long, device=self.env.device),
                'names': valid_names
            }

        if self.gripper_lock_enabled:
            self._setup_gripper_lock()

        self._initialized = True

    # ------------------------------------------------------------------
    # slider gripper lock
    # ------------------------------------------------------------------
    def _setup_gripper_lock(self):
        robot = self.env.scene["robot"]
        self._gripper_lock = {}
        for side in ("left_hand", "right_hand"):
            names = [n for n in self.joint_map.get(side, []) if n in self.joint_to_index]
            if len(names) != 2:
                continue
            ids = [self.joint_to_index[n] for n in names]
            limits0 = robot.data.joint_pos_limits[0, ids].clone()          # (2, 2) [lower, upper]
            default0 = robot.data.default_joint_pos[0, ids].clone()        # (2,)
            # closing end = the limit farther from zero travel (Slider_1 -> lower, Slider_2 -> upper)
            close_end = torch.where(limits0[:, 0].abs() > limits0[:, 1].abs(), limits0[:, 0], limits0[:, 1])
            open_end = torch.where(limits0[:, 0].abs() > limits0[:, 1].abs(), limits0[:, 1], limits0[:, 0])
            self._gripper_lock[side] = {
                "names": names,
                "ids": ids,
                "ids_t": torch.tensor(ids, dtype=torch.long, device=robot.device),
                "limits0": limits0,
                "default0": default0,
                "open_end": open_end,
                "close_end": close_end,
                # +1: closing increases the joint position (Slider_2), -1: decreases (Slider_1)
                "close_sign": torch.sign(close_end - open_end),
                "locked": False,
                "open_lim": None,      # (2,) current opening-side limit while locked
                "tightened": 0,        # number of ratchet re-tightenings since engage
                "want_closed": False,
            }
        print(f"[ZenohActionProvider] slider gripper ratchet lock enabled for {list(self._gripper_lock.keys())} "
              f"(margin={self.gripper_lock_margin} m, step={self.gripper_lock_ratchet_step} m, "
              f"status={self._gripper_lock_status_path})")

    def _slider_targets_from_action(self, action: Optional[torch.Tensor]) -> dict:
        """{side: [target_s1, target_s2]} (absolute joint targets) read from a final action tensor."""
        out = {}
        if action is None:
            return out
        flat = action.detach().reshape(-1)
        for side in self._gripper_lock:
            m = self.pos_mapping.get(side)
            if m is None:
                continue
            idx = m["action_indices"]
            if int(idx.max()) >= flat.numel():
                continue
            out[side] = (flat[idx.to(flat.device)] + m["default_pos"].to(flat.device)).tolist()
        return out

    def _hand_close_intent(self, side: str, targets: dict) -> bool:
        """True when the slider targets ask for more than half travel toward the closed end."""
        st = self._gripper_lock[side]
        cmd = targets.get(side)
        if not cmd or len(cmd) < 2:
            return False
        target = torch.tensor(cmd[:2], dtype=torch.float32, device=st["open_end"].device)
        frac = (target - st["open_end"]) / (st["close_end"] - st["open_end"])
        return bool((frac > 0.5).all())

    def observe_action(self, action: Optional[torch.Tensor]) -> None:
        """Feed the action tensor that is actually sent to the robot this step (lock bookkeeping)."""
        if not self.gripper_lock_enabled or not self._gripper_lock:
            return
        try:
            self._update_gripper_lock(self._slider_targets_from_action(action))
        except Exception:
            if not self._gripper_lock_error_logged:
                self._gripper_lock_error_logged = True
                print("[ZenohActionProvider] gripper lock update failed (further errors suppressed):")
                traceback.print_exc()

    def _write_slider_limits(self, side: str, limits: torch.Tensor):
        """Write new [lower, upper] limits for one hand's two sliders into PhysX.

        Mirrors Articulation.write_joint_position_limit_to_sim, except that continuous
        joints (wheels report +-FLT_MAX) are written as +-2*pi: PhysX validates every
        revolute limit it receives (the tensor API always writes all DOFs) and raises a
        GUI error popup for values outside [-2pi, 2pi], while a free joint ignores the
        stored limit anyway. Also skips the default_joint_pos clamping IsaacLab does.
        """
        robot = self.env.scene["robot"]
        st = self._gripper_lock[side]
        data = robot.data
        ids_t = st["ids_t"]
        data.joint_pos_limits[0, ids_t] = limits.to(data.joint_pos_limits.device, data.joint_pos_limits.dtype)
        lim = data.joint_pos_limits.clone()
        big = lim.abs() > 1e6
        if bool(big.any()):
            lim[big] = torch.sign(lim[big]) * (2.0 * math.pi)
        robot.root_physx_view.set_dof_limits(lim.cpu(), indices=robot._ALL_INDICES.cpu())
        # keep IsaacLab's derived soft limits consistent for the sliders
        mean = (data.joint_pos_limits[0, ids_t, 0] + data.joint_pos_limits[0, ids_t, 1]) / 2
        rng = data.joint_pos_limits[0, ids_t, 1] - data.joint_pos_limits[0, ids_t, 0]
        factor = robot.cfg.soft_joint_pos_limit_factor
        data.soft_joint_pos_limits[0, ids_t, 0] = mean - 0.5 * rng * factor
        data.soft_joint_pos_limits[0, ids_t, 1] = mean + 0.5 * rng * factor

    def _ratchet_limits(self, st: dict, open_lim: torch.Tensor) -> torch.Tensor:
        """(2, 2) [lower, upper]: original closing-side end, ``open_lim`` on the opening side."""
        lim0 = st["limits0"]
        closing_pos = st["close_sign"] > 0
        lower = torch.where(closing_pos, open_lim, lim0[:, 0])
        upper = torch.where(closing_pos, lim0[:, 1], open_lim)
        return torch.stack([lower, upper], dim=-1)

    def _open_side_limit_for(self, st: dict, pos: torch.Tensor) -> torch.Tensor:
        """Opening-side limit that leaves ``margin`` of play behind the finger at ``pos``."""
        lim = pos - st["close_sign"] * self.gripper_lock_margin
        return torch.minimum(torch.maximum(lim, st["limits0"][:, 0]), st["limits0"][:, 1])

    def _update_gripper_lock(self, targets: dict):
        if not self._gripper_lock:
            return
        robot = self.env.scene["robot"]
        changed = False
        for side, st in self._gripper_lock.items():
            want_closed = self._hand_close_intent(side, targets)
            st["want_closed"] = want_closed
            if not want_closed:
                if st["locked"]:
                    self._write_slider_limits(side, st["limits0"])
                    st["locked"] = False
                    st["open_lim"] = None
                    changed = True
                    print(f"[ZenohActionProvider] {side} slider ratchet released")
                continue
            pos = robot.data.joint_pos[0, st["ids_t"]].clone()
            if not st["locked"]:
                st["open_lim"] = self._open_side_limit_for(st, pos)
                self._write_slider_limits(side, self._ratchet_limits(st, st["open_lim"]))
                st["locked"] = True
                st["tightened"] = 0
                changed = True
                print(f"[ZenohActionProvider] {side} slider ratchet engaged at "
                      f"{[round(v, 4) for v in pos.tolist()]} ({st['names']}); fingers can only close now")
                continue
            # locked: follow the finger as it advances toward the closed end (never loosen)
            new_lim = self._open_side_limit_for(st, pos)
            advance = (new_lim - st["open_lim"]) * st["close_sign"]   # >0 = finger moved further closed
            tighten = advance > self.gripper_lock_ratchet_step
            if bool(tighten.any()):
                st["open_lim"] = torch.where(tighten, new_lim, st["open_lim"])
                self._write_slider_limits(side, self._ratchet_limits(st, st["open_lim"]))
                st["tightened"] += 1
                changed = True
        self._write_gripper_lock_status(targets, force=changed)

    def _write_gripper_lock_status(self, targets: dict, force: bool = False):
        """Dump lock state to <PROJECT_ROOT>/slider_lock_status.txt (on change and every 0.5 s)."""
        now = time.time()
        if not force and now - self._gripper_lock_status_t < 0.5:
            return
        self._gripper_lock_status_t = now
        try:
            robot = self.env.scene["robot"]
            lines = [f"# slider lock status  t={now:.2f}"]
            for side, st in self._gripper_lock.items():
                pos = robot.data.joint_pos[0, st["ids_t"]].tolist()
                vel = robot.data.joint_vel[0, st["ids_t"]].tolist()
                lim = robot.data.joint_pos_limits[0, st["ids_t"]].tolist()
                tgt = targets.get(side)
                lines.append(
                    f"{side} locked={int(st['locked'])} want_closed={int(st['want_closed'])} tightened={st['tightened']} "
                    f"target={[round(v, 4) for v in tgt] if tgt else None} pos={[round(v, 5) for v in pos]} vel={[round(v, 5) for v in vel]} "
                    f"limits={[[round(a, 5), round(b, 5)] for a, b in lim]}"
                )
            tmp = self._gripper_lock_status_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            os.replace(tmp, self._gripper_lock_status_path)
        except Exception:
            pass

    def _release_all_gripper_locks(self):
        for side, st in self._gripper_lock.items():
            if st["locked"]:
                try:
                    self._write_slider_limits(side, st["limits0"])
                except Exception as e:
                    print(f"[ZenohActionProvider] failed to release {side} slider lock: {e}")
                st["locked"] = False
            st["open_lim"] = None

    def get_action(self, env) -> Optional[torch.Tensor]:
        if not self.zenoh_manager:
            return None
            
        if not hasattr(self, '_initialized') or not self._initialized:
            self._setup_joint_mapping()
            if not hasattr(self, '_initialized') or not self._initialized:
                return None
            
        # {part: {pos: [], vel: []}}
        commands = self.zenoh_manager.get_commands()
        
        # We build the action tensor in the order expected by ActionsCfg:
        # [PositionTerm (22, 24, 26, or 46 joints), VelocityTerm (2 joints)]
        if self.enable_zenoh_hand == "gripper":
            pos_size = 24
        elif self.enable_zenoh_hand == "vertical":
            pos_size = 26
        elif self.enable_zenoh_hand == "xhand":
            pos_size = 46
        else:
            pos_size = 22

        pos_action = torch.zeros(pos_size, device=self.env.device)
        vel_action = torch.zeros(2, device=self.env.device)
        
        # Fill pos_action
        for part, mapping in self.pos_mapping.items():
            if part in commands:
                pos_data = commands[part].get('pos', [0.0]*len(mapping['names']))
                if len(pos_data) >= len(mapping['names']):
                    # subtract the default pose so the action term's
                    # use_default_offset adds it back (absolute target preserved)
                    pos_action[mapping['action_indices']] = torch.tensor(
                        pos_data[:len(mapping['names'])], dtype=torch.float, device=self.env.device
                    ) - mapping['default_pos']
            
        # Fill vel_action with drive wheel negation logic
        if 'chassis_drive' in commands:
            vel_data = commands['chassis_drive'].get('vel', [0.0, 0.0])
            if len(vel_data) >= 2:
                # Assuming Zenoh sends [L_vel, R_vel]
                # Right wheel (R_wheel_j2) usually needs negation
                l_vel = vel_data[0]
                r_vel = -vel_data[1]
                
                if 'L_wheel_j2' in self.vel_name_to_action_idx:
                    vel_action[self.vel_name_to_action_idx['L_wheel_j2']] = float(l_vel)
                if 'R_wheel_j2' in self.vel_name_to_action_idx:
                    vel_action[self.vel_name_to_action_idx['R_wheel_j2']] = float(r_vel)
                    
        #         print(f"[ZenohActionProvider] chassis_drive vel_data: {vel_data}, vel_action: {vel_action.tolist()}, L_idx: {self.vel_name_to_action_idx.get('L_wheel_j2')}, R_idx: {self.vel_name_to_action_idx.get('R_wheel_j2')}")
        #     else:
        #         print(f"[ZenohActionProvider] Warning: chassis_drive vel_data length < 2: {vel_data}")
        # else:
        #     print("[ZenohActionProvider] Warning: 'chassis_drive' not in commands")
            
        result = torch.cat([pos_action, vel_action]).unsqueeze(0)
        if self.gripper_lock_enabled and not self.gripper_lock_external:
            self.observe_action(result)
        return result
        
    def cleanup(self):
        pass # Manager handles cleanup

    def reset(self):
        self._release_all_gripper_locks()
        if self.zenoh_manager:
            self.zenoh_manager.clear_commands()
