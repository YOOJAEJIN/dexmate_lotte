import argparse
import traceback
import rclpy
from dexbot_utils import HandType
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.executors import MultiThreadedExecutor
import dexcomm
import logging
import time
from functools import partial

# 제어 명령을 보낼 때
# control_msg = {"pos": target_pos, "vel": target_vel}
# self._publish_control(
#     control_msg)  # -> 최종적으로 JointCmdCodec.encode(control_msg) 호출# dexcontrol 라이브러리가 내부적으로 사용하는 코덱을 그대로 사용합니다.
from dexcomm.codecs import JointStateCodec, JointCmdCodec, DictDataCodec, JointModeCodec, SoftwareEstopCodec

# DexComm 로그 레벨 설정 (디버깅 용)
logging.getLogger('dexcomm').setLevel(logging.INFO)

# --- 설정값 ---
ROS2_COMMAND_TOPIC = '/joint_command'
ROS2_STATE_TOPIC = '/joint_states'
NODE_NAME = 'dexcomm_ros2_bridge'
# ------------

# 실행 ---------------------------------
# zenoh 토픽은 ROBOT_NAME 환경변수를 읽어서 앞에 붙인다.
# vega-1 + f5d6
# export ROBOT_NAME=dm/vg1234567890-1
# export ZENOH_CONFIG=/home/ubuntu/.dexmate/comm/zenoh/zenoh_config.json5
# python examples/bridge/zenoh_to_ros2.py --hand_type=hand
# vega-1p + gripper
# export ROBOT_NAME=dm/vg1234567890-1p
# export ZENOH_CONFIG=/home/ubuntu/.dexmate/comm/zenoh/zenoh_config.json5
# python examples/bridge/zenoh_to_ros2.py --hand_type=gripper
# -------------------------------------

class DexcommToRos2Bridge:
    """
    Dexcomm Node를 사용하여 Zenoh Key와 ROS2 Topic 간의 데이터를 중계하는 브리지 클래스
    """

    def __init__(self, node: Node, hand_type: str = "",
                 namespace: str = ""):
        self._node = node
        self._hand_type = hand_type
        self._logger = self._node.get_logger()

        # 1. Dexcomm Node 초기화
        try:
            self.dexcomm_node = dexcomm.Node(name="zenoh_bridge", namespace=namespace)
            time.sleep(0.2)  # Node 스레드가 준비될 시간을 줍니다.
            self._logger.info('DexComm Node initialized successfully.')
        except Exception as e:
            self._logger.error(f'Failed to initialize DexComm Node: {e}')
            raise

        # service
        self._hand_querier = self.dexcomm_node.create_service(
            "/info/hand_type", self.handle_hand_type,
            request_decoder=None,
            response_encoder=DictDataCodec.encode,
        )
        self._left_arm_mode_querier = self.dexcomm_node.create_service(
            "/mode/arm/left", self.handle_mode_arm,
            request_decoder=JointModeCodec.decode,
            response_encoder=DictDataCodec.encode,
        )
        self._right_arm_mode_querier = self.dexcomm_node.create_service(
            "/mode/arm/right", self.handle_mode_arm,
            request_decoder=JointModeCodec.decode,
            response_encoder=DictDataCodec.encode,
        )
        self._head_mode_querier = self.dexcomm_node.create_service(
            "/mode/head", self.handle_mode_head,
            request_decoder=JointModeCodec.decode,
            response_encoder=DictDataCodec.encode,
        )
        self._estop_querier = self.dexcomm_node.create_service(
            "/system/estop", self.handle_estop,
            request_decoder=SoftwareEstopCodec.decode,
            response_encoder=None,
        )

        # 2. Joint 이름 매핑 정의
        self.joint_map = {
            'head': ['head_j1', 'head_j2', 'head_j3'],
            'torso': ['torso_j1', 'torso_j2', 'torso_j3'],
            'chassis_steer': ["L_wheel_j1", "R_wheel_j1"],
            'chassis_drive': ["L_wheel_j2", "R_wheel_j2"],
            'left_arm': ['L_arm_j1', 'L_arm_j2', 'L_arm_j3', 'L_arm_j4', 'L_arm_j5', 'L_arm_j6', 'L_arm_j7'],
            'right_arm': ['R_arm_j1', 'R_arm_j2', 'R_arm_j3', 'R_arm_j4', 'R_arm_j5', 'R_arm_j6', 'R_arm_j7'],
        }
        if self._hand_type == "gripper":
            self.joint_map.update({
                'left_hand': ['L_gripper_j1'],
                'right_hand': ['R_gripper_j1'],
            })
        elif self._hand_type == "hand":
            self.joint_map.update({
                'left_hand': ['L_ff_j1', 'L_lf_j1', 'L_mf_j1', 'L_rf_j1', 'L_th_j1', 'L_th_j2'],
                'right_hand': ['R_ff_j1', 'R_lf_j1', 'R_mf_j1', 'R_rf_j1', 'R_th_j1', 'R_th_j2'],
            })
        self.name_to_part_map = {name: part for part, names in self.joint_map.items() for name in names}

        # 3. ROS2 퍼블리셔 및 서브스크라이버 초기화
        self.ros2_pub = self._node.create_publisher(JointState, ROS2_COMMAND_TOPIC, 10)
        self.ros2_sub = self._node.create_subscription(JointState, ROS2_STATE_TOPIC, self.ros2_listener_callback, 10)
        self._logger.info(f'ROS2 Publisher on topic: {ROS2_COMMAND_TOPIC}')
        self._logger.info(f'ROS2 Subscriber on topic: {ROS2_STATE_TOPIC}')

        # 4. Dexcomm 퍼블리셔 및 서브스크라이버 초기화 (dexcontrol 라이브러리와 코덱 일치)
        self.publishers = {
            'head': self.dexcomm_node.create_publisher('/state/head',
                                                       encoder=JointStateCodec.encode),
            'torso': self.dexcomm_node.create_publisher('/state/torso',
                                                        encoder=JointStateCodec.encode),
            'chassis_drive': self.dexcomm_node.create_publisher('/state/chassis/drive',
                                                        encoder=JointStateCodec.encode),
            'chassis_steer': self.dexcomm_node.create_publisher('/state/chassis/steer',
                                                        encoder=JointStateCodec.encode),
            'left_arm': self.dexcomm_node.create_publisher('/state/arm/left',
                                                           encoder=JointStateCodec.encode),
            'right_arm': self.dexcomm_node.create_publisher('/state/arm/right',
                                                            encoder=JointStateCodec.encode),
        }

        if self._hand_type == 'gripper':
            self.publishers['left_hand'] = self.dexcomm_node.create_publisher('/state/gripper/left',
                                                                             encoder=JointStateCodec.encode)
            self.publishers['right_hand'] = self.dexcomm_node.create_publisher('/state/gripper/right',
                                                                              encoder=JointStateCodec.encode)
        elif self._hand_type == 'hand':
            self.publishers['left_hand'] = self.dexcomm_node.create_publisher('/state/hand/left',
                                                                             encoder=JointStateCodec.encode)
            self.publishers['right_hand'] = self.dexcomm_node.create_publisher('/state/hand/right',
                                                                              encoder=JointStateCodec.encode)

        self.subscribers = {}
        for part_name in self.joint_map.keys():
            topic_key = ""
            if "arm" in part_name:
                side = part_name.split("_")[0]
                topic_key = f"/control/arm/{side}"
            elif "hand" in part_name:
                side = part_name.split("_")[0]
                if self._hand_type == "hand":
                    topic_key = f"/control/hand/{side}"
                elif self._hand_type == "gripper":
                    topic_key = f"/control/gripper/{side}"
            elif "chassis" in part_name:
                sub_part = part_name.split("_")[1] # steer or drive
                topic_key = f"/control/chassis/{sub_part}"
            else:
                topic_key = f"/control/{part_name}"

            callback = partial(self.dexcomm_listener, part=part_name)
            self.subscribers[part_name] = self.dexcomm_node.create_subscriber(
                topic_key,
                callback,
                decoder=JointCmdCodec.decode
            )
        self._logger.info('Dexcomm Publishers and Subscribers initialized.')

    def handle_hand_type(self, request):
        if self._hand_type == "gripper":
            return {"left": HandType.DexGripper, "right": HandType.DexGripper}
        elif self._hand_type == "hand":
            return {"left": HandType.HandF5D6_V2, "right": HandType.HandF5D6_V2}
        else:
            return {"left": HandType.UNKNOWN, "right": HandType.UNKNOWN}

    def handle_mode_arm(self, request):
        return {"mode": [4, 4, 4, 4, 4, 4, 4]}

    def handle_mode_head(self, request):
        return {"mode": [4, 4, 4]}

    def handle_estop(self, request):
        return None

    def ros2_listener_callback(self, msg: JointState):
        """ROS2 /joint_states 토픽을 수신하여 Dexcomm으로 전송합니다."""
        # self._logger.info(f"msg: {msg}")

        part_data = {}
        for i, name in enumerate(msg.name):
            part = self.name_to_part_map.get(name)
            if not part:
                continue

            if part not in part_data:
                part_data[part] = {'position': [], 'velocity': [], 'torque': []}

            part_data[part]['position'].append(msg.position[i])
            part_data[part]['velocity'].append(msg.velocity[i] if msg.velocity else 0.0)
            part_data[part]['torque'].append(msg.effort[i] if msg.effort else 0.0)

        for part, data in part_data.items():
            if part in self.publishers:
                try:
                    # JointStateCodec.encode는 딕셔너리를 인자로 받습니다.
                    state_dict = {
                        'pos': data['position'],
                        'vel': data['velocity'],
                        'torque': data['torque']
                    }
                    self.publishers[part].publish(state_dict)
                    self._logger.debug(f"Published {part} state to Dexcomm")
                except Exception as e:
                    self._logger.error(f"Failed to publish {part} state: {e}")

    def dexcomm_listener(self, msg: dict, *, part: str):
        """Dexcomm 제어 메시지를 수신하여 ROS2 /joint_command 토픽으로 전송합니다."""
        try:
            # JointCmdCodec.decode는 dict를 반환합니다: {'pos': [...], 'vel': [...]}
            positions = msg.get('pos', [])
            velocities = msg.get('vel', [])

            joint_names = self.joint_map.get(part)
            if not joint_names:
                self._logger.warning(f"Received command for unknown part: {part}")
                return

            if part == 'chassis_drive':
                # 양쪽 바퀴가 반대 방향으로 장착되어 있을 수 있으므로, 
                # 한쪽(예: 오른쪽 바퀴)의 회전 방향(부호)을 반전시킵니다.
                if positions and len(positions) == 2:
                    positions = [positions[0], -positions[1]]
                if velocities and len(velocities) == 2:
                    velocities = [velocities[0], -velocities[1]]

            # if not positions:
            #     self._logger.warning(f"Received command for {part} with no position data.")
            #     return

            js_msg = JointState()
            js_msg.header.stamp = self._node.get_clock().now().to_msg()
            js_msg.name = joint_names
            if positions:
                js_msg.position = [float(p) for p in positions]
            if velocities:
                js_msg.velocity = [float(v) for v in velocities]

            self.ros2_pub.publish(js_msg)
            self._logger.info(f"Published JointState command from {part} to ROS2.")

        except Exception as e:
            self._logger.error(f'Error processing Dexcomm data for part {part}: {e}\n{traceback.format_exc()}')


def main(args=None):
    parser = argparse.ArgumentParser(description='Dexcomm-ROS2 Bridge')
    parser.add_argument('--hand_type', type=str, default='',
                        help='Hand type: hand or gripper')
    parsed_args, remaining_args = parser.parse_known_args(args)

    rclpy.init(args=remaining_args)
    node = None
    bridge = None
    try:
        node = Node(NODE_NAME)
        bridge = DexcommToRos2Bridge(node, hand_type=parsed_args.hand_type)

        executor = MultiThreadedExecutor()
        executor.add_node(node)

        node.get_logger().info('Dexcomm-ROS2 Bridge starting...')
        executor.spin()

    except KeyboardInterrupt:
        print("KeyboardInterrupt, shutting down.")
    except Exception as e:
        print("--- An exception occurred in main ---")
        traceback.print_exc()
    finally:
        if bridge and hasattr(bridge, 'dexcomm_node'):
            bridge.dexcomm_node.close()
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
