import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from dexcomm import Node as dexNode
from dexcomm.codecs import DictDataCodec

ROBOT_COMMANDS_TOPIC = "robot/safe_commands"

class XsensToVega(Node):
    def __init__(self):
        super().__init__('xsens_to_vega')
        
        # Xsens 조인트 상태 구독
        self.subscription = self.create_subscription(
            JointState, 
            '/joint_states', 
            self.listener_callback, 
            10)

        self.dex_node = dexNode(name="dex_node")

        self.dexPub = self.dex_node.create_publisher(
            ROBOT_COMMANDS_TOPIC, encoder=DictDataCodec.encode
        )
            
        self.get_logger().info('Xsens to Vega conversion node has been started.')

    def listener_callback(self, msg):
        # 메시지를 이름 기반 딕셔너리로 변환
        joint_map = dict(zip(msg.name, msg.position))
        
        try:
            # 필수 조인트 확인
            left_joints = [
                'skeleton_left_shoulder_y', 'skeleton_left_shoulder_x', 'skeleton_left_shoulder_z',
                'skeleton_left_elbow_x',
                'skeleton_left_wrist_y', 'skeleton_left_wrist_x', 'skeleton_left_wrist_z'
            ]
            right_joints = [
                'skeleton_right_shoulder_y', 'skeleton_right_shoulder_x', 'skeleton_right_shoulder_z',
                'skeleton_right_elbow_x',
                'skeleton_right_wrist_y', 'skeleton_right_wrist_x', 'skeleton_right_wrist_z'
            ]
            
            missing_joints = [j for j in left_joints + right_joints if j not in joint_map]
            if missing_joints:
                self.get_logger().warning(f'Missing joints in message: {missing_joints}')

            # 1. 왼팔 7-DOF 추출
            left_arm_7dof = [
                joint_map.get('skeleton_left_shoulder_y', 0.0), # J1: Shoulder Yaw
                joint_map.get('skeleton_left_shoulder_x', 0.0), # J2: Shoulder Pitch
                joint_map.get('skeleton_left_shoulder_z', 0.0), # J3: Shoulder Roll
                joint_map.get('skeleton_left_elbow_x', 0.0),    # J4: Elbow Pitch
                joint_map.get('skeleton_left_wrist_y', 0.0),    # J5: Wrist Yaw
                joint_map.get('skeleton_left_wrist_x', 0.0),    # J6: Wrist Pitch
                joint_map.get('skeleton_left_wrist_z', 0.0)     # J7: Wrist Roll
            ]

            # 2. 오른팔 7-DOF 추출
            right_arm_7dof = [
                joint_map.get('skeleton_right_shoulder_y', 0.0), # J1
                joint_map.get('skeleton_right_shoulder_x', 0.0), # J2
                joint_map.get('skeleton_right_shoulder_z', 0.0), # J3
                joint_map.get('skeleton_right_elbow_x', 0.0),    # J4
                joint_map.get('skeleton_right_wrist_y', 0.0),    # J5
                joint_map.get('skeleton_right_wrist_x', 0.0),    # J6
                joint_map.get('skeleton_right_wrist_z', 0.0)     # J7
            ]

            # 3. 통합 명령 메시지 생성 및 발행
            combined_command = {
                "timestamp_ns": time.time_ns(),
                "components": {
                    "left_arm": {
                        "pos": left_arm_7dof,
                        "mode": "absolute",
                    },
                    "right_arm": {
                        "pos": right_arm_7dof,
                        "mode": "absolute",
                    },
                    "safety_flags": {
                        "emergency_stop": False,
                        "exit_requested": False,
                    },
                },
            }

            # Publish the integrated command
            self.dexPub.publish(combined_command)

        except Exception as e:
            self.get_logger().error(f'Error during conversion: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = XsensToVega()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
