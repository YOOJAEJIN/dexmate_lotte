import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock dexcomm before importing CommonHandler
mock_dexcomm = MagicMock()
sys.modules['dexcomm'] = mock_dexcomm
sys.modules['dexcomm.codecs'] = MagicMock()
sys.modules['dexcomm.utils'] = MagicMock()

from omniteleop.follower.input_handlers.common_handler import CommonHandler
from omniteleop.follower.input_handlers.base_handler import CommandMode, RobotCommand, ArmCommandType

class TestCommonHandler(unittest.TestCase):
    def setUp(self):
        self.config = {
            "topics": {
                "common_input": "test/common/input"
            }
        }
        self.handler = CommonHandler(self.config)

    def test_process_inputs_standard_components(self):
        # Prepare test data
        test_data = {
            "components": {
                "torso": {"mode": "absolute", "pos": [1.0, 2.0, 3.0]},
                "left_hand": {"pos": [0.5]},
                "left_arm": {"mode": "relative", "pos": [0.1] * 7}
            }
        }
        
        # Simulate command received
        self.handler._on_command_received(test_data)
        
        # Process inputs
        command = self.handler.process_inputs()
        
        self.assertIsNotNone(command)
        self.assertIsInstance(command, RobotCommand)
        
        # Check torso
        self.assertEqual(command.input_components["torso"]["mode"], CommandMode.ABSOLUTE)
        self.assertEqual(command.input_components["torso"]["pos"], [1.0, 2.0, 3.0])
        
        # Check left_hand (uses default mode 'absolute')
        self.assertEqual(command.input_components["left_hand"]["mode"], CommandMode.ABSOLUTE)
        self.assertEqual(command.input_components["left_hand"]["pos"], [0.5])
        
        # Check left_arm
        self.assertEqual(command.input_components["left_arm"]["mode"], CommandMode.RELATIVE)
        self.assertEqual(command.input_components["left_arm"]["pos"], [0.1] * 7)
        
        # Check components not in test_data are not in command
        self.assertNotIn("head", command.input_components)
        self.assertNotIn("right_hand", command.input_components)

    def test_process_inputs_chassis(self):
        test_data = {
            "components": {
                "chassis": {"vx": 1.0, "vy": -1.0, "wz": 0.5}
            }
        }
        self.handler._on_command_received(test_data)
        command = self.handler.process_inputs()
        
        self.assertEqual(command.input_components["chassis"]["vx"], 1.0)
        self.assertEqual(command.input_components["chassis"]["vy"], -1.0)
        self.assertEqual(command.input_components["chassis"]["wz"], 0.5)

    def test_invalid_mode_fallback(self):
        test_data = {
            "components": {
                "torso": {"mode": "invalid_mode", "pos": [1.0, 2.0, 3.0]}
            }
        }
        self.handler._on_command_received(test_data)
        
        # This should log a warning but not crash, falling back to default 'relative'
        command = self.handler.process_inputs()
        
        self.assertEqual(command.input_components["torso"]["mode"], CommandMode.RELATIVE)
        self.assertEqual(command.input_components["torso"]["pos"], [1.0, 2.0, 3.0])

    def test_process_inputs_cartesian_translation_jog(self):
        test_data = {
            "components": {
                "left_arm": {
                    "command_type": "ee_pose",
                    "delta_xyz": [0.0, 0.0, 0.005],
                }
            }
        }
        self.handler._on_command_received(test_data)
        command = self.handler.process_inputs()

        left_arm = command.input_components["left_arm"]
        self.assertEqual(left_arm["command_type"], ArmCommandType.EE_POSE)
        self.assertEqual(left_arm["delta_xyz"], [0.0, 0.0, 0.005])
        self.assertNotIn("delta_R", left_arm)

    def test_process_inputs_cartesian_rotation_jog(self):
        delta_R = [
            [1.0, 0.0, 0.0],
            [0.0, 0.99875026, -0.04997917],
            [0.0, 0.04997917, 0.99875026],
        ]
        test_data = {
            "components": {
                "right_arm": {
                    "command_type": "ee_pose",
                    "delta_xyz": [0.0, 0.0, 0.0],
                    "delta_R": delta_R,
                    "rot_frame": "ee",
                }
            }
        }
        self.handler._on_command_received(test_data)
        command = self.handler.process_inputs()

        right_arm = command.input_components["right_arm"]
        self.assertEqual(right_arm["command_type"], ArmCommandType.EE_POSE)
        self.assertEqual(right_arm["delta_xyz"], [0.0, 0.0, 0.0])
        self.assertEqual(right_arm["delta_R"], delta_R)
        self.assertEqual(right_arm["rot_frame"], "ee")

    def test_empty_command(self):
        self.assertIsNone(self.handler.process_inputs())

if __name__ == '__main__':
    unittest.main()
