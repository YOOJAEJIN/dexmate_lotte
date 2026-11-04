#!/usr/bin/env python3
"""
A script to publish robot joint commands via Zenoh, moving one joint at a time.
"""

import time
import numpy as np
from dexcomm import Node
from dexcomm.codecs import DictDataCodec

# Define the topic for safe robot commands
ROBOT_COMMANDS_TOPIC = "robot/safe_commands"

def main():
    """
    Moves all 12 joints of the left and right hand together every 3 seconds.
    """
    # Initialize the starting joint positions for the left hand and right hand
    left_hand_pos = np.array([0.0] * 12)
    right_hand_pos = np.array([0.0] * 12)

    # Create a Zenoh node
    node = Node(name="joint_mover_sequential")

    # Create a publisher for the robot commands topic
    safe_command_pub = node.create_publisher(
        ROBOT_COMMANDS_TOPIC, encoder=DictDataCodec.encode
    )

    print(f"Starting sequential joint movement...")
    print(f"Publishing to topic: {ROBOT_COMMANDS_TOPIC}")

    try:
        # Loop to modify all 12 joints every 3 seconds
        for i in range(100):
            # Wait for 3 seconds before the next movement
            time.sleep(3)

            # Toggle all 12 joints between 0.0 and 1.0
            target_val = 1.0 if i % 2 == 0 else 0.0
            left_hand_pos = np.array([target_val] * 12)
            right_hand_pos = np.array([target_val] * 12)
            print(f"\nMoving all 12 joints to {target_val}...")

            # Construct the command message
            command_dict = {
                "timestamp_ns": time.time_ns(),
                "components": {
                    "left_hand": {
                        "pos": left_hand_pos.tolist(),
                        "mode": "absolute",
                    },
                    "right_hand": {
                        "pos": right_hand_pos.tolist(),
                        "mode": "absolute",
                    },
                    "safety_flags": {
                        "emergency_stop": False,
                        "exit_requested": False,
                    },
                },
            }

            # Publish the command
            safe_command_pub.publish(command_dict)

            print(f"Published: {command_dict}")

        print("\nFinished all joint movements.")

    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Cleanly shut down the Zenoh node
        print("Shutting down Zenoh node.")
        node.shutdown()
        print("Exiting.")

if __name__ == "__main__":
    main()
