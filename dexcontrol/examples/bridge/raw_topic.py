# This is a low-level example that requires manual configuration.
# Before running, you must set the ZENOH_CONFIG environment variable.
# For example:
# export ZENOH_CONFIG=/home/ubuntu/.dexmate/comm/zenoh/zenoh_config.json5
# python examples/bridge/raw_topic.py

import time
from dexcomm import Node
from dexcomm.codecs import DictDataCodec

class RobotController:
    def __init__(self):
        # Create node with namespace
        self.node = Node("controller", namespace="")
        self.last_print_time = time.time()

        # Publishers and subscribers are namespaced
        # self.cmd_pub = self.node.create_publisher(f"state/head", encoder=JointStateCodec.encode)
        # self.node.create_subscriber("exo/joycon",
        #                                              self.on_exo_joycon,
        #                                              decoder=DictDataCodec.decode)
        self.node.create_subscriber("robot/safe_commands",
                                     self.on_commands,
                                     decoder=DictDataCodec.decode)
        self.node.create_subscriber("robot/joints",
                                     self.on_joints,
                                     decoder=DictDataCodec.decode)
        self.node.create_subscriber("recorder/control",
                                     self.on_recorder_control,
                                     decoder=DictDataCodec.decode)

        # Services
        # self.node.create_service("reset", self.handle_reset,
        #                          request_decoder=JointStateCodec.decode,
        #                          response_encoder=JointStateCodec.encode)

    def on_exo_joycon(self, msg):
        current_time = time.time()
        if current_time - self.last_print_time >= 0.5:
            print(f"exo/joycon topic: {msg}")
            self.last_print_time = current_time

    def on_commands(self, msg):
        current_time = time.time()
        if current_time - self.last_print_time >= 0.5:
            print(f"robot/safe_commands topic: {msg}")
            self.last_print_time = current_time

    def on_joints(self, msg):
        current_time = time.time()
        if current_time - self.last_print_time >= 2.0:
            print(f"robot/joints topic: {msg}")
            self.last_print_time = current_time

    def on_recorder_control(self, msg):
        print(f"recorder_control topic: {msg}")

    # def handle_reset(self, request):
    #     # Reset robot state
    #     return {"success": True}

    def run(self):
        self.node.spin()  # Process callbacks

# Usage
controller = RobotController()
controller.run()