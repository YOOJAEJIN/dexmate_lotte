#!/usr/bin/env python3
"""
Xsens ROS2 TF Reader
Reads Xsens TF data from ROS2 and publishes it to Zenoh topic "xsens/ros".
"""

import time
import re
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from dataclasses import asdict

import rclpy
from rclpy.node import Node as RosNode
from rclpy.time import Time
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener

from dexcomm import Node as DexNode
from dexcomm.codecs import DictDataCodec
from omniteleop.common import get_config
from omniteleop.common.schemas import XsensRosData, XsensArmData

RATE_HZ = 40.0
TF_TIMEOUT = 0.05

class XsensReader(RosNode):
    def __init__(self, node_name="xsens_reader", zenoh_topic="xsens/ros"):
        super().__init__(node_name)

        # Load configuration
        self.config = get_config()
        
        # Initialize ROS2 TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Initialize DexComm Zenoh publisher
        self.dex_node = DexNode(name="xsens_reader_zenoh")
        self.topic = self.config.get_topic("xsens_ros")
        self.publisher = self.dex_node.create_publisher(
            self.topic, encoder=DictDataCodec.encode
        )
        
        self.pelvis = None
        self.frames = {
            "left":  {"upper_arm": None, "forearm": None, "hand": None},
            "right": {"upper_arm": None, "forearm": None, "hand": None},
        }
        self.frames_resolved = False
        
        # Timer to run at RATE_HZ
        self.timer = self.create_timer(1.0 / RATE_HZ, self.on_timer)
        self.get_logger().info(f"XsensReader initialized. Publishing to Zenoh topic: '{zenoh_topic}' at {RATE_HZ} Hz")

    # ---- TF 트리에서 프레임 자동 탐지 ----
    def resolve_frames(self):
        names = re.findall(r"Frame (\S+) exists", self.tf_buffer.all_frames_as_string())
        if not names:
            return False

        def pick(words, side=None):
            cands = [n for n in names if (side is None or side in n.lower())]
            for w in words:
                for n in cands:
                    if w in n.lower():
                        return n
            return None

        if not self.pelvis:
            self.pelvis = pick(["pelvis", "hip", "t8", "spine", "root"])
        want = {"hand": ["hand", "wrist"],
                "forearm": ["forearm", "lower_arm", "lowerarm", "elbow"],
                "upper_arm": ["upper_arm", "upperarm", "shoulder"]}
        for side in ("left", "right"):
            for key, words in want.items():
                if not self.frames[side][key]:
                    self.frames[side][key] = pick(words, side)

        self.get_logger().info(f"[TF] pelvis={self.pelvis} frames={self.frames}")
        missing = [f"{s}/{k}" for s in self.frames for k in self.frames[s]
                   if self.frames[s][k] is None]
        if not self.pelvis or missing:
            self.get_logger().warning(
                f"Frame auto-detection failed ({missing}). Please ensure TF is active."
            )
            return False
        return True

    def lookup_pos(self, frame):
        """Get position of frame relative to pelvis."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.pelvis, frame, Time(), timeout=Duration(seconds=TF_TIMEOUT))
        except Exception:
            return None
        t = tf.transform.translation
        return np.array([t.x, t.y, t.z])

    def lookup_rot(self, frame):
        """Get rotation matrix of frame relative to pelvis."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.pelvis, frame, Time(), timeout=Duration(seconds=TF_TIMEOUT))
        except Exception:
            return None
        q = tf.transform.rotation
        return Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

    def on_timer(self):
        if not self.frames_resolved:
            if not self.resolve_frames():
                return
            self.frames_resolved = True

        left_arm = XsensArmData()
        right_arm = XsensArmData()

        # Process left arm
        l_sh = self.lookup_pos(self.frames["left"]["upper_arm"])
        l_el = self.lookup_pos(self.frames["left"]["forearm"])
        l_ha = self.lookup_pos(self.frames["left"]["hand"])
        l_rot = self.lookup_rot(self.frames["left"]["hand"])

        if l_sh is not None and l_el is not None and l_ha is not None:
            left_arm.upper_arm_pos = l_sh.tolist()
            left_arm.forearm_pos = l_el.tolist()
            left_arm.hand_pos = l_ha.tolist()
            if l_rot is not None:
                left_arm.hand_rot = l_rot.flatten().tolist()
            left_arm.valid = True

        # Process right arm
        r_sh = self.lookup_pos(self.frames["right"]["upper_arm"])
        r_el = self.lookup_pos(self.frames["right"]["forearm"])
        r_ha = self.lookup_pos(self.frames["right"]["hand"])
        r_rot = self.lookup_rot(self.frames["right"]["hand"])

        if r_sh is not None and r_el is not None and r_ha is not None:
            right_arm.upper_arm_pos = r_sh.tolist()
            right_arm.forearm_pos = r_el.tolist()
            right_arm.hand_pos = r_ha.tolist()
            if r_rot is not None:
                right_arm.hand_rot = r_rot.flatten().tolist()
            right_arm.valid = True

        # Combine and publish
        msg = XsensRosData(
            timestamp_ns=time.time_ns(),
            left=left_arm,
            right=right_arm
        )
        self.publisher.publish(asdict(msg))

def main(args=None):
    rclpy.init(args=args)
    node = XsensReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
