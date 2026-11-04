#!/usr/bin/env python3
"""Test script to publish exo_joints topic without physical ExoSkeleton hardware.

This script publishes a test ExoJointData message with the right arm's first joint
set to 0.2 radians.
"""

import sys
import time
from dataclasses import asdict
from typing import Optional
import tyro

from dexcomm import Node
from dexcomm.codecs import DictDataCodec
from dexcomm import RateLimiter
from omniteleop.common import ExoJointData, get_config
from omniteleop.common.logging import setup_logging
from omniteleop.common.debug_display import get_debug_display
from loguru import logger


class TestArmReader():
    """Test arm reader that publishes mock exo joint data."""

    def __init__(
        self,
        namespace: str = "",
        publish_rate: int = 40,
        topic: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        """Initialize test arm reader.

        Args:
            namespace: Optional namespace prefix
            publish_rate: Publishing rate in Hz
            topic: Override topic (uses config if None)
            debug: Enable debug output
        """
        self.node = Node(name="test_arm", namespace=namespace)

        # Load configuration
        config = get_config()
        self.topic = topic or config.get_topic("exo_joints")
        self.publish_rate = publish_rate

        # Publisher
        self.publisher = None

        # Rate limiter
        self.rate_limiter = RateLimiter(self.publish_rate)

        logger.info(f"Test arm reader configured at {self.publish_rate}Hz")

    def initialize(self) -> bool:
        """Initialize Zenoh publisher.

        Returns:
            True if initialization successful
        """
        try:
            self.publisher = self.node.create_publisher(
                self.topic, encoder=DictDataCodec.encode
            )
            logger.success("Initialized test arm reader")
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def run(self) -> None:
        """Main publishing loop."""
        resolved_topic = self.node.resolve_topic(self.topic)
        logger.info(f"Started publishing to {resolved_topic} at {self.publish_rate}Hz")
        logger.info("Moving left arm joints 1-7 sequentially by 0.2 radians for 1 second each.")

        try:
            for i in range(7):
                # Create test data with the i-th joint of the left arm at 0.2 radians
                # All other joints set to 0.0
                left_pos = [0.0] * 7
                left_pos[i] = 0.2
                
                data = ExoJointData(
                    timestamp_ns=time.time_ns(),
                    left_arm_pos=left_pos,
                    left_arm_vel=[],
                    right_arm_pos=[0.0] * 7,
                    right_arm_vel=[],
                )

                # Publish
                logger.info(f"Moving left arm joint {i+1} to 0.2 radians.")
                self.publisher.publish(asdict(data))

                # Sleep for 1 second
                time.sleep(1)
            
            # After the sequence, reset the arm position to all zeros
            logger.info("Sequence finished. Resetting arm position.")
            data = ExoJointData(
                timestamp_ns=time.time_ns(),
                left_arm_pos=[0.0] * 7,
                left_arm_vel=[],
                right_arm_pos=[0.0] * 7,
                right_arm_vel=[],
            )
            self.publisher.publish(asdict(data))

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        self.node.shutdown()
        logger.info("Test arm reader stopped")


def main(
    namespace: str = "",
    publish_rate: int = 40,
    topic: Optional[str] = None,
    debug: bool = False,
):
    """Main entry point for test arm reader.

    Args:
        namespace: Optional namespace prefix
        publish_rate: Publishing rate in Hz
        topic: Override topic (uses config if None)
        debug: Enable debug output
    """
    logger = setup_logging(debug)
    logger.info(
        f"Starting Test Arm Reader{f' (namespace: {namespace})' if namespace else ''}"
    )

    reader = TestArmReader(
        namespace=namespace,
        publish_rate=publish_rate,
        topic=topic,
        debug=debug,
    )

    if not reader.initialize():
        logger.error("Failed to initialize test arm reader")
        return 1

    try:
        reader.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        reader.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(tyro.cli(main))
