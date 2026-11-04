#!/usr/bin/env python3
import sys
import time
import ast
import os
from typing import Optional
import tyro

from dexcomm import Node
from dexcomm.codecs import DictDataCodec
from dexcomm import RateLimiter
from omniteleop.common.logging import setup_logging
from loguru import logger


class TestVrPose:
    """Test vr pose that publishes mock vr pose data."""

    def __init__(
        self,
        namespace: str = "",
        publish_rate: int = 20,
        topic: Optional[str] = "vr/pose",
    ) -> None:
        """Initialize test arm reader.

        Args:
            namespace: Optional namespace prefix
            publish_rate: Publishing rate in Hz
            topic: Override topic (uses config if None)
        """
        self.node = Node(name="test_vr_pose", namespace=namespace)

        self.topic = topic
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
            logger.success("Initialized test vr pose publisher")
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def run(self) -> None:
        """Main publishing loop."""
        resolved_topic = self.node.resolve_topic(self.topic)
        logger.info(f"Started publishing to {resolved_topic} at {self.publish_rate}Hz")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(current_dir, "vr_pose.txt")
        
        if not os.path.exists(data_file):
            logger.error(f"Data file not found at {data_file}")
            return

        try:
            with open(data_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        # Safely parse the dictionary string
                        data = ast.literal_eval(line.strip())
                        
                        # Update timestamp to current time for realistic playback
                        data['timestamp_ns'] = time.time_ns()
                        
                        # Publish
                        self.publisher.publish(data)
                        
                        # Sleep to maintain publish rate
                        self.rate_limiter.sleep()
                        
                    except Exception as e:
                        logger.warning(f"Failed to parse or publish line: {e}")
                        continue
                        
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        self.node.shutdown()
        logger.info("Test vr pose publisher stopped")


def main(
    namespace: str = "",
    publish_rate: int = 20,
    topic: Optional[str] = "vr/pose",
    debug: bool = False,
):
    """Main entry point for test arm reader.

    Args:
        namespace: Optional namespace prefix
        publish_rate: Publishing rate in Hz
        topic: Override topic (uses config if None)
        debug: Enable debug output
    """
    setup_logging(debug)
    logger.info(
        f"Starting Test VR Pose Publisher{f' (namespace: {namespace})' if namespace else ''}"
    )

    reader = TestVrPose(
        namespace=namespace,
        publish_rate=publish_rate,
        topic=topic,
    )

    if not reader.initialize():
        logger.error("Failed to initialize test vr pose publisher")
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
