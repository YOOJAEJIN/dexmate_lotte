import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional, Any

try:
    from mcap_protobuf.writer import Writer as ProtobufWriter
    from mcap.writer import CompressionType
except ImportError:
    raise ImportError(
        "MCAP libraries not installed. Install with: pip install mcap mcap-protobuf-support"
    )

# Import generated protobuf classes (after compiling .proto files)
try:
    from .episode import RobotState, RobotAction, PolicyStepAnnotation
except ImportError:
    raise ImportError(
        "Protobuf classes not found. Compile .proto files first: "
        "cd internal/we_record/mcap_utils/proto && ./compile_proto.sh"
    )

from .metadata import EpisodeMetadata

class TeleopMcapWriter:
    """
    MCAP writer for teleoperation data.
    Writes state+action to one file and each camera to its own separate file.
    Uses EpisodeMetadata for progressive metadata building.
    """

    def __init__(
        self,
        output_path: Path,
        task_name: str,
        task_language_instruction: str,
        operator: str = "unknown",
        record_rate_hz: float = 20.0,
        init_pos: Optional[Dict] = None,
        record_components: Optional[Dict[str, bool]] = None,
        compression: CompressionType = CompressionType.ZSTD,
        image_encoding: str = "jpeg",  # 'rgb8' or 'jpeg'
        jpeg_quality: int = 90,
        source_mode: Optional[str] = None,
        policy_tag: Optional[str] = None,
        action_chunk_threshold: Optional[float] = None,
    ):
        """
        Initialize MCAP writer.

        Args:
            output_path: Episode folder path.
            task_name: Name of the task being recorded.
            task_language_instruction: Natural language instruction for the task.
            operator: Name of the operator.
            record_rate_hz: Recording rate in Hz.
            init_pos: Initial joint positions from config (may include inactive joints).
            record_components: Dict of component names to enabled status.
            image_encoding: Image encoding format ('rgb8' or 'jpeg')
            jpeg_quality: JPEG quality (1-100) if encoding is 'jpeg'
        """
        self.episode_dir = Path(output_path)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.image_encoding = image_encoding
        self.jpeg_quality = jpeg_quality

        # Initialize metadata with user-provided fields
        self.metadata = EpisodeMetadata(
            task_name=task_name,
            task_language_instruction=task_language_instruction,
            operator=operator,
            record_rate_hz=record_rate_hz,
            init_pos=init_pos,
            record_components=record_components,
            source_mode=source_mode,
            policy_tag=policy_tag,
            action_chunk_threshold=action_chunk_threshold,
        )
        self.metadata.start()  # Sets start_time, initializes counts

        # Accumulate per-step inference durations for avg_inference_time_ms
        self._policy_step_times_ns: list[int] = []

        # Generate state/action file path inside episode folder
        self.state_action_path = self.episode_dir / "state_action.mcap"

        # Create state/action writer
        self._state_action_file = open(self.state_action_path, "wb")
        self._state_action_writer = ProtobufWriter(
            self._state_action_file, compression=compression
        )

    def write_state(
        self,
        state: RobotState,
        timestamp_ns: int,
        publish_time_ns: Optional[int] = None,
    ):
        """
        Write robot state to MCAP.

        Args:
            state: RobotState dataclass instance
            timestamp_ns: Timestamp in nanoseconds (used as log_time)
            publish_time_ns: Sensor timestamp in nanoseconds (used as publish_time).
                            Falls back to timestamp_ns if None.
        """
        self._state_action_writer.write_message(
            topic="/robot/state",
            message=state.to_proto(),
            log_time=timestamp_ns,
            publish_time=publish_time_ns
            if publish_time_ns is not None
            else timestamp_ns,
        )
        self.metadata.state_count += 1

    def write_action(
        self,
        action: RobotAction,
        timestamp_ns: int,
        publish_time_ns: Optional[int] = None,
    ):
        """
        Write robot action to MCAP.

        Args:
            action: RobotAction dataclass instance
            timestamp_ns: Timestamp in nanoseconds (used as log_time)
            publish_time_ns: Sensor timestamp in nanoseconds (used as publish_time).
                            Falls back to timestamp_ns if None.
        """
        self._state_action_writer.write_message(
            topic="/robot/action",
            message=action.to_proto(),
            log_time=timestamp_ns,
            publish_time=publish_time_ns
            if publish_time_ns is not None
            else timestamp_ns,
        )
        self.metadata.action_count += 1

    def write_policy_step(self, annotation: PolicyStepAnnotation, timestamp_ns: int):
        """Write policy step annotation to MCAP.

        Args:
            annotation: PolicyStepAnnotation dataclass instance.
            timestamp_ns: Timestamp in nanoseconds.
        """
        self._state_action_writer.write_message(
            topic="/policy/step",
            message=annotation.to_proto(),
            log_time=timestamp_ns,
            publish_time=timestamp_ns,
        )
        # Accumulate inference duration for episode-level avg
        if (
            annotation.inference_begin_ns is not None
            and annotation.inference_end_ns is not None
        ):
            self._policy_step_times_ns.append(
                int(annotation.inference_end_ns) - int(annotation.inference_begin_ns)
            )

    def close(self, success: bool = True, extra_metadata: Optional[Dict] = None):
        """
        Close all MCAP writers, finalize files, and save metadata.

        Args:
            success: Whether the episode was successful
            extra_metadata: Optional dict of extra runtime metadata (e.g., episode_num, avg_rate)
        """
        # Compute avg inference time from accumulated per-step data
        if self._policy_step_times_ns:
            self.metadata.avg_inference_time_ms = (
                float(np.mean(self._policy_step_times_ns)) / 1e6
            )

        # Finalize metadata
        self.metadata.finalize(success)
        if extra_metadata:
            self.metadata.extra = extra_metadata
        # self.metadata.save(self.episode_dir / "metadata.json")
        self.metadata.save(self.episode_dir / "info.json")

        # Close state/action writer
        self._state_action_writer.finish()
        self._state_action_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If exception occurred, mark as unsuccessful
        success = exc_type is None
        self.close(success=success)
        return False
