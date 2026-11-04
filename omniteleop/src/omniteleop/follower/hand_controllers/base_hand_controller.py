from abc import ABC, abstractmethod

class BaseHandController(ABC):
    """Abstract base class for hand controllers."""

    @abstractmethod
    def activate(self):
        """Initialize and activate the hand hardware."""
        pass

    @abstractmethod
    def set_position(self, pos: list[float]):
        """Set hand position (list of floats, 0.0 to 1.0)."""
        pass

    @abstractmethod
    def open(self):
        """Open the hand."""
        pass

    @abstractmethod
    def close(self):
        """Close the hand."""
        pass

    @abstractmethod
    def start_tracking(self, interval: float = 0.05):
        """Start monitoring hand status in a separate thread."""
        pass

    @abstractmethod
    def stop_tracking(self):
        """Stop the monitoring thread."""
        pass

    @abstractmethod
    def get_status(self) -> list[float]:
        """Get the latest status of the hand."""
        pass
