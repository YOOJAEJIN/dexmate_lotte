"""Keyboard input handler for Robotis exoskeleton teleoperation.

Captures keyboard input and converts to mode change events and control commands.
"""

from typing import Dict, Callable, Optional, Any
from enum import Enum
from loguru import logger
import threading
import queue


class KeyboardEvent(Enum):
    """Keyboard event types."""
    PRESSED = "pressed"
    RELEASED = "released"


class RobotisExoKeyboardHandler:
    """Handles keyboard input for Robotis exoskeleton control.
    
    Provides:
    - Mode switching (head, torso, etc.)
    - Emergency controls
    - Direct command inputs
    
    Usage:
        handler = RobotisExoKeyboardHandler()
        handler.register_callback("h", lambda: print("Head mode toggled"))
        handler.start()
    """
    
    # Default key mappings (can be customized)
    DEFAULT_KEY_MAP = {
        # Mode toggles (to be defined by user)
        
        "t": "torso_toggle",       # Torso mode
        "y": "head_toggle",        # Head mode
        # "k": "base_toggle",        # Base mode
        # "l": "arm_toggle",         # Arm control toggle
        # "f": "fine_toggle",        # Fine adjustment
        "r": "recording_toggle",   # Recording
        "g": "gripper_mode_toggle", # Toggle gripper control mode (keyboard vs reader)
        
        # Torso keyboard control
        "a": "torso_down",         # Torso down
        "b": "torso_neutral",      # Torso neutral (center position)
        "c": "torso_up",           # Torso up
        "h": "home_shape",         # Home/L-shape position (torso + both arms)
        
        # Gripper control
        "o": "gripper_open",       # Open gripper (toggle)
        
        # Emergency
        "e": "estop_toggle",       # Emergency stop
        
        # Direct values (to be defined later)
        # These could be for continuous head/torso movement
    }
    
    def __init__(self, key_map: Optional[Dict[str, str]] = None):
        """Initialize keyboard handler.
        
        Args:
            key_map: Custom key to command mapping. Uses DEFAULT_KEY_MAP if None.
        """
        self.key_map = key_map or self.DEFAULT_KEY_MAP.copy()
        
        # Callbacks for each command
        self.callbacks: Dict[str, list[Callable]] = {}
        
        # Thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._key_queue: queue.Queue = queue.Queue()
        
        # Key state tracking
        self._pressed_keys = set()
        
        # Keyboard state tracking for gripper
        self._gripper_open_pressed = False
        self._gripper_close_pressed = False
        # Gripper toggle state: True = open, False = closed
        self._gripper_is_open = True  # Start with gripper open
        
        # Keyboard state tracking for torso
        self._torso_up_pressed = False
        self._torso_down_pressed = False
        self._torso_neutral_pressed = False
        self._home_shape_pressed = False  # H-key: L-shape position for torso + arms
        
        # Gripper control mode: True = keyboard mode, False = reader mode
        self._gripper_keyboard_mode = False  # Default to reader mode
        
        logger.info(f"Robotis keyboard handler initialized with {len(self.key_map)} key mappings (gripper mode: reader)")
    
    def register_callback(self, command: str, callback: Callable) -> None:
        """Register a callback for a command.
        
        Args:
            command: Command name (e.g., "head_toggle", "torso_toggle")
            callback: Function to call when command is triggered
            
        Example:
            handler.register_callback("head_toggle", lambda: print("Head mode!"))
        """
        if command not in self.callbacks:
            self.callbacks[command] = []
        self.callbacks[command].append(callback)
        logger.debug(f"Registered callback for command: {command}")
    
    def unregister_callback(self, command: str, callback: Callable) -> None:
        """Unregister a callback.
        
        Args:
            command: Command name
            callback: Callback function to remove
        """
        if command in self.callbacks:
            self.callbacks[command] = [
                cb for cb in self.callbacks[command] if cb != callback
            ]
            logger.debug(f"Unregistered callback for command: {command}")
    
    def _trigger_command(self, command: str) -> None:
        """Trigger all callbacks for a command.
        
        Args:
            command: Command name
        """
        if command not in self.callbacks:
            logger.debug(f"No callbacks registered for command: {command}")
            return
        
        for callback in self.callbacks[command]:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in callback for '{command}': {e}")
    
    def get_pressed_keys(self) -> set:
        """Get currently pressed keys.
        
        Returns:
            Set of currently pressed key names
        """
        return self._pressed_keys.copy()
    
    def update_key_state(self, key: str, pressed: bool) -> None:
        """Update the state of a key (for external input sources).
        
        Args:
            key: Key name
            pressed: True if key is pressed, False if released
        """
        if pressed:
            if key not in self._pressed_keys:
                self._pressed_keys.add(key)
                self._process_key_press(key)
        else:
            if key in self._pressed_keys:
                self._pressed_keys.discard(key)
                self._process_key_release(key)
    
    def _process_key_press(self, key: str) -> None:
        """Process a key press event.
        
        Args:
            key: Key that was pressed
        """
        if key in self.key_map:
            command = self.key_map[key]
            logger.info(f"Key '{key}' pressed → triggering command: {command}")
            
            # Handle gripper mode toggle
            if command == "gripper_mode_toggle":
                self._gripper_keyboard_mode = not self._gripper_keyboard_mode
                mode_name = "keyboard" if self._gripper_keyboard_mode else "reader"
                logger.warning(f"Gripper control mode switched to: {mode_name}")
                self._trigger_command(command)
                return
            
            # Track torso state
            if command == "torso_up":
                self._torso_up_pressed = True
                self._torso_down_pressed = False
                self._torso_neutral_pressed = False
            elif command == "torso_down":
                self._torso_down_pressed = True
                self._torso_up_pressed = False
                self._torso_neutral_pressed = False
            elif command == "torso_neutral":
                self._torso_neutral_pressed = True
                self._torso_up_pressed = False
                self._torso_down_pressed = False
            elif command == "home_shape":
                self._home_shape_pressed = True
            
            # Track gripper state (only in keyboard mode) - o toggles open/closed
            if self._gripper_keyboard_mode:
                if command == "gripper_open":
                    # Toggle gripper state
                    self._gripper_is_open = not self._gripper_is_open
                    if self._gripper_is_open:
                        self._gripper_open_pressed = True
                        self._gripper_close_pressed = False
                        logger.info("Gripper toggled to OPEN")
                    else:
                        self._gripper_close_pressed = True
                        self._gripper_open_pressed = False
                        logger.info("Gripper toggled to CLOSED")
                elif command == "gripper_close":
                    # p key is not used in toggle mode
                    pass
            
            self._trigger_command(command)
        else:
            logger.debug(f"Key '{key}' pressed but not in key map")
    
    def _process_key_release(self, key: str) -> None:
        """Process a key release event.
        
        Args:
            key: Key that was released
        """
        if key in self.key_map:
            command = self.key_map[key]
            logger.debug(f"Key '{key}' released")
            
            # Reset torso state
            if command == "torso_up":
                self._torso_up_pressed = False
            elif command == "torso_down":
                self._torso_down_pressed = False
            elif command == "torso_neutral":
                self._torso_neutral_pressed = False
            elif command == "home_shape":
                self._home_shape_pressed = False
            
            # Gripper toggle state is maintained (don't reset)
            # The gripper will stay in the current state until o is pressed again
            if self._gripper_keyboard_mode:
                if command == "gripper_open":
                    # Keep the current toggle state
                    pass
                elif command == "gripper_close":
                    # p key is not used in toggle mode
                    pass
    
    def set_key_mapping(self, key: str, command: str) -> None:
        """Set or update a key mapping.
        
        Args:
            key: Key name
            command: Command name to trigger
            
        Example:
            handler.set_key_mapping("h", "head_toggle")
        """
        old_command = self.key_map.get(key)
        self.key_map[key] = command
        logger.info(f"Key mapping updated: '{key}' → '{command}' (was: {old_command})")
    
    def remove_key_mapping(self, key: str) -> None:
        """Remove a key mapping.
        
        Args:
            key: Key name to remove
        """
        if key in self.key_map:
            command = self.key_map.pop(key)
            logger.info(f"Key mapping removed: '{key}' (was mapped to '{command}')")
    
    def get_key_map(self) -> Dict[str, str]:
        """Get current key mappings.
        
        Returns:
            Dictionary of key → command mappings
        """
        return self.key_map.copy()
    
    def reset_key_map(self) -> None:
        """Reset to default key mappings."""
        self.key_map = self.DEFAULT_KEY_MAP.copy()
    
    def get_current_state(self) -> Dict[str, bool]:
        """Get current keyboard state for gripper and torso.
        
        Returns:
            Dictionary with gripper_open, gripper_close, torso_up, torso_down, torso_neutral flags
        """
        return {
            "gripper_open": self._gripper_open_pressed if self._gripper_keyboard_mode else False,
            "gripper_close": self._gripper_close_pressed if self._gripper_keyboard_mode else False,
            "gripper_keyboard_mode": self._gripper_keyboard_mode,
            "torso_up": self._torso_up_pressed,
            "torso_down": self._torso_down_pressed,
            "torso_neutral": self._torso_neutral_pressed,
            "home_shape": self._home_shape_pressed,
        }
    
    def get_gripper_mode(self) -> str:
        """Get current gripper control mode.
        
        Returns:
            "keyboard" or "reader"
        """
        return "keyboard" if self._gripper_keyboard_mode else "reader"
    
    def start(self) -> None:
        """Start listening for keyboard input.
        
        Note: Actual keyboard listening implementation depends on platform:
        - Linux/Unix: Can use pynput or similar
        - Interactive mode: Can use input() with threading
        
        For now, this is a placeholder. Users should call update_key_state()
        from their input source (e.g., pynput listener, raw input, etc.)
        """
        self._running = True
        logger.info("Keyboard handler started (waiting for key updates via update_key_state)")
    
    def stop(self) -> None:
        """Stop listening for keyboard input."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        logger.info("Keyboard handler stopped")
    
    def is_running(self) -> bool:
        """Check if handler is running.
        
        Returns:
            True if handler is active
        """
        return self._running
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics.
        
        Returns:
            Dictionary with current state info
        """
        return {
            "running": self._running,
            "pressed_keys": list(self._pressed_keys),
            "registered_commands": list(self.callbacks.keys()),
            "key_mappings_count": len(self.key_map),
            "gripper_mode": self.get_gripper_mode(),
        }


# Optional: Pynput integration helper.
#
# pynput is imported lazily *inside* create_pynput_listener(), never at module
# level: pynput's X11 backend opens a connection to the X display at import time,
# which blocks forever when DISPLAY points to a display that never accepts
# clients (e.g. mutter's private Xwayland display ":1"). This module is imported
# by command_processor / robot_controller in every teleop mode, so importing it
# must not touch the display.
def create_pynput_listener(handler: RobotisExoKeyboardHandler) -> "keyboard.Listener":
    """Create a pynput keyboard listener that updates the handler.
    
    Args:
        handler: RobotisExoKeyboardHandler instance
        
    Returns:
        pynput.keyboard.Listener instance
        
    Raises:
        ImportError: pynput is not installed.
        
    Usage:
        handler = RobotisExoKeyboardHandler()
        listener = create_pynput_listener(handler)
        listener.start()
    """
    from pynput import keyboard  # lazy import: connects to the X display

    def on_press(key):
        try:
            # Handle regular keys
            if hasattr(key, 'char'):
                handler.update_key_state(key.char, True)
            else:
                # Handle special keys
                key_name = key.name if hasattr(key, 'name') else str(key)
                handler.update_key_state(key_name, True)
        except Exception as e:
            logger.debug(f"Error processing key press: {e}")
    
    def on_release(key):
        try:
            if hasattr(key, 'char'):
                handler.update_key_state(key.char, False)
            else:
                key_name = key.name if hasattr(key, 'name') else str(key)
                handler.update_key_state(key_name, False)
        except Exception as e:
            logger.debug(f"Error processing key release: {e}")
    
    return keyboard.Listener(on_press=on_press, on_release=on_release)


if __name__ == "__main__":
    # Example usage
    handler = RobotisExoKeyboardHandler()
    
    # Register callbacks
    handler.register_callback("head_toggle", lambda: logger.info("→ HEAD MODE"))
    handler.register_callback("torso_toggle", lambda: logger.info("→ TORSO MODE"))
    handler.register_callback("base_toggle", lambda: logger.info("→ BASE MODE"))
    handler.register_callback("estop_toggle", lambda: logger.warning("⚠ ESTOP"))
    
    # Simulate key presses
    print("Simulating key presses...")
    handler.start()
    handler.update_key_state("h", True)
    handler.update_key_state("h", False)
    handler.update_key_state("t", True)
    handler.update_key_state("t", False)
    
    print("\nCurrent stats:", handler.get_stats())
    print("Key mappings:", handler.get_key_map())
