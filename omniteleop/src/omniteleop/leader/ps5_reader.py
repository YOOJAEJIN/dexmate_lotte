#!/usr/bin/env python3
"""Leader program for reading PS5 (DualSense) controller inputs and publishing JoyCon-like events.

This module emulates the output format of `joycon_reader.py` by publishing `JoyConData`
to the same topic (default: config topic `exo_joycon`).
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from typing import Optional, List, Tuple

import tyro
from loguru import logger

try:
    import pygame
except ModuleNotFoundError as e:
    raise SystemExit(
        "pygame is required for omni-ps5.\n"
        "Install it with:\n"
        "  uv add pygame\n"
        "or\n"
        "  uv pip install pygame\n"
    ) from e

from dexcomm import Node
from dexcomm.codecs import DictDataCodec
from dexcomm import RateLimiter
from omniteleop.common import JoyConData, get_config
from omniteleop.common.logging import setup_logging
from omniteleop.common.debug_display import get_debug_display


class PS5JoyConEmulator():
    """Reads PS5 inputs and publishes JoyConData via Zenoh using Node interface."""

    def __init__(
        self,
        namespace: str = "",
        publish_rate: Optional[int] = None,
        topic: Optional[str] = None,
        deadzone: float = 0.1,
        debug: bool = False,
        device_index: int = 0,
        axis_lx: int = 0,
        axis_ly: int = 1,
        axis_rx: int = 3,
        axis_ry: int = 4,
        axis_l2: int = 2,
        axis_r2: int = 5,
        invert_ly: bool = True,
        invert_ry: bool = False,
        trigger_deadzone: float = 0.10,
        dump_buttons: bool = False,
        dump_axes: bool = False,
        dump_eps: float = 0.05,
        dump_heartbeat_s: float = 1.0,
    ):
        # Initialize Node (namespace is handled automatically by Node)
        self.node = Node(name="ps5_reader", namespace=namespace)

        config = get_config()
        self.publish_rate = publish_rate or config.get_rate("input_rate", 40)
        self.topic = topic or config.get_topic("exo_joycon")
        self.deadzone = deadzone
        self.trigger_deadzone = trigger_deadzone

        self.device_index = device_index
        self.axis_lx = axis_lx
        self.axis_ly = axis_ly
        self.axis_rx = axis_rx
        self.axis_ry = axis_ry
        self.axis_l2 = axis_l2
        self.axis_r2 = axis_r2
        self.invert_ly = invert_ly
        self.invert_ry = invert_ry

        self.dump_buttons = dump_buttons
        self.dump_axes = dump_axes
        self.dump_eps = dump_eps
        self.dump_heartbeat_s = dump_heartbeat_s
        self._last_heartbeat_ns: int = 0

        self._joy: Optional["pygame.joystick.Joystick"] = None
        self.publisher = None
        self.rate_limiter = RateLimiter(self.publish_rate)

        self._last_buttons: Optional[Tuple[int, ...]] = None
        self._last_hat: Optional[Tuple[int, int]] = None
        self._last_axes: Optional[Tuple[float, ...]] = None

        self.debug = debug
        self._debug_display = None
        if debug:
            self._debug_display = get_debug_display(
                "PS5->JoyCon", self.publish_rate, refresh_rate=10
            )

        logger.info(
            f"PS5 JoyCon-emulator configured: {self.publish_rate}Hz -> {self.topic} "
            f"(device_index={self.device_index})"
        )

    def initialize(self) -> bool:
        try:
            self.publisher = self.node.create_publisher(
                self.topic, encoder=DictDataCodec.encode
            )

            pygame.init()
            pygame.joystick.init()

            count = pygame.joystick.get_count()
            if count <= self.device_index:
                logger.error(
                    f"No joystick found at index {self.device_index}. Found {count} device(s)."
                )
                return False

            joy = pygame.joystick.Joystick(self.device_index)
            joy.init()
            self._joy = joy

            logger.success(f"PS5 controller connected: {joy.get_name()}")
            logger.info(
                f"axes={joy.get_numaxes()} buttons={joy.get_numbuttons()} hats={joy.get_numhats()}"
            )
            logger.info(
                "axis mapping: "
                f"lx={self.axis_lx} ly={self.axis_ly} rx={self.axis_rx} ry={self.axis_ry} "
                f"l2={self.axis_l2} r2={self.axis_r2} "
                f"invert_ly={self.invert_ly} invert_ry={self.invert_ry}"
            )

            if self.dump_buttons or self.dump_axes:
                msg = (
                    f"[ps5][dump] enabled dump_buttons={self.dump_buttons} "
                    f"dump_axes={self.dump_axes} eps={self.dump_eps} "
                    f"heartbeat_s={self.dump_heartbeat_s}\n"
                )
                sys.stdout.write(msg)
                sys.stdout.flush()
                sys.stderr.write(msg)
                sys.stderr.flush()

            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def _axis(self, i: int) -> float:
        assert self._joy is not None
        if 0 <= i < self._joy.get_numaxes():
            return float(self._joy.get_axis(i))
        return 0.0

    def _trigger_to_01_signed(self, raw: float) -> float:
        v = (raw + 1.0) * 0.5
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    def _dump_inputs_if_changed(self) -> None:
        """Print raw buttons/hat/axes only when something changes."""
        assert self._joy is not None

        pygame.event.pump()

        if self.dump_buttons:
            buttons = tuple(int(self._joy.get_button(i)) for i in range(self._joy.get_numbuttons()))
            hat: Tuple[int, int] = (0, 0)
            if self._joy.get_numhats() > 0:
                hx, hy = self._joy.get_hat(0)
                hat = (int(hx), int(hy))

            if buttons != self._last_buttons or hat != self._last_hat:
                self._last_buttons = buttons
                self._last_hat = hat

                pressed = [str(i) for i, v in enumerate(buttons) if v]
                pressed_str = ",".join(pressed) if pressed else "-"
                msg = f"[ps5][raw] buttons_pressed={pressed_str} hat={hat}\n"
                sys.stdout.write(msg)
                sys.stdout.flush()
                sys.stderr.write(msg)
                sys.stderr.flush()

        if self.dump_axes:
            axes = tuple(float(self._joy.get_axis(i)) for i in range(self._joy.get_numaxes()))
            changed = False
            if self._last_axes is None or len(self._last_axes) != len(axes):
                changed = True
            else:
                for a, b in zip(self._last_axes, axes):
                    if abs(a - b) >= self.dump_eps:
                        changed = True
                        break

            if changed:
                self._last_axes = axes
                axes_str = " ".join(f"{i}:{v:+.3f}" for i, v in enumerate(axes))
                msg = f"[ps5][raw] axes {axes_str}\n"
                sys.stdout.write(msg)
                sys.stdout.flush()
                sys.stderr.write(msg)
                sys.stderr.flush()

        # Heartbeat (lets us verify dump loop is alive even if nothing changes)
        if self.dump_heartbeat_s > 0:
            now_ns = time.time_ns()
            if self._last_heartbeat_ns == 0:
                self._last_heartbeat_ns = now_ns
            elif now_ns - self._last_heartbeat_ns >= int(self.dump_heartbeat_s * 1e9):
                self._last_heartbeat_ns = now_ns
                msg = "[ps5][dump] heartbeat\n"
                sys.stderr.write(msg)
                sys.stderr.flush()

    def _read_ps5_state(self) -> dict:
        """Read current PS5 state and convert to JoyCon-like dict structure."""
        assert self._joy is not None

        # Optional raw dump (change-only)
        if self.dump_buttons or self.dump_axes:
            self._dump_inputs_if_changed()
        else:
            pygame.event.pump()

        lx = self._apply_deadzone(self._axis(self.axis_lx))
        ly = self._apply_deadzone(self._axis(self.axis_ly))
        rx = self._apply_deadzone(self._axis(self.axis_rx))
        ry = self._apply_deadzone(self._axis(self.axis_ry))

        if self.invert_ly:
            ly = -ly
        if self.invert_ry:
            ry = -ry

        l2 = self._trigger_to_01_signed(self._axis(self.axis_l2))
        r2 = self._trigger_to_01_signed(self._axis(self.axis_r2))

        dpad_x, dpad_y = 0, 0
        if self._joy.get_numhats() > 0:
            dpad_x, dpad_y = self._joy.get_hat(0)

        def btn(i: int) -> bool:
            if 0 <= i < self._joy.get_numbuttons():
                return bool(self._joy.get_button(i))
            return False

        # NOTE: Updated indices based on --dump-buttons output
        joy_a = btn(1)          # circle
        joy_b = btn(0)          # cross
        joy_x = btn(2)          # triangle
        joy_y = btn(3)          # square

        l1 = btn(4)
        r1 = btn(5)
        l2_btn = btn(6)
        r2_btn = btn(7)
        l3 = btn(11)
        r3 = btn(12)
        share = btn(9)          # create
        ps = btn(10)
        options = btn(8)        # options
        touchpad = btn(13)

        left = {
            "buttons": {
                "up": bool(dpad_y == 1),
                "down": bool(dpad_y == -1),
                "left": bool(dpad_x == -1),
                "right": bool(dpad_x == 1),
                "l": bool(l1),
                "zl": bool(l2_btn or l2 > self.trigger_deadzone),
                "minus": bool(share),
                "capture": bool(touchpad),
                "stick": bool(l3),
            },
            "stick": {"x": float(lx), "y": float(ly)},
        }

        right = {
            "buttons": {
                "a": bool(joy_a),
                "b": bool(joy_b),
                "x": bool(joy_x),
                "y": bool(joy_y),
                "r": bool(r1),
                "zr": bool(r2_btn or r2 > self.trigger_deadzone),
                "plus": bool(options),
                "home": bool(ps),
                "stick": bool(r3),
            },
            "stick": {"x": float(rx), "y": float(ry)},
        }

        return {"left": left, "right": right}

    def run(self) -> None:
        resolved_topic = self.node.resolve_topic(self.topic)
        logger.info(f"Started publishing to {resolved_topic} at {self.publish_rate}Hz")

        if self._debug_display:
            logger.info("Starting debug display...")
            self._debug_display.start()
            logger.info("Debug display started")

        try:
            while True:
                state = self._read_ps5_state()

                data = JoyConData(
                    timestamp_ns=time.time_ns(),
                    left=state["left"],
                    right=state["right"],
                )

                self.publisher.publish(asdict(data))

                if self._debug_display:
                    debug_data = {"left": data.left, "right": data.right}
                    self._debug_display.print_joycon(debug_data)

                self.rate_limiter.sleep()

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.cleanup()

    def cleanup(self):
        if self._debug_display:
            self._debug_display.stop()

        try:
            if self._joy is not None:
                self._joy.quit()
        except Exception:
            pass

        try:
            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass

        self.node.shutdown()
        logger.info("PS5 JoyCon-emulator stopped")


def main(
    namespace: str = "",
    publish_rate: Optional[int] = None,
    topic: Optional[str] = None,
    deadzone: float = 0.1,
    debug: bool = True,
    device_index: int = 0,
    axis_lx: int = 0,
    axis_ly: int = 1,
    axis_rx: int = 3,
    axis_ry: int = 4,
    axis_l2: int = 2,
    axis_r2: int = 5,
    invert_ly: bool = False,
    invert_ry: bool = False,
    trigger_deadzone: float = 0.10,
    dump_buttons: bool = False,
    dump_axes: bool = False,
    dump_eps: float = 0.05,
    dump_heartbeat_s: float = 1.0,
):
    logger = setup_logging(debug)
    logger.info(
        f"Starting PS5 JoyCon-emulator{f' (namespace: {namespace})' if namespace else ''}"
    )

    reader = PS5JoyConEmulator(
        namespace=namespace,
        publish_rate=publish_rate,
        topic=topic,
        deadzone=deadzone,
        debug=debug,
        device_index=device_index,
        axis_lx=axis_lx,
        axis_ly=axis_ly,
        axis_rx=axis_rx,
        axis_ry=axis_ry,
        axis_l2=axis_l2,
        axis_r2=axis_r2,
        invert_ly=invert_ly,
        invert_ry=invert_ry,
        trigger_deadzone=trigger_deadzone,
        dump_buttons=dump_buttons,
        dump_axes=dump_axes,
        dump_eps=dump_eps,
        dump_heartbeat_s=dump_heartbeat_s,
    )

    if not reader.initialize():
        logger.error("Failed to initialize PS5 JoyCon-emulator")
        return 1

    reader.run()

    return 0


if __name__ == "__main__":
    sys.exit(tyro.cli(main))