"""Targeting System mode: SEG (box segmentation, before the grasp) <-> EDGE (place-target edges).

EDGE when BOTH hold for SWITCH_S:
  holding - the gripper's own sensor: motor current > HOLD_MA and motor angle strictly between
            "open" and "closed on nothing". Left KORAS, 2026-11-18: box held 815 mA / 3414 deg,
            closed empty 58 mA / 6135 deg, open 279 mA / 284 deg. (Arm joint current was tried
            first and rejected: static friction let the same empty pose hold at -4.6 or -10.8.)
  aligned - the gripper x axis (FK of the live joints) is within ALIGN_DEG of vertical
            (0.4 deg in the place pose = box upright). Yaw-free, so the body turn does not matter.
Anything else -> SEG. Run `python targeting.py` for the self-check.
"""
import os

HOLD_MA = float(os.getenv("TGT_HOLD_MA", "500"))
POS_MIN = float(os.getenv("TGT_POS_MIN", "500"))
POS_MAX = float(os.getenv("TGT_POS_MAX", "5500"))
ALIGN_DEG = float(os.getenv("TGT_ALIGN_DEG", "15"))
SWITCH_S = float(os.getenv("TGT_SWITCH_S", "0.5"))
STATUS_MAX_AGE_S = 1.0  # gripper reply older than this = no sensor -> not holding


def holding(status, status_age_s):
    if not status or status_age_s is None or status_age_s > STATUS_MAX_AGE_S:
        return False
    return status.get("motor_current_ma", 0) > HOLD_MA and POS_MIN < status.get("motor_position_deg", 0) < POS_MAX


class Mode:
    """Debounced SEG / EDGE switch."""

    def __init__(self):
        self.mode, self._want, self._since, self.why = "SEG", "SEG", 0.0, ""

    def update(self, status, status_age_s, tilt_deg, now):
        h = holding(status, status_age_s)
        a = tilt_deg is not None and tilt_deg < ALIGN_DEG
        want = "EDGE" if h and a else "SEG"
        if want != self._want:
            self._want, self._since = want, now
        if want != self.mode and now - self._since >= SWITCH_S:
            self.mode = want
        s = status or {}
        self.why = (f"grip {'HOLD' if h else '-'} {s.get('motor_current_ma', '?')}mA {s.get('motor_position_deg', '?')}deg"
                    f" | tilt {'-' if tilt_deg is None else f'{tilt_deg:.0f}'}deg{' OK' if a else ''}")
        return self.mode


if __name__ == "__main__":
    box, empty, opened = ({"motor_current_ma": 815, "motor_position_deg": 3414},
                          {"motor_current_ma": 58, "motor_position_deg": 6135},
                          {"motor_current_ma": 279, "motor_position_deg": 284})
    assert holding(box, 0.1) and not holding(empty, 0.1) and not holding(opened, 0.1)
    assert not holding(box, 5.0) and not holding(None, 0.1)          # stale / no sensor
    m = Mode()
    assert m.update(box, 0.1, 2.0, 0.0) == "SEG"                     # debounce: not yet
    assert m.update(box, 0.1, 2.0, 0.6) == "EDGE"                    # held + upright for 0.6 s
    assert m.update(box, 0.1, 40.0, 0.7) == "EDGE"                   # tilted: still debouncing
    assert m.update(box, 0.1, 40.0, 1.3) == "SEG"                    # tilted 0.6 s -> SEG
    assert m.update(opened, 0.1, 2.0, 2.0) == "SEG"                  # released -> SEG
    assert m.update(empty, 0.1, 2.0, 3.0) == "SEG"                   # closed on nothing -> SEG
    print("self-check ok:", m.why)
