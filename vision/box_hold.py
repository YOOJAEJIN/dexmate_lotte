"""Keep the grasp box's last unoccluded geometry while a gripper covers it.

The box does not move until it is grasped, so when a scan's face looks cut
(an edge narrower than the other, or narrower than the last full view - a
gripper over a corner or side) the previous full measurement is reported
instead. Height changes are NOT treated as occlusion: taking a 20-sheet bundle
off the top lowers the top edge but keeps both widths.

Run `python box_hold.py` for the self-check.
"""
import os

import numpy as np

# Measured 2026-11-12: unoccluded faces show top/bottom up to ~16% apart
# (mask bleed, perspective); a gripper over the top corners gave 21.4 vs 40.5.
SELF_RATIO = float(os.getenv("SEG_HOLD_SELF", "0.7"))  # short/long edge below => cut
RATIO = float(os.getenv("SEG_HOLD_RATIO", "0.85"))     # long edge below x last short edge => cut


def widths(box):
    """3D (top, bottom) edge widths in metres; corners ordered BR, BL, TL, TR."""
    c = [np.array(k["cam"]) for k in box.get("corners", [])]
    if len(c) != 4:
        return None
    return float(np.linalg.norm(c[3] - c[2])), float(np.linalg.norm(c[0] - c[1]))


def _center(box):
    return np.array(box["xyz_base"] if box["xyz_base"] is not None else box["xyz_cam"])


def choose(held, new, ratio=RATIO, self_ratio=SELF_RATIO):
    """-> (box to report, box to remember). A substituted box carries held=True."""
    w = widths(new)
    if w is None:
        return new, held
    wt, wb = w
    occluded = min(wt, wb) < self_ratio * max(wt, wb)
    # Same frame (both base or both camera) so the centre distance is meaningful.
    if held is not None and (held["xyz_base"] is None) == (new["xyz_base"] is None):
        ref = held["_width"]
        # ponytail: "same box" = centre within half a box width; swap of two
        # look-alike neighbours would fool it - track by mask IoU if that happens.
        same = np.linalg.norm(_center(new) - _center(held)) < 0.5 * ref
        if same and (occluded or max(wt, wb) < ratio * ref):
            return dict(held, held=True), held
    if occluded:
        return new, held  # no full view of this box yet: report the cut one as-is
    return new, dict(new, _width=min(wt, wb))


def _box(w_top, w_bot, h=0.455, at=(0.0, 0.0, 1.2), base=True):
    x, y, z = at
    cam = [[x + w_bot / 2, y + h, z], [x - w_bot / 2, y + h, z],   # BR, BL
           [x - w_top / 2, y, z], [x + w_top / 2, y, z]]           # TL, TR
    ctr = [x, y + h / 2, z]
    return {"corners": [{"cam": p} for p in cam],
            "xyz_cam": ctr, "xyz_base": ctr if base else None}


if __name__ == "__main__":
    full = _box(0.405, 0.405)
    out, held = choose(None, full)
    assert out is full and held["_width"] > 0.40                     # first full view kept

    out, held2 = choose(held, _box(0.214, 0.405))                    # gripper over top corners
    assert out.get("held") and held2 is held

    out, _ = choose(held, _box(0.30, 0.30))                          # both sides cut evenly
    assert out.get("held")

    lower = _box(0.405, 0.405, h=0.394)                              # top bundle taken off
    out, held3 = choose(held, lower)
    assert out is lower and not out.get("held") and held3["_width"] > 0.40

    moved = _box(0.214, 0.405, at=(0.6, 0.0, 1.2))                   # another (cut) box elsewhere
    out, held4 = choose(held, moved)
    assert out is moved and held4 is held

    out, held5 = choose(held, _box(0.214, 0.405, base=False))        # frame changed: no hold
    assert out is not held and held5 is held

    out, held6 = choose(None, _box(0.214, 0.405))                    # cut but nothing to hold
    assert not out.get("held") and held6 is None

    noisy = _box(0.406, 0.483)                                       # live unoccluded face (16% apart)
    out, held7 = choose(None, noisy)
    assert out is noisy and held7 is not None
    out, held8 = choose(held7, _box(0.41, 0.41))                     # cleaner view replaces it
    assert not out.get("held") and abs(held8["_width"] - 0.41) < 1e-6
    print("box_hold self-check OK")
