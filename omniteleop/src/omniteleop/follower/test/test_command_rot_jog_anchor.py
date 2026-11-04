"""Unit tests for the dashboard rotation-jog position anchor.

Covers CommandProcessor._resolve_dashboard_jog_reference:
  - rotation-only jogs pin the position reference to an anchor captured at
    jog start, so per-tick IK residual does not accumulate as drift
  - a jog is "continuing" while identical ticks arrive within the timeout
  - timeout / changed delta / translation jog all re- or clear the anchor
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock dexcomm before importing command_processor (no Zenoh in unit tests)
sys.modules['dexcomm'] = MagicMock()
sys.modules['dexcomm.codecs'] = MagicMock()
sys.modules['dexcomm.utils'] = MagicMock()

import numpy as np

from omniteleop.follower.command_processor import CommandProcessor

RZ_PLUS = [
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
]
RZ_MINUS = [
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
]


class FakeClock:
    """Stand-in for time.monotonic with manual advance."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def make_processor(fk_positions):
    """Build a bare CommandProcessor whose FK returns scripted EE positions.

    fk_positions: list of (x, y, z) consumed one per _resolve call, so each
    tick "sees" the drifted pose the previous tick's IK residual produced.
    """
    cp = object.__new__(CommandProcessor)
    cp._rot_jog_anchor = {}
    poses = iter(fk_positions)

    def fk(frames):
        x, y, z = next(poses)
        T = np.eye(4)
        T[:3, 3] = [x, y, z]
        obj = MagicMock()
        obj.np = T
        return {frames[0]: obj}

    mm = MagicMock()
    mm.fk = fk
    cp.motion_manager = mm
    return cp


class TestRotJogAnchor(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        patcher = patch(
            'omniteleop.follower.command_processor.time.monotonic', self.clock
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def resolve(self, cp, **kwargs):
        arm_data = dict(kwargs)
        ok = cp._resolve_dashboard_jog_reference(arm_data, 'L_ee')
        return ok, arm_data

    def test_first_rot_tick_captures_anchor(self):
        cp = make_processor([(0.7, 0.2, 1.3)])
        ok, arm_data = self.resolve(
            cp, delta_xyz=[0.0, 0.0, 0.0], delta_R=RZ_PLUS
        )
        self.assertTrue(ok)
        self.assertIn('pose', arm_data)
        np.testing.assert_allclose(arm_data['pose'][:3, 3], [0.7, 0.2, 1.3])
        self.assertIn('L_ee', cp._rot_jog_anchor)

    def test_drifted_fk_keeps_original_anchor(self):
        # Tick 1 anchors at z=1.300; tick 2's FK "drifted" to z=1.299 — the
        # reference must still aim at 1.300, not fold the drift in.
        cp = make_processor([(0.7, 0.2, 1.300), (0.7, 0.2, 1.299)])
        self.resolve(cp, delta_xyz=[0.0, 0.0, 0.0], delta_R=RZ_PLUS)
        self.clock.advance(0.1)
        ok, arm_data = self.resolve(
            cp, delta_xyz=[0.0, 0.0, 0.0], delta_R=RZ_PLUS
        )
        self.assertTrue(ok)
        np.testing.assert_allclose(arm_data['pose'][:3, 3], [0.7, 0.2, 1.300])

    def test_timeout_reanchors(self):
        cp = make_processor([(0.7, 0.2, 1.300), (0.7, 0.2, 1.200)])
        self.resolve(cp, delta_xyz=[0.0, 0.0, 0.0], delta_R=RZ_PLUS)
        self.clock.advance(1.0)  # button released long enough
        ok, arm_data = self.resolve(
            cp, delta_xyz=[0.0, 0.0, 0.0], delta_R=RZ_PLUS
        )
        self.assertTrue(ok)
        np.testing.assert_allclose(arm_data['pose'][:3, 3], [0.7, 0.2, 1.200])

    def test_changed_delta_reanchors(self):
        cp = make_processor([(0.7, 0.2, 1.300), (0.7, 0.2, 1.100)])
        self.resolve(cp, delta_xyz=[0.0, 0.0, 0.0], delta_R=RZ_PLUS)
        self.clock.advance(0.1)
        ok, arm_data = self.resolve(
            cp, delta_xyz=[0.0, 0.0, 0.0], delta_R=RZ_MINUS  # direction flip
        )
        self.assertTrue(ok)
        np.testing.assert_allclose(arm_data['pose'][:3, 3], [0.7, 0.2, 1.100])

    def test_translation_jog_resamples_and_clears_anchor(self):
        cp = make_processor([(0.7, 0.2, 1.300), (0.7, 0.2, 1.350), (0.7, 0.2, 1.350)])
        self.resolve(cp, delta_xyz=[0.0, 0.0, 0.0], delta_R=RZ_PLUS)
        self.clock.advance(0.1)
        ok, arm_data = self.resolve(cp, delta_xyz=[0.0, 0.0, 0.05])
        self.assertTrue(ok)
        np.testing.assert_allclose(arm_data['pose'][:3, 3], [0.7, 0.2, 1.350])
        self.assertNotIn('L_ee', cp._rot_jog_anchor)
        # A rotation jog after the translation re-anchors at the new spot.
        self.clock.advance(0.1)
        ok, arm_data = self.resolve(
            cp, delta_xyz=[0.0, 0.0, 0.0], delta_R=RZ_PLUS
        )
        np.testing.assert_allclose(arm_data['pose'][:3, 3], [0.7, 0.2, 1.350])

    def test_anchors_are_per_arm(self):
        poses = iter([(0.7, 0.2, 1.3), (-0.7, -0.2, 1.3)])

        def fk(frames):
            x, y, z = next(poses)
            T = np.eye(4)
            T[:3, 3] = [x, y, z]
            obj = MagicMock()
            obj.np = T
            return {frames[0]: obj}

        cp = object.__new__(CommandProcessor)
        cp._rot_jog_anchor = {}
        cp.motion_manager = MagicMock()
        cp.motion_manager.fk = fk

        left = {'delta_xyz': [0.0, 0.0, 0.0], 'delta_R': RZ_PLUS}
        right = {'delta_xyz': [0.0, 0.0, 0.0], 'delta_R': RZ_PLUS}
        self.assertTrue(cp._resolve_dashboard_jog_reference(left, 'L_ee'))
        self.assertTrue(cp._resolve_dashboard_jog_reference(right, 'R_ee'))
        self.assertEqual(set(cp._rot_jog_anchor), {'L_ee', 'R_ee'})
        np.testing.assert_allclose(
            cp._rot_jog_anchor['R_ee']['pos'], [-0.7, -0.2, 1.3]
        )

    def test_no_delta_returns_false(self):
        cp = make_processor([(0.7, 0.2, 1.3)])
        ok, arm_data = self.resolve(cp)
        self.assertFalse(ok)
        self.assertNotIn('pose', arm_data)


class TestBuildEeTargetPoseFrame(unittest.TestCase):
    """rot_frame selects pre- vs post-multiplication of delta_R."""

    def setUp(self):
        # Non-trivial reference rotation so world/ee results differ.
        th = np.radians(35.0)
        self.ref_rot = np.array([
            [np.cos(th), 0.0, np.sin(th)],
            [0.0, 1.0, 0.0],
            [-np.sin(th), 0.0, np.cos(th)],
        ])
        self.pose = np.eye(4)
        self.pose[:3, :3] = self.ref_rot
        self.pose[:3, 3] = [0.6, -0.15, 1.1]
        self.delta_R = np.array(RZ_PLUS, dtype=float)

    def test_world_frame_pre_multiplies(self):
        target = CommandProcessor._build_ee_target_pose({
            'pose': self.pose, 'delta_R': RZ_PLUS,
        })
        np.testing.assert_allclose(
            target[:3, :3], self.delta_R @ self.ref_rot, atol=1e-12
        )
        np.testing.assert_allclose(target[:3, 3], [0.6, -0.15, 1.1])

    def test_ee_frame_post_multiplies(self):
        target = CommandProcessor._build_ee_target_pose({
            'pose': self.pose, 'delta_R': RZ_PLUS, 'rot_frame': 'ee',
        })
        np.testing.assert_allclose(
            target[:3, :3], self.ref_rot @ self.delta_R, atol=1e-12
        )
        np.testing.assert_allclose(target[:3, 3], [0.6, -0.15, 1.1])

    def test_frames_differ_for_tilted_pose(self):
        world = CommandProcessor._build_ee_target_pose({
            'pose': self.pose, 'delta_R': RZ_PLUS,
        })
        ee = CommandProcessor._build_ee_target_pose({
            'pose': self.pose, 'delta_R': RZ_PLUS, 'rot_frame': 'ee',
        })
        self.assertFalse(np.allclose(world[:3, :3], ee[:3, :3]))
        # Both remain pure rotations at the same position.
        np.testing.assert_allclose(np.linalg.det(ee[:3, :3]), 1.0)
        np.testing.assert_allclose(world[:3, 3], ee[:3, 3])


if __name__ == '__main__':
    unittest.main()
