"""Key-driven per-motion action recording and playback for the simulator.

Wraps the live action provider (e.g. ``ZenohActionProvider``). Every env step
the wrapper passes the provider's action through unchanged, and, while
recording, appends it (plus the measured joint positions) to the current
segment. A segment is one motion: press the record key when a lotte_chilsung
script is about to run, press it again when the motion has finished. Each
segment is saved as ``seg_NNN.npz`` together with a full scene snapshot
(robot root/joint state + every rigid object pose) taken at its start, so it
can be replayed from the same situation later — also after a sim restart.

Playback feeds the saved action tensors back into ``env.step`` one per step,
so it is exact in *sim time* regardless of wall-clock planning pauses or the
sim's real-time factor. Long stretches of an unchanged action (planning gaps,
an idle tail before the record key was pressed again) are compressed to
``--record_idle_hold_s`` seconds on playback -- except around wheel drives,
which are replayed 1:1 (see ``_compress_idle_idx``).

Base turns are tracked, not just replayed (2026-09-28): the robot's root state
is recorded per step (``root_state`` in the npz) and, while a recorded wheel
drive with a yaw change is replayed, the wheel speed is scaled by the yaw lag
and the drive is extended at the end of the run (steering still at the recorded
angles) until the base has reached the recorded yaw -- the wheels slip and the
rear caster flips, so the same wheel commands alone do not give the same
heading (seg_014's -106 deg turn replayed short). Segments recorded before this
have no ``root_state`` and stay open-loop (re-record them to get tracking);
``recctl yawtrack off`` disables it for the next play. See ``YawTracker``.

Keys (Isaac Sim viewport must have focus). Function keys are used because
the viewport binds most letters (P = parent prim, R = scale gizmo,
Backspace/Delete = delete prim ...):

    F5     start / stop recording a segment (auto-saved on stop)
    F6     play ALL segments in order, restoring the first snapshot
           (a running recording is saved first)
    F7     play ALL segments from the current state (no restore)
    F8     play only the NEXT segment (cursor), no restore
    F9     stop playback (hold the last replayed pose)
    F10    delete the last recorded segment (press twice within 2 s)
    F3     reload segments from disk (also done at startup)
    F4     print this help

The same commands can be sent from a terminal without viewport focus, and
there you can also pick segments by their file number::

    python tools/recctl.py rec                 # F5
    python tools/recctl.py play                # F6: all, restore first snapshot
    python tools/recctl.py play 3              # only seg_003 (its snapshot restored first)
    python tools/recctl.py play 2 4 5          # several, in the given order
    python tools/recctl.py play 2-4            # a range
    python tools/recctl.py play-here 3         # seg_003 from the current state
    python tools/recctl.py restore 3           # only put the scene at seg_003's start
    python tools/recctl.py goto 3              # F8 (next) will play seg_003
    python tools/recctl.py speed 2             # playback speed x2 (0.25–8; used by F6/F7/F8 and play)
    python tools/recctl.py video off|on        # mp4 of every playback → recordings/videos/ (default on;
                                               #  960x540 @ 25 fps, encoded in a background thread)
    python tools/recctl.py release             # drop the post-playback hold right away
    python tools/recctl.py list | status | next | stop | delete | reload

After playback (or a restore) the wrapper keeps commanding the last replayed
pose and releases joints *per robot part* as soon as a fresh Zenoh command
message for that part arrives (the sim's ZenohManager counts received
messages per part, so a new command is recognised even when its values equal
the buffered old ones — e.g. "open" sent again after the hands were already
commanded open before the playback). Stale Zenoh targets from before the
playback therefore never reach the robot — not when recording is started
again (F5), and not when only the hands are commanded afterwards. If the live
provider does not expose those counters, a value-change test is used instead. Status is printed and mirrored to
``record_status.txt`` next to ``record_dir`` (its parent folder).
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from action_provider.action_base import ActionProvider

_STATUS_FILE = "record_status.txt"


# --------------------------------------------------------------------------- #
# scene snapshot
# --------------------------------------------------------------------------- #
def capture_snapshot(env) -> dict[str, np.ndarray]:
    """Flat dict of numpy arrays: robot/articulation root+joint state, rigid poses."""
    snap: dict[str, np.ndarray] = {}
    try:
        for name, art in env.scene.articulations.items():
            snap[f"art__{name}__root_state"] = art.data.root_state_w[0].detach().cpu().numpy().copy()
            snap[f"art__{name}__joint_pos"] = art.data.joint_pos[0].detach().cpu().numpy().copy()
            snap[f"art__{name}__joint_vel"] = art.data.joint_vel[0].detach().cpu().numpy().copy()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[Recorder] articulation snapshot failed: {exc}")
    try:
        for name, obj in env.scene.rigid_objects.items():
            snap[f"rig__{name}__root_state"] = obj.data.root_state_w[0].detach().cpu().numpy().copy()
    except Exception as exc:  # pragma: no cover
        print(f"[Recorder] rigid-object snapshot failed: {exc}")
    return snap


def _is_kinematic(asset) -> bool:
    """True for kinematic rigid bodies (PhysX rejects velocity writes on them)."""
    try:
        return bool(asset.cfg.spawn.rigid_props.kinematic_enabled)
    except AttributeError:
        return False


def restore_snapshot(env, snap: dict[str, np.ndarray]) -> None:
    """Write a snapshot back into the simulation (single env).

    Kinematic rigid objects (the debug marker spheres) get only their pose;
    writing velocities to them triggers
    ``PxRigidDynamic::setLinearVelocity: Body must be non-kinematic!``.
    """
    dev = env.device
    arts = getattr(env.scene, "articulations", {})
    rigs = getattr(env.scene, "rigid_objects", {})
    for key, arr in snap.items():
        kind, name, field = key.split("__", 2)
        t = torch.as_tensor(np.asarray(arr), dtype=torch.float32, device=dev).unsqueeze(0)
        try:
            if kind == "art" and name in arts:
                art = arts[name]
                if field == "root_state":
                    art.write_root_state_to_sim(t)
                elif field == "joint_pos":
                    vel = snap.get(f"art__{name}__joint_vel")
                    v = torch.as_tensor(np.asarray(vel), dtype=torch.float32, device=dev).unsqueeze(0) if vel is not None else torch.zeros_like(t)
                    art.write_joint_state_to_sim(t, v)
                    # keep the PD targets consistent until the first replayed action lands
                    try:
                        art.set_joint_position_target(t)
                        art.set_joint_velocity_target(torch.zeros_like(t))
                    except Exception:
                        pass
            elif kind == "rig" and name in rigs and field == "root_state":
                obj = rigs[name]
                if _is_kinematic(obj):
                    obj.write_root_pose_to_sim(t[:, :7])
                else:
                    obj.write_root_state_to_sim(t)
        except Exception as exc:
            print(f"[Recorder] restore {key} failed: {exc}")
    try:
        env.scene.write_data_to_sim()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# recorder / player
# --------------------------------------------------------------------------- #
def _quat_yaw(q) -> np.ndarray:
    """Yaw (rad) of quaternion(s) shaped (..., 4) in (w, x, y, z) order."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class YawTracker:
    """Closed-loop base yaw during the playback of one segment.

    A recorded base turn is a stretch of open-loop wheel speeds. On this robot the wheels slip heavily (the
    base makes roughly half of the kinematic yaw) and the rear caster has to flip over when the turn direction
    reverses, so the same wheel commands do not reproduce the same heading (2026-09-28: seg_014's -106 deg
    turn replayed short while seg_008's +89 deg turn replayed fine). With the base yaw recorded per step this
    tracker

    1. scales the replayed wheel speed by the yaw lag while a turn run is being replayed,
    2. at the end of each turn run keeps driving -- steering still at the recorded in-place angles, cursor
       held -- until the base is predicted to settle on the recorded settled yaw (the coast / unwinding after
       the wheel stop is predicted from the current yaw rate with the coast-per-rate ratio measured in the
       recording, so an inertial coast and an elastic unwinding are both handled),
    3. if it had to correct, holds the brake for ``SETTLE_STEPS``, compares with the recorded settled yaw and,
       if still off, drives once more aiming past the target by the coast just observed (sign flipped when the
       new pass drives the other way; ``MAX_ROUNDS`` passes),

    then lets the replay continue with the recorded brake / steer-home steps. Everything is relative to the
    segment's own start yaw, so it also works for play-here. Gains are calibrated from the recording itself:
    achieved yaw per wheel-command unit, slowest / fastest wheel command of the run.
    """

    TURN_MIN_DEG = 5.0      # a drive run with less recorded yaw change is a translation: left alone
    TOL_DEG = 1.0           # the end-of-run correction stops inside this (at the wheel stop)
    SETTLE_TOL_DEG = 1.5    # accepted after the base has settled
    SETTLE_STEPS = 60       # 0.6 s sim of brake before the settled check (only after a correction)
    MAX_ROUNDS = 2          # correction passes per turn run
    KP = 2.0                # yaw rate (rad/s) per rad of lag, as in _base_turn.turn_in_place
    MAX_CORR_STEPS = 600    # per pass (6 s sim at 0.01 s/step)
    DIVERGE_DEG = 3.0       # abort a pass if the lag grew by this after 100 steps (sign convention?)
    VEL_LIMIT = 0.10        # |wheel command| cap: sim joint velocity limit 10 rad/s / chassis_drive_vel scale 100

    def __init__(self, actions: np.ndarray, yaw: np.ndarray, vel_slice: slice, step_dt: float):
        self.actions = actions
        self.yaw = np.asarray(yaw, dtype=np.float64)
        self.vs = vel_slice
        self.dt = float(step_dt)
        n = len(actions)
        nz = (np.abs(actions[:, vel_slice]).max(axis=1) > 1e-6) if n else np.zeros(0, dtype=bool)
        self.run_of_step = np.full(n, -1, dtype=int)
        self.runs: list[dict] = []
        i = 0
        while i < n:
            if not nz[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and nz[j + 1]:
                j += 1
            drive = np.asarray(actions[i:j + 1, vel_slice], dtype=np.float64)
            mag = np.abs(drive).max(axis=1)                      # dominant wheel per step
            dyaw = float(_wrap(self.yaw[j] - self.yaw[i]))
            turn = abs(dyaw) >= np.deg2rad(self.TURN_MIN_DEG)
            cmd_integral = float((mag * self.dt).sum())          # command-unit * s
            rate_per_unit = abs(dyaw) / cmd_integral if cmd_integral > 1e-9 else 0.0
            last = drive[-1]
            settle_idx = min(j + self.SETTLE_STEPS, n - 1)
            m = min(10, j - i)
            omega_rec = float(_wrap(self.yaw[j] - self.yaw[j - m]) / (m * self.dt)) if m > 0 else 0.0
            coast_rec = float(_wrap(self.yaw[settle_idx] - self.yaw[j]))
            tau_rec = float(np.clip(coast_rec / omega_rec, -0.5, 0.5)) if abs(omega_rec) > 1e-3 else 0.0
            self.runs.append({
                "start": i, "end": j, "dyaw": dyaw, "turn": turn,
                "pattern": last / max(float(np.abs(last).max()), 1e-9),   # per-wheel sign / ratio pattern
                "v_min": float(mag[mag > 1e-6].min()) if turn else 0.0,
                "v_max": float(min(mag.max() * 1.25, self.VEL_LIMIT)),
                "k": (self.KP / rate_per_unit) if rate_per_unit > 1e-9 else 0.0,  # command units per rad
                "settle_idx": settle_idx,  # clipped below to the next run's start
                "tau": tau_rec,            # s: coast after the wheel stop per unit of yaw rate (negative = unwinds)
                "phase": "correct", "round": 0, "corr": 0, "settle_n": 0, "target": None,
                "e_first": None, "stop_actual": None, "last_dir": 0.0, "done": not turn, "resid": None,
            })
            self.run_of_step[i:j + 1] = len(self.runs) - 1
            i = j + 1
        for r, nxt in zip(self.runs, self.runs[1:]):
            r["settle_idx"] = min(r["settle_idx"], max(nxt["start"] - 1, r["end"]))
        self._prev_actual: float | None = None
        self.omega = 0.0  # filtered actual yaw rate (rad/s)
        self.n_turns = sum(1 for r in self.runs if r["turn"])
        self.ref: tuple[float, float] | None = None  # (actual yaw, recorded yaw) at the reference step
        self.last_err = 0.0
        self.total_corr = 0
        self.total_settle = 0

    # -- geometry helpers (all relative to the reference step) --
    def _rel(self, yaw_rec: float) -> float:
        return float(_wrap(yaw_rec - self.ref[1]))

    def _actual(self, yaw_now: float) -> float:
        return float(_wrap(yaw_now - self.ref[0]))

    def lag(self, pos: int, yaw_now: float):
        """Recorded-minus-actual yaw progress (rad, + = base behind the recording); None until referenced."""
        if self.ref is None:
            if pos >= 1:  # step 0 may still show the pre-restore pose: reference one step later
                self.ref = (float(yaw_now), float(self.yaw[pos]))
            return None
        return float(_wrap(self._rel(self.yaw[pos]) - self._actual(yaw_now)))

    def step(self, pos: int, action: np.ndarray, yaw_now: float):
        """Return (action to send, advance the cursor?, status message or None)."""
        e = self.lag(pos, yaw_now)
        if e is None:
            return action, True, None
        self.last_err = e
        actual = self._actual(yaw_now)
        if self._prev_actual is not None:
            self.omega += 0.3 * (float(_wrap(actual - self._prev_actual)) / self.dt - self.omega)
        self._prev_actual = actual
        prev = int(self.run_of_step[pos - 1]) if pos > 0 else -1
        if prev >= 0 and prev != int(self.run_of_step[pos]) and not self.runs[prev]["done"]:
            return self._after_run(self.runs[prev], action, yaw_now)
        cur = int(self.run_of_step[pos])
        if cur >= 0 and self.runs[cur]["turn"]:
            # inside a turn run: scale the recorded wheel command by the lag (never reverse it)
            r = self.runs[cur]
            drive = np.asarray(action[self.vs], dtype=np.float64)
            mag_rec = float(np.abs(drive).max())
            if mag_rec > 1e-6:
                along = e * np.sign(r["dyaw"])
                mag_new = float(np.clip(mag_rec + r["k"] * along, 0.0, r["v_max"]))
                out = np.array(action, dtype=np.float32, copy=True)
                out[self.vs] = drive * (mag_new / mag_rec)
                return out, True, None
        return action, True, None

    def _after_run(self, r: dict, brake_action: np.ndarray, yaw_now: float):
        """First recorded step after a turn run (zero drive, steering still at the turn angles): correct / settle.

        ``brake_action`` is that recorded step; it is re-sent (cursor held) while settling."""
        actual = self._actual(yaw_now)
        if r["phase"] == "correct":
            target = r["target"] if r["target"] is not None else self._rel(self.yaw[r["settle_idx"]])
            # where the base will settle if the wheels stop now: current yaw + the coast the recording showed
            e_end = float(_wrap(target - (actual + self.omega * r["tau"])))
            along = e_end * np.sign(r["dyaw"])  # + = still short of the recorded turn, - = past it
            n = r["corr"]
            if r["e_first"] is None:
                r["e_first"] = e_end
            finished = abs(e_end) < np.deg2rad(self.TOL_DEG) or n >= self.MAX_CORR_STEPS
            overshoot = n > 0 and np.sign(e_end) != np.sign(r["e_first"])
            diverging = n >= 100 and abs(e_end) > abs(r["e_first"]) + np.deg2rad(self.DIVERGE_DEG)
            if finished or overshoot or diverging:
                r["stop_actual"] = actual
                r["resid"] = e_end
                if diverging or n >= self.MAX_CORR_STEPS or (n == 0 and r["round"] == 0):
                    # nothing to settle: the replay already matched, or the correction cannot work
                    r["done"] = True
                    why = ("DIVERGING - wheel/yaw sign convention?" if diverging else
                           "timeout" if n >= self.MAX_CORR_STEPS else "matched, no correction needed")
                    return brake_action, True, (f"base turn {np.rad2deg(r['dyaw']):+.1f} deg: residual "
                                                f"{np.rad2deg(e_end):+.1f} deg at the wheel stop ({why})")
                r["phase"], r["settle_n"] = "settle", 0
                return brake_action, False, None
            r["corr"] = n + 1
            r["last_dir"] = float(np.sign(along))
            self.total_corr += 1
            corr = np.array(self.actions[r["end"]], dtype=np.float32, copy=True)  # last drive step: steer + pattern
            mag = float(np.clip(r["k"] * abs(e_end), r["v_min"], r["v_max"]))
            corr[self.vs] = r["pattern"] * (np.sign(along) * mag)
            return corr, False, None
        # settle: hold the brake, then compare with the recorded settled yaw
        r["settle_n"] += 1
        self.total_settle += 1
        if r["settle_n"] < self.SETTLE_STEPS:
            return brake_action, False, None
        target_settled = self._rel(self.yaw[r["settle_idx"]])
        e_s = float(_wrap(target_settled - actual))
        if abs(e_s) < np.deg2rad(self.SETTLE_TOL_DEG) or r["round"] + 1 >= self.MAX_ROUNDS:
            r["done"] = True
            r["resid"] = e_s
            return brake_action, True, (f"base turn {np.rad2deg(r['dyaw']):+.1f} deg: residual {np.rad2deg(e_s):+.1f} deg "
                                        f"after settling ({r['round'] + 1} pass(es), {self.total_corr} correction step(s))")
        # what the base did after the wheels stopped; the next pass gets the same coast, mirrored if it drives
        # the other way, so aim past the settled target by that much
        coast = float(_wrap(actual - r["stop_actual"]))
        next_dir = float(np.sign(e_s * np.sign(r["dyaw"]))) or 1.0
        if r["last_dir"] and next_dir != r["last_dir"]:
            coast = -coast
        r["target"] = float(_wrap(target_settled - coast))
        r["round"] += 1
        r["phase"], r["corr"], r["e_first"] = "correct", 0, None
        return brake_action, False, None

    def summary(self, yaw_now: float) -> str:
        rec = float(_wrap(self.yaw[-1] - self.yaw[0]))
        if self.ref is None:
            return f"base yaw: recorded delta {np.rad2deg(rec):+.1f} deg, not tracked (segment too short)"
        e = self.lag(len(self.yaw) - 1, yaw_now)
        return (f"base yaw: recorded delta {np.rad2deg(rec):+.1f} deg, replay error {np.rad2deg(e or 0.0):+.1f} deg "
                f"({self.n_turns} turn run(s), {self.total_corr} correction + {self.total_settle} settle step(s) inserted)")

class Segment:
    def __init__(self, path: Path, actions: np.ndarray, joint_pos: np.ndarray, snapshot: dict, meta: dict,
                 root_state: np.ndarray | None = None):
        self.path = path
        self.actions = actions
        self.joint_pos = joint_pos
        self.snapshot = snapshot
        self.meta = meta
        self.root_state = root_state  # (steps, 13) robot root state per step; None for pre-2026-09-28 files
        self._yaw = None

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    @property
    def yaw(self) -> np.ndarray | None:
        """Base yaw per recorded step (rad) or None when the root state was not recorded."""
        if self._yaw is None and self.root_state is not None and len(self.root_state) == len(self.actions):
            self._yaw = _quat_yaw(self.root_state[:, 3:7])
        return self._yaw

    def drives_chassis(self) -> bool:
        return self.actions.shape[1] >= 2 and float(np.abs(self.actions[:, -2:]).max()) > 1e-6


class SegmentRecorder(ActionProvider):
    """Pass-through action provider with keyed segment recording and playback."""

    IDLE, RECORDING, PLAYING = "idle", "recording", "playing"

    def __init__(
        self,
        inner: ActionProvider,
        env,
        record_dir: str | os.PathLike,
        *,
        idle_hold_s: float = 0.5,
        play_speed: float = 1.0,
        record_video: bool = True,
        video_every: int = 4,
        video_size: tuple[int, int] = (960, 540),
        status_path: str | os.PathLike | None = None,
        yaw_track: bool = True,
    ):
        super().__init__(f"SegmentRecorder({inner.name})")
        self.yaw_track = bool(yaw_track)  # closed-loop base yaw on playback (YawTracker); 'yawtrack on|off'
        self._tracker: YawTracker | None = None
        self._last_track_summary = ""
        self.inner = inner
        # Slider gripper lock (ZenohActionProvider): we decide what the robot receives
        # (replayed / held / live), so feed it the final action each step.
        self._observe = getattr(inner, "observe_action", None)
        if callable(self._observe):
            setattr(inner, "gripper_lock_external", True)
        self.env = env
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = Path(status_path) if status_path else self.record_dir.parent / _STATUS_FILE
        self.idle_hold_s = float(idle_hold_s)
        self.play_speed = float(np.clip(play_speed, 0.25, 8.0))
        # playback video (viewport camera via env.render(), encoded with imageio-ffmpeg / cv2)
        self.video_enabled = bool(record_video)
        self.video_every = max(int(video_every), 1)
        self.video_size = (int(video_size[0]) // 8 * 8, int(video_size[1]) // 8 * 8)
        self.video_dir = self.record_dir / "videos"
        self._vid_writer = None
        self._vid_backend = ""
        self._vid_path: Path | None = None
        self._vid_frames = 0
        self._vid_dropped = 0
        self._vid_tick = 0
        self._vid_queue: queue.Queue | None = None
        self._vid_thread: threading.Thread | None = None
        self._vid_annotator = None  # dedicated low-res render product on the viewport camera
        self._lock = threading.Lock()
        self._state = self.IDLE

        # env info
        robot = env.scene["robot"]
        self.joint_names = list(robot.data.joint_names)
        try:
            am = env.action_manager
            self.action_terms = list(am.active_terms)
            self.action_dims = [int(d) for d in am.action_term_dim]
        except Exception:
            self.action_terms, self.action_dims = [], []
        try:
            self.step_dt = float(env.step_dt)
        except Exception:
            self.step_dt = 0.01

        # recording buffers
        self._rec_actions: list[np.ndarray] = []
        self._rec_qpos: list[np.ndarray] = []
        self._rec_root: list[np.ndarray] = []
        self._rec_snapshot: dict | None = None
        self._rec_t0 = 0.0

        # playback
        self.segments: list[Segment] = []
        self._play_list: list[int] = []
        self._play_seg = 0
        self._play_pos = 0
        self._play_actions: np.ndarray | None = None
        self._last_played: torch.Tensor | None = None
        # hold overlay (survives state changes): per-element mask of joints still
        # pinned to _hold_values until the live command for them changes
        self._hold_mask: torch.Tensor | None = None
        self._hold_values: torch.Tensor | None = None
        self._hold_ref: torch.Tensor | None = None
        self._hold_seq_ref: dict | None = None  # per-part command counters at hold start
        self.cursor = 0  # next segment for the N key

        self._keyboard_sub = None
        self._input = None
        self._keyboard = None
        self._pending_keys: list[str] = []
        self._delete_armed_at = 0.0
        self.cmd_path = self.record_dir / "cmd.txt"
        try:
            self.cmd_path.unlink()
        except OSError:
            pass
        self.reload()
        self._status("ready", force=True)

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        self.inner.start()
        self.is_running = True

    def stop(self):
        with self._lock:
            if self._state == self.RECORDING:
                self._finish_recording_locked()
            self._video_stop()
        self.is_running = False
        self.inner.stop()

    def cleanup(self):
        self.detach_keyboard()
        self.inner.cleanup()

    def reset(self):
        self.inner.reset()

    # ------------------------------------------------------------------ keyboard
    def attach_keyboard(self) -> bool:
        try:
            import carb.input  # type: ignore
            import omni.appwindow  # type: ignore

            appwindow = omni.appwindow.get_default_app_window()
            self._input = carb.input.acquire_input_interface()
            self._keyboard = appwindow.get_keyboard()
            self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                self._keyboard, self._on_keyboard_event
            )
            self._key_press = carb.input.KeyboardEventType.KEY_PRESS
            print(self.help_text())
            return True
        except Exception as exc:
            print(f"[Recorder] keyboard not available ({exc}); use the file trigger instead.")
            return False

    def detach_keyboard(self) -> None:
        if self._input is not None and self._keyboard_sub is not None:
            try:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            except Exception:
                pass
        self._keyboard_sub = None

    def _on_keyboard_event(self, event, *args, **kwargs):
        try:
            if event.type != self._key_press:
                return True
            self.handle_key(str(event.input.name))
        except Exception as exc:
            print(f"[Recorder] key handling error: {exc}")
        return True

    KEYMAP = {
        "F5": "rec", "F6": "play", "F7": "play-here", "F8": "next",
        "F9": "stop", "F10": "delete", "F3": "reload", "F4": "help",
    }
    COMMANDS = ("rec", "play", "play-here", "next", "stop", "delete", "reload", "help", "restore", "goto", "list", "release", "speed", "video", "yawtrack")

    @staticmethod
    def help_text() -> str:
        return (
            "[Recorder] keys (viewport focused): F5=record start/stop  F6=play all (restore)  "
            "F7=play all (no restore)  F8=play next  F9=stop playback  F10=delete last (x2)  "
            "F3=reload  F4=help   |  terminal: python tools/recctl.py <rec|play|play-here|next|stop|delete|reload|status>"
        )

    def handle_key(self, key: str) -> None:
        """Queue a key; it is applied at the next ``get_action`` (step boundary).

        carb dispatches keyboard events during the app update inside
        ``env.step``, so scene writes (snapshot restore) must not happen there.
        """
        cmd = self.KEYMAP.get(key.upper())
        if cmd is None:
            return
        with self._lock:
            self._pending_keys.append(cmd)

    def handle_command(self, cmd: str) -> None:
        """Queue a textual command, optionally with segment selectors (e.g. 'play 2-4')."""
        cmd = " ".join(cmd.strip().lower().split())
        head = cmd.split(" ", 1)[0]
        if head in self.COMMANDS:
            with self._lock:
                self._pending_keys.append(cmd)
        else:
            print(f"[Recorder] unknown command '{cmd}'; one of {self.COMMANDS}")

    def _poll_command_file(self) -> None:
        """Pick up commands written by tools/recctl.py (one word per line)."""
        try:
            if not self.cmd_path.exists():
                return
            text = self.cmd_path.read_text(encoding="utf-8")
            self.cmd_path.unlink()
        except OSError:
            return
        for line in text.splitlines():
            if line.strip():
                self.handle_command(line)

    def _drain_keys(self) -> None:
        with self._lock:
            cmds, self._pending_keys = self._pending_keys, []
        for cmd in cmds:
            self._do_key(cmd)

    def _do_key(self, cmd: str) -> None:
        cmd = self.KEYMAP.get(cmd.upper(), cmd.lower())  # accept key names too
        parts = cmd.split()
        cmd, args = parts[0], parts[1:]
        if cmd == "rec":
            self.toggle_recording()
        elif cmd in ("play", "play-here"):
            with self._lock:
                if self._state == self.RECORDING:
                    self._finish_recording_locked()
                    print("[Recorder] recording saved before playback.")
            if args:
                self.play_selected(args, restore=(cmd == "play"))
            else:
                self.play_all(restore=(cmd == "play"))
        elif cmd == "restore":
            self.restore_selected(args)
        elif cmd == "goto":
            self.goto(args)
        elif cmd == "list":
            self.list_segments()
        elif cmd == "next":
            with self._lock:
                if self._state == self.RECORDING:
                    self._finish_recording_locked()
            self.play_next()
        elif cmd == "stop":
            self.stop_playback()
        elif cmd == "release":
            self.release_hold()
        elif cmd == "speed":
            self.set_speed(args[0] if args else None)
        elif cmd == "video":
            if args and args[0] in ("on", "off"):
                self.video_enabled = args[0] == "on"
            self._status(f"playback video {'ON' if self.video_enabled else 'OFF'} → {self.video_dir}", force=True)
        elif cmd == "yawtrack":
            if args and args[0] in ("on", "off"):
                self.yaw_track = args[0] == "on"
            self._status(f"base yaw tracking on playback {'ON' if self.yaw_track else 'OFF'} (applies to the next play; "
                         "only segments recorded with root_state)", force=True)
        elif cmd == "delete":
            now = time.time()
            if now - self._delete_armed_at < 2.0:
                self._delete_armed_at = 0.0
                self.delete_last()
            else:
                self._delete_armed_at = now
                self._status("press delete again within 2 s to remove the last segment", force=True)
        elif cmd == "reload":
            self.reload()
            self._status("reloaded", force=True)
        elif cmd == "help":
            print(self.help_text())

    # ------------------------------------------------------------------ recording
    def toggle_recording(self) -> None:
        with self._lock:
            if self._state == self.RECORDING:
                self._finish_recording_locked()
            elif self._state == self.IDLE:
                self._rec_actions, self._rec_qpos, self._rec_root = [], [], []
                self._rec_snapshot = capture_snapshot(self.env)
                self._rec_t0 = time.time()
                self._state = self.RECORDING
                self._status(f"REC ● segment {len(self.segments) + 1} started", force=True)
            else:
                print("[Recorder] cannot record while playing (press X first).")

    def _finish_recording_locked(self) -> None:
        n = len(self._rec_actions)
        self._state = self.IDLE
        if n == 0:
            self._status("recording stopped: empty, nothing saved", force=True)
            return
        self._sync_from_disk_locked()  # files may have been deleted/added outside
        idx = self._next_index()
        path = self.record_dir / f"seg_{idx:03d}.npz"
        actions = np.stack(self._rec_actions).astype(np.float32)
        qpos = np.stack(self._rec_qpos).astype(np.float32)
        root = np.stack(self._rec_root).astype(np.float32) if len(self._rec_root) == n else None
        travel = float(np.ptp(actions, axis=0).max()) if len(actions) > 1 else 0.0
        yaw_txt = ""
        if root is not None:
            yaw_rec = _quat_yaw(root[:, 3:7])
            yaw_txt = f", base yaw {np.rad2deg(yaw_rec[0]):+.1f} -> {np.rad2deg(yaw_rec[-1]):+.1f} deg (delta {np.rad2deg(_wrap(yaw_rec[-1] - yaw_rec[0])):+.1f})"
        meta = {
            "action_travel": travel,
            "index": idx,
            "steps": n,
            "step_dt": self.step_dt,
            "sim_seconds": n * self.step_dt,
            "wall_seconds": time.time() - self._rec_t0,
            "joint_names": self.joint_names,
            "action_terms": self.action_terms,
            "action_dims": self.action_dims,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if root is not None:
            meta["base_yaw_delta_deg"] = float(np.rad2deg(_wrap(yaw_rec[-1] - yaw_rec[0])))
        payload = {"actions": actions, "joint_pos": qpos, "meta": np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)}
        if root is not None:
            payload["root_state"] = root
        payload.update({f"snap__{k}": v for k, v in (self._rec_snapshot or {}).items()})
        np.savez(path, **payload)
        self.segments.append(Segment(path, actions, qpos, self._rec_snapshot or {}, meta, root))
        self.cursor = len(self.segments) - 1 if self.cursor >= len(self.segments) - 1 else self.cursor
        self._rec_actions, self._rec_qpos, self._rec_root, self._rec_snapshot = [], [], [], None
        self._status(
            f"saved {path.name}: {n} steps = {meta['sim_seconds']:.1f}s sim ({meta['wall_seconds']:.1f}s wall), "
            f"max action travel {travel:.3f}{yaw_txt}",
            force=True,
        )
        if travel < 1e-4:
            self._status(
                f"WARNING {path.name} contains NO motion (constant commands). Did the script/teleop actually "
                "send commands while recording? Check its terminal output; delete with F10 / recctl delete.",
                force=True,
            )

    def delete_last(self) -> None:
        with self._lock:
            if self._state != self.IDLE:
                print("[Recorder] stop recording/playback before deleting.")
                return
            self._sync_from_disk_locked()
            if not self.segments:
                self._status("no segments to delete", force=True)
                return
            seg = self.segments.pop()
            try:
                seg.path.unlink()
            except OSError:
                pass
            self.cursor = min(self.cursor, len(self.segments))
            self._status(f"deleted {seg.path.name}; {len(self.segments)} segment(s) left", force=True)

    @staticmethod
    def _seg_number(path: Path) -> int:
        try:
            return int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    def _scan_disk(self) -> list[Segment]:
        segs: list[Segment] = []
        for path in sorted(self.record_dir.glob("seg_*.npz"), key=self._seg_number):
            try:
                with np.load(path, allow_pickle=False) as z:
                    meta = json.loads(bytes(z["meta"]).decode("utf-8")) if "meta" in z else {}
                    snap = {k[len("snap__"):]: z[k] for k in z.files if k.startswith("snap__")}
                    root = z["root_state"] if "root_state" in z.files else None
                    segs.append(Segment(path, z["actions"], z["joint_pos"], snap, meta, root))
            except Exception as exc:
                print(f"[Recorder] skip {path.name}: {exc}")
        return segs

    def _sync_from_disk_locked(self) -> None:
        """Refresh the in-memory list from disk (caller holds the lock)."""
        known = {seg.path: seg for seg in self.segments}
        segs: list[Segment] = []
        for path in sorted(self.record_dir.glob("seg_*.npz"), key=self._seg_number):
            seg = known.get(path)
            if seg is None:
                try:
                    with np.load(path, allow_pickle=False) as z:
                        meta = json.loads(bytes(z["meta"]).decode("utf-8")) if "meta" in z else {}
                        snap = {k[len("snap__"):]: z[k] for k in z.files if k.startswith("snap__")}
                        root = z["root_state"] if "root_state" in z.files else None
                        seg = Segment(path, z["actions"], z["joint_pos"], snap, meta, root)
                except Exception as exc:
                    print(f"[Recorder] skip {path.name}: {exc}")
                    continue
            segs.append(seg)
        self.segments = segs
        self.cursor = min(self.cursor, len(segs))

    def _next_index(self) -> int:
        """1 when the folder has no seg_*.npz, else max existing number + 1."""
        nums = [self._seg_number(p) for p in self.record_dir.glob("seg_*.npz")]
        nums = [n for n in nums if n > 0]
        return (max(nums) + 1) if nums else 1

    def reload(self) -> None:
        segs = self._scan_disk()
        with self._lock:
            self.segments = segs
            self.cursor = 0
        total = sum(len(s) for s in segs) * self.step_dt
        print(f"[Recorder] {len(segs)} segment(s) in {self.record_dir} ({total:.1f}s sim total)")

    # ------------------------------------------------------------------ video
    def _video_fps(self) -> float:
        return 1.0 / (self.step_dt * self.video_every)

    def _video_grab(self):
        """Latest viewport frame (H,W,3 uint8) or None. Uses a dedicated low-res render
        product on the viewer camera; falls back to env.render() at the viewer resolution."""
        if self._vid_annotator is None and not getattr(self, "_vid_annotator_failed", False):
            try:
                import omni.replicator.core as rep  # type: ignore

                cam = self.env.cfg.viewer.cam_prim_path
                rp = rep.create.render_product(cam, self.video_size)
                ann = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                ann.attach([rp])
                self._vid_annotator = ann
                print(f"[Recorder] video capture {self.video_size[0]}x{self.video_size[1]} from {cam}")
            except Exception as exc:
                self._vid_annotator_failed = True
                print(f"[Recorder] video: replicator capture unavailable ({exc}); using env.render()")
        if self._vid_annotator is not None:
            data = self._vid_annotator.get_data()
            arr = np.frombuffer(data, dtype=np.uint8).reshape(*data.shape) if hasattr(data, "shape") else np.asarray(data)
            return arr[:, :, :3] if arr.size else None
        if getattr(self.env, "render_mode", None) is None:
            self.env.render_mode = "rgb_array"
        frame = self.env.render()
        return None if frame is None or getattr(frame, "size", 0) == 0 else frame[:, :, :3]

    def _video_start(self, label: str) -> None:
        """Arm frame capture for this playback; encoding runs in a background thread."""
        self._video_stop()
        if not self.video_enabled:
            return
        try:
            self.video_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            safe = label.replace(".npz", "").replace(", ", "+").replace(" ", "_")[:60]
            self._vid_path = self.video_dir / f"{stamp}_{safe}_x{self.play_speed:g}.mp4"
            self._vid_frames = 0
            self._vid_dropped = 0
            self._vid_tick = 0
            self._vid_queue = queue.Queue(maxsize=64)
            self._vid_thread = threading.Thread(target=self._video_worker, name="recorder-video", daemon=True)
            self._vid_thread.start()
        except Exception as exc:
            print(f"[Recorder] video disabled ({exc})")
            self._vid_path = None

    def _video_capture(self) -> None:
        """Called on the sim thread while playing: grab a frame every video_every steps, never block."""
        if self._vid_path is None or self._vid_queue is None:
            return
        self._vid_tick += 1
        if self._vid_tick % self.video_every:
            return
        try:
            frame = self._video_grab()
        except Exception as exc:
            print(f"[Recorder] video: frame grab failed ({exc}); video off for this playback")
            self._vid_path = None
            return
        if frame is None or not frame.any():
            return  # renderer warm-up frames are empty
        try:
            self._vid_queue.put_nowait(np.ascontiguousarray(frame))
        except queue.Full:
            self._vid_dropped += 1  # encoder is behind: drop rather than stall the simulation

    def _video_worker(self) -> None:
        q = self._vid_queue
        while True:
            item = q.get()
            if item is None:
                break
            try:
                if self._vid_writer is None:
                    self._video_open(item.shape[1], item.shape[0])
                    if self._vid_writer is None:
                        continue
                if self._vid_backend == "imageio":
                    self._vid_writer.append_data(item)
                else:
                    import cv2  # type: ignore
                    self._vid_writer.write(cv2.cvtColor(item, cv2.COLOR_RGB2BGR))
                self._vid_frames += 1
            except Exception as exc:
                print(f"[Recorder] video: write failed ({exc})")

    def _video_open(self, width: int, height: int) -> None:
        fps = self._video_fps()
        try:
            import imageio.v2 as imageio  # type: ignore

            self._vid_writer = imageio.get_writer(
                str(self._vid_path), fps=fps, codec="libx264", quality=7, pixelformat="yuv420p",
                macro_block_size=8, ffmpeg_params=["-preset", "veryfast", "-threads", "2"],
            )
            self._vid_backend = "imageio"
        except Exception as exc:
            try:
                import cv2  # type: ignore

                self._vid_writer = cv2.VideoWriter(
                    str(self._vid_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                self._vid_backend = "cv2"
                if not self._vid_writer.isOpened():
                    raise RuntimeError("cv2.VideoWriter failed to open")
            except Exception as exc2:
                print(f"[Recorder] video: no encoder available ({exc}; {exc2})")
                self._vid_writer = None
                self._vid_path = None
                return
        print(f"[Recorder] video → {self._vid_path} ({width}x{height} @ {fps:.1f} fps, {self._vid_backend})")

    def _video_stop(self) -> None:
        q, th = self._vid_queue, self._vid_thread
        self._vid_queue = None
        self._vid_thread = None
        if q is not None:
            try:
                q.put(None, timeout=1.0)
            except queue.Full:
                pass
        if th is not None:
            th.join(timeout=10.0)
        if self._vid_writer is not None:
            try:
                if self._vid_backend == "imageio":
                    self._vid_writer.close()
                else:
                    self._vid_writer.release()
            except Exception:
                pass
            if self._vid_frames:
                dur = self._vid_frames / self._video_fps()
                drop = f", {self._vid_dropped} dropped" if self._vid_dropped else ""
                self._status(f"video saved: {self._vid_path} ({self._vid_frames} frames, {dur:.1f}s{drop})", force=True)
            else:
                try:
                    self._vid_path.unlink()
                except Exception:
                    pass
        self._vid_writer = None
        self._vid_path = None
        self._vid_frames = 0

    # ------------------------------------------------------------------ playback
    def set_speed(self, value) -> None:
        """Playback speed multiplier (0.25–8). >1 skips recorded steps, <1 repeats them."""
        if value is None:
            self._status(f"playback speed x{self.play_speed:g}", force=True)
            return
        try:
            v = float(str(value).lower().lstrip("x"))
        except ValueError:
            self._status(f"speed: bad value '{value}' (e.g. speed 2, speed 0.5)", force=True)
            return
        self.play_speed = float(np.clip(v, 0.25, 8.0))
        self._status(f"playback speed set to x{self.play_speed:g} (applies to the next play)", force=True)

    def _resample_idx(self, idx: np.ndarray) -> np.ndarray:
        """Time-scale a segment: stride through the kept steps by play_speed (last step kept)."""
        n = len(idx)
        if n < 2 or abs(self.play_speed - 1.0) < 1e-6:
            return idx
        sub = np.round(np.arange(0.0, n - 1, self.play_speed)).astype(int)
        if sub.size == 0 or sub[-1] != n - 1:
            sub = np.append(sub, n - 1)
        return idx[np.clip(sub, 0, n - 1)]

    def _prepare_idx(self, actions: np.ndarray) -> np.ndarray:
        """Indices of the recorded steps that will be replayed, in order (idle compression, then speed)."""
        return self._resample_idx(self._compress_idle_idx(actions))

    def _prepare(self, actions: np.ndarray) -> np.ndarray:
        return actions[self._prepare_idx(actions)]

    DRIVE_GUARD_BEFORE_S = 6.0  # never compress this long before a wheel drive (the wheels steering into place)
    DRIVE_GUARD_AFTER_S = 3.0   # ... nor after it (brake, coast, the settled yaw the tracker compares with)

    def _compress_idle_idx(self, actions: np.ndarray) -> np.ndarray:
        """Cap runs of identical actions (planning pauses, idle tails) to idle_hold_s.

        A constant wheel command is *motion*, not a pause, and the constant steer-only stretch before it is the
        wheels aligning, so every drive run plus ``DRIVE_GUARD_BEFORE_S`` / ``DRIVE_GUARD_AFTER_S`` around it is
        replayed 1:1 (compressing those once shortened a recorded base turn: 1.7 s of driving replayed as 1.2 s,
        steering cut to 0.5 s). Until 2026-09-28 a segment with any wheel drive was left uncompressed as a whole,
        so a 75 s idle tail after a return-home run replayed for ~5 min of wall time.
        """
        n = len(actions)
        max_run = max(int(self.idle_hold_s / self.step_dt), 1)
        if n <= max_run:
            return np.arange(n)
        protect = np.zeros(n, dtype=bool)
        if actions.shape[1] >= 2:
            nz = np.abs(actions[:, -2:]).max(axis=1) > 1e-6  # chassis_drive_vel is the last two action entries
            if nz.any():
                g_before = int(round(self.DRIVE_GUARD_BEFORE_S / self.step_dt))
                g_after = int(round(self.DRIVE_GUARD_AFTER_S / self.step_dt))
                idx = np.flatnonzero(nz)
                breaks = np.flatnonzero(np.diff(idx) > 1)
                starts = np.concatenate([[idx[0]], idx[breaks + 1]])
                ends = np.concatenate([idx[breaks], [idx[-1]]])
                for s_, e_ in zip(starts, ends):
                    protect[max(int(s_) - g_before, 0):min(int(e_) + g_after, n - 1) + 1] = True
        keep = [0]
        run = 1
        for i in range(1, n):
            same = np.allclose(actions[i], actions[i - 1], atol=1e-7)
            run = run + 1 if same else 1
            if run <= max_run or protect[i]:
                keep.append(i)
        return np.asarray(keep, dtype=int)

    def _robot_yaw(self):
        """Current base yaw (rad) from the articulation data, None if unavailable."""
        try:
            q = self.env.scene["robot"].data.root_quat_w[0]
            return float(_quat_yaw(q.detach().cpu().numpy()))
        except Exception:
            return None

    def _begin_segment_locked(self, seg: Segment) -> None:
        """Prepare one segment for playback: replay indices, and the yaw tracker if the base yaw was recorded."""
        idx = self._prepare_idx(seg.actions)
        self._play_actions = seg.actions[idx]
        self._play_pos = 0
        self._tracker = None
        vs = self._velocity_slice()
        if self.yaw_track and vs is not None and seg.yaw is not None:
            tracker = YawTracker(self._play_actions, seg.yaw[idx], vs, self.step_dt)
            if tracker.n_turns > 0:
                self._tracker = tracker
                self._status(f"▶ {seg.path.name}: {tracker.n_turns} base turn run(s) tracked against the recorded yaw", force=True)
        elif self.yaw_track and seg.yaw is None and seg.drives_chassis():
            self._status(f"▶ {seg.path.name} drives the base but has no recorded base yaw: open-loop replay "
                         "(re-record this segment to get yaw tracking)", force=True)

    def _select(self, tokens) -> list[int]:
        """Resolve selectors (file numbers, ranges 'a-b', names 'seg_003[.npz]') to list positions."""
        by_num = {self._seg_number(seg.path): i for i, seg in enumerate(self.segments)}
        out: list[int] = []
        for tok in tokens:
            t = str(tok).lower().replace(".npz", "")
            if t.startswith("seg_"):
                t = t[4:]
            nums: list[int] = []
            if "-" in t and all(part.strip().isdigit() for part in t.split("-", 1)):
                a, b = (int(x) for x in t.split("-", 1))
                nums = list(range(a, b + 1)) if a <= b else list(range(a, b - 1, -1))
            elif t.isdigit():
                nums = [int(t)]
            else:
                raise ValueError(f"bad selector '{tok}'")
            for n in nums:
                if n not in by_num:
                    raise ValueError(f"no segment seg_{n:03d} (have {sorted(by_num)})")
                out.append(by_num[n])
        return out

    def play_selected(self, tokens, restore: bool) -> None:
        with self._lock:
            self._sync_from_disk_locked()
            if self._state == self.RECORDING:
                print("[Recorder] stop recording first (F5).")
                return
            try:
                sel = self._select(tokens)
            except ValueError as exc:
                self._status(f"play: {exc}", force=True)
                return
            if not sel:
                self._status("play: nothing selected", force=True)
                return
            self._play_list = sel
            self._begin_playback_locked(restore)

    def restore_selected(self, tokens) -> None:
        """Put the scene at the start state of one segment (no playback)."""
        with self._lock:
            self._sync_from_disk_locked()
            if self._state in (self.RECORDING, self.PLAYING):
                print("[Recorder] stop recording/playback first.")
                return
            try:
                sel = self._select(tokens or ["1"])
            except ValueError as exc:
                self._status(f"restore: {exc}", force=True)
                return
            seg = self.segments[sel[0]]
            if seg.snapshot:
                restore_snapshot(self.env, seg.snapshot)
            else:
                self._status(f"restore: {seg.path.name} has no scene snapshot; holding its first pose only", force=True)
            # hold the restored joint pose (first recorded action) until new live commands
            self._last_played = torch.as_tensor(seg.actions[0], dtype=torch.float32, device=self.env.device).unsqueeze(0)
            self._enter_hold_locked(f"scene restored to start of {seg.path.name}")

    def goto(self, tokens) -> None:
        with self._lock:
            self._sync_from_disk_locked()
            try:
                sel = self._select(tokens or ["1"])
            except ValueError as exc:
                self._status(f"goto: {exc}", force=True)
                return
            self.cursor = sel[0]
            self._status(f"cursor → {self.segments[self.cursor].path.name} (F8 / next plays it)", force=True)

    def list_segments(self) -> None:
        with self._lock:
            self._sync_from_disk_locked()
            lines = [f"[Recorder] {len(self.segments)} segment(s) in {self.record_dir}:"]
            for i, seg in enumerate(self.segments):
                mark = "→" if i == self.cursor else " "
                travel = seg.meta.get("action_travel")
                if travel is None:
                    travel = float(np.ptp(seg.actions, axis=0).max()) if len(seg) > 1 else 0.0
                if seg.yaw is not None:
                    base = f", base yaw delta {np.rad2deg(_wrap(seg.yaw[-1] - seg.yaw[0])):+.1f} deg (tracked on play)"
                elif seg.drives_chassis():
                    base = ", drives the base but has no recorded base yaw: open-loop replay (re-record to track)"
                else:
                    base = ""
                lines.append(
                    f"  {mark} {seg.path.name}: {len(seg)} steps = {len(seg) * self.step_dt:.1f}s sim, "
                    f"recorded {seg.meta.get('created', '?')}, wall {seg.meta.get('wall_seconds', 0):.0f}s, "
                    f"motion {travel:.3f}" + ("  ← EMPTY" if travel < 1e-4 else "") + base
                )
        print("\n".join(lines))
        try:
            self.status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            pass

    def play_all(self, restore: bool) -> None:
        with self._lock:
            self._sync_from_disk_locked()
            if not self.segments:
                self._status("nothing to play", force=True)
                return
            if self._state == self.RECORDING:
                print("[Recorder] stop recording first (R).")
                return
            self._play_list = list(range(len(self.segments)))
            self._begin_playback_locked(restore)

    def play_next(self) -> None:
        with self._lock:
            self._sync_from_disk_locked()
            if not self.segments:
                self._status("nothing to play", force=True)
                return
            if self._state == self.RECORDING:
                print("[Recorder] stop recording first (R).")
                return
            if self.cursor >= len(self.segments):
                self.cursor = 0
            self._play_list = [self.cursor]
            self.cursor += 1
            self._begin_playback_locked(restore=False)

    def _begin_playback_locked(self, restore: bool) -> None:
        first = self.segments[self._play_list[0]]
        if restore and first.snapshot:
            restore_snapshot(self.env, first.snapshot)
        self._play_seg = 0
        self._last_track_summary = ""
        self._begin_segment_locked(first)
        self._state = self.PLAYING
        names = ", ".join(self.segments[i].path.name for i in self._play_list)
        self._video_start(names)
        self._status(
            f"▶ playing {names} x{self.play_speed:g}" + (" (state restored)" if restore else ""), force=True
        )

    def stop_playback(self) -> None:
        with self._lock:
            if self._state == self.PLAYING:
                self._enter_hold_locked("playback stopped by key")

    def _velocity_slice(self) -> slice | None:
        """Indices of the chassis velocity term (held at zero, never at a stale speed)."""
        if self.action_terms and self.action_dims and len(self.action_terms) == len(self.action_dims):
            start = 0
            for term, dim in zip(self.action_terms, self.action_dims):
                if "vel" in term:
                    return slice(start, start + dim)
                start += dim
            return None
        return slice(-2, None)  # ActionsCfg default: [..., L_wheel_j2, R_wheel_j2]

    def release_hold(self) -> None:
        """Drop the post-playback hold: every joint follows the live provider again."""
        with self._lock:
            if self._hold_mask is None:
                self._status("nothing held", force=True)
                return
            self._hold_mask = self._hold_values = self._hold_ref = self._hold_seq_ref = None
            self._status("hold released by command; live provider drives all joints", force=True)

    def _held_parts(self) -> str:
        """Names of parts that still have held joints (for status lines)."""
        if self._hold_mask is None:
            return ""
        total = int(self._hold_mask.numel())
        held = self._hold_mask.reshape(-1)
        parts = []
        for attr in ("pos_mapping", "vel_mapping"):
            m = getattr(self.inner, attr, None)
            if not isinstance(m, dict):
                continue
            for part in m:
                idx = self._part_indices(part, total)
                if idx and any(bool(held[i]) for i in idx if 0 <= i < total):
                    parts.append(part)
        return ",".join(dict.fromkeys(parts)) or f"{int(held.sum())} joints"

    def _enter_hold_locked(self, why: str) -> None:
        """Leave PLAYING; pin every joint to the last replayed value until it gets a new live command."""
        self._state = self.IDLE
        self._play_actions = None
        self._tracker = None
        if self._last_track_summary:
            why = f"{why} [{self._last_track_summary}]"
            self._last_track_summary = ""
        self._video_stop()
        if self._last_played is not None:
            vals = self._last_played.detach().clone()
            vs = self._velocity_slice()
            if vs is not None:
                vals[..., vs] = 0.0
            self._hold_values = vals
            self._hold_mask = torch.ones_like(vals, dtype=torch.bool)
            self._hold_ref = None  # captured from the first live sample (fallback path)
            self._hold_seq_ref = self._command_seq()
        self._status(f"{why}; holding last replayed pose per joint until new live commands arrive", force=True)

    def _command_seq(self) -> dict | None:
        """Per-part receive counters from the live provider's ZenohManager (None if unavailable)."""
        zm = getattr(self.inner, "zenoh_manager", None)
        get = getattr(zm, "get_command_seq", None)
        try:
            return dict(get()) if callable(get) else None
        except Exception:
            return None

    def _part_indices(self, part: str, total: int) -> list[int]:
        """Absolute action indices of a part.

        ``pos_mapping`` indices are absolute (the position term comes first);
        ``vel_mapping`` indices are relative to the velocity term, which sits at
        the end of the action tensor, so they are offset by its start.
        """
        idx: list[int] = []
        pos_m = getattr(self.inner, "pos_mapping", None)
        if isinstance(pos_m, dict) and part in pos_m and "action_indices" in pos_m[part]:
            ai = pos_m[part]["action_indices"]
            idx.extend(int(i) for i in (ai.tolist() if hasattr(ai, "tolist") else ai))
        vel_m = getattr(self.inner, "vel_mapping", None)
        if isinstance(vel_m, dict) and part in vel_m and "action_indices" in vel_m[part]:
            ai = vel_m[part]["action_indices"]
            rel = [int(i) for i in (ai.tolist() if hasattr(ai, "tolist") else ai)]
            vs = self._velocity_slice()
            start = vs.start if (vs is not None and vs.start is not None and vs.start >= 0) else total - 2
            idx.extend(start + i for i in rel)
        return idx

    def _fresh_command_mask(self, like: torch.Tensor) -> Optional[torch.Tensor]:
        """Elements whose part received a new command since hold start (None = counters unavailable)."""
        if self._hold_seq_ref is None:
            return None
        seq = self._command_seq()
        if seq is None:
            return None
        fresh = torch.zeros_like(like, dtype=torch.bool)
        flat = fresh.reshape(-1)
        for part, n in seq.items():
            if n > self._hold_seq_ref.get(part, 0):
                for i in self._part_indices(part, flat.numel()):
                    if 0 <= i < flat.numel():
                        flat[i] = True
        return fresh

    def _apply_hold(self, live: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Overlay held joints on the live action; release parts that received fresh commands."""
        if self._hold_mask is None or self._hold_values is None:
            return live
        if live is None:
            return self._hold_values
        changed = self._fresh_command_mask(live)
        if changed is None:  # fallback: value change vs. the first live sample
            if self._hold_ref is None or self._hold_ref.shape != live.shape:
                self._hold_ref = live.detach().clone()
                return self._hold_values
            changed = ~torch.isclose(live, self._hold_ref, atol=1e-6)
        changed = changed & self._hold_mask
        if changed.any():
            self._hold_mask = self._hold_mask & ~changed
            n_left = int(self._hold_mask.sum())
            if n_left == 0:
                self._hold_mask = self._hold_values = self._hold_ref = self._hold_seq_ref = None
                self._status("all joints released to live commands", force=True)
                return live
            self._status(f"released {int(changed.sum())} joint(s) to live commands; {n_left} still held", force=True)
        return torch.where(self._hold_mask, self._hold_values, live)

    def _next_play_action(self) -> Optional[torch.Tensor]:
        """Return the next replayed action or None when the playlist is exhausted.

        With a yaw tracker the replayed wheel command is scaled by the base's yaw lag, and at the end of a
        recorded turn extra drive steps are inserted (cursor not advanced) until the recorded yaw is reached.
        """
        while True:
            if self._play_actions is not None and self._play_pos < len(self._play_actions):
                pos = self._play_pos
                a = self._play_actions[pos]
                advance = True
                if self._tracker is not None:
                    yaw_now = self._robot_yaw()
                    if yaw_now is not None:
                        a, advance, msg = self._tracker.step(pos, a, yaw_now)
                        if msg:
                            self._status(f"▶ {msg}", force=True)
                if advance:
                    self._play_pos += 1
                    if self._play_pos % 100 == 0 or self._play_pos == len(self._play_actions):
                        seg = self.segments[self._play_list[self._play_seg]]
                        lag = f" | yaw lag {np.rad2deg(self._tracker.last_err):+.1f} deg" if self._tracker is not None else ""
                        self._status(f"▶ {seg.path.name} {self._play_pos}/{len(self._play_actions)}{lag}")
                return torch.as_tensor(a, dtype=torch.float32, device=self.env.device).unsqueeze(0)
            if self._tracker is not None:
                seg = self.segments[self._play_list[self._play_seg]]
                yaw_now = self._robot_yaw()
                if yaw_now is not None:
                    self._last_track_summary = f"{seg.path.name} {self._tracker.summary(yaw_now)}"
                    self._status(f"▶ {self._last_track_summary}", force=True)
                self._tracker = None
            self._play_seg += 1
            if self._play_seg >= len(self._play_list):
                return None
            self._begin_segment_locked(self.segments[self._play_list[self._play_seg]])

    # ------------------------------------------------------------------ provider API
    def get_action(self, env) -> Optional[torch.Tensor]:
        self._poll_command_file()
        if self._pending_keys:
            self._drain_keys()
        with self._lock:
            if self._state == self.PLAYING:
                a = self._next_play_action()
                if a is not None:
                    self._last_played = a
                    self._video_capture()
                    self._notify_action(a)
                    return a
                self._enter_hold_locked("playback finished")

        live = self.inner.get_action(env)

        with self._lock:
            out = self._apply_hold(live)
            if self._state == self.RECORDING and out is not None:
                self._rec_actions.append(out.detach().reshape(-1).cpu().numpy().copy())
                try:
                    self._rec_qpos.append(env.scene["robot"].data.joint_pos[0].detach().cpu().numpy().copy())
                except Exception:
                    self._rec_qpos.append(np.zeros(len(self.joint_names), dtype=np.float32))
                try:  # base pose per step: lets playback track a recorded base turn (YawTracker)
                    self._rec_root.append(env.scene["robot"].data.root_state_w[0].detach().cpu().numpy().copy())
                except Exception:
                    pass
                if len(self._rec_actions) % 200 == 0:
                    self._status(f"REC ● {len(self._rec_actions)} steps ({len(self._rec_actions) * self.step_dt:.1f}s sim)")
        self._notify_action(out)
        return out

    def _notify_action(self, action: Optional[torch.Tensor]) -> None:
        """Tell the live provider which action is actually being sent (gripper lock bookkeeping)."""
        if callable(self._observe):
            try:
                self._observe(action)
            except Exception:
                pass

    # ------------------------------------------------------------------ status
    def _status(self, msg: str, force: bool = False) -> None:
        held = int(self._hold_mask.sum()) if self._hold_mask is not None else 0
        line = (
            f"[Recorder] {msg} | state={self._state}"
            + (f" hold={held} joints [{self._held_parts()}]" if held else "")
            + f" segments={len(self.segments)} cursor={self.cursor}"
        )
        if force:
            print(line)
        try:
            self.status_path.write_text(line + "\n", encoding="utf-8")
        except OSError:
            pass
