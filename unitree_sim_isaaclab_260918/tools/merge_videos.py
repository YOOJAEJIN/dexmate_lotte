#!/usr/bin/env python3
"""Concatenate the playback videos in recordings/videos into one mp4.

Files are taken in name order (the recorder names them
``<timestamp>_seg_NNN_xS.mp4``, so that is segment order). Previously merged
files (``merged_*.mp4``) are skipped. When every input shares the same codec,
resolution and frame rate the streams are copied without re-encoding;
otherwise (or with ``--reencode``) everything is scaled to the first video's
size and re-encoded with H.264.

Examples::

    python tools/merge_videos.py                       # all videos -> merged_<stamp>.mp4
    python tools/merge_videos.py --select 1-8          # only seg_001 .. seg_008
    python tools/merge_videos.py --out demo.mp4
    python tools/merge_videos.py --dir /path/to/videos --reencode
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Same default as sim_main.py --record_dir: ../dexcontrol/recordings, else ./recordings.
_RECORD_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "dexcontrol", "recordings")
if not os.path.isdir(_RECORD_DIR):
    _RECORD_DIR = os.path.join(PROJECT_ROOT, "recordings")
DEFAULT_DIR = os.path.join(_RECORD_DIR, "videos")


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def probe(ff: str, path: str) -> tuple[str, str, str]:
    """(codec, WxH, fps) of the first video stream via ffmpeg -i."""
    out = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    m = re.search(r"Video: (\w+).*?, (\d+x\d+).*?, ([\d.]+) fps", out)
    return (m.group(1), m.group(2), m.group(3)) if m else ("?", "?", "?")


def seg_number(name: str) -> int | None:
    m = re.search(r"seg_(\d+)", name)
    return int(m.group(1)) if m else None


def parse_select(spec: str) -> set[int]:
    nums: set[int] = set()
    for tok in spec.replace(",", " ").split():
        if "-" in tok:
            a, b = (int(x) for x in tok.split("-", 1))
            nums.update(range(min(a, b), max(a, b) + 1))
        else:
            nums.add(int(tok))
    return nums


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR, help="folder with the playback mp4 files")
    ap.add_argument("--out", default=None, help="output file (default: <dir>/merged_<stamp>.mp4)")
    ap.add_argument("--select", default=None, help="segment numbers to include, e.g. '1-8' or '1 3 5'")
    ap.add_argument("--reencode", action="store_true", help="force re-encoding (needed if formats differ)")
    args = ap.parse_args()

    files = sorted(
        f for f in os.listdir(args.dir)
        if f.lower().endswith(".mp4") and not f.startswith("merged_")
    )
    if args.select:
        wanted = parse_select(args.select)
        files = [f for f in files if seg_number(f) in wanted]
    if not files:
        print("no input videos found")
        return 1

    ff = ffmpeg_exe()
    fmts = [probe(ff, os.path.join(args.dir, f)) for f in files]
    same = len(set(fmts)) == 1
    for f, fm in zip(files, fmts):
        print(f"  {f}  {fm[0]} {fm[1]} {fm[2]}fps")

    out = args.out or os.path.join(args.dir, f"merged_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as lst:
        for f in files:
            lst.write(f"file '{os.path.abspath(os.path.join(args.dir, f))}'\n")
        list_path = lst.name

    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", list_path]
    if same and not args.reencode:
        cmd += ["-c", "copy", "-movflags", "+faststart", out]
        mode = "stream copy"
    else:
        w, h = fmts[0][1].split("x")
        cmd += ["-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps={fmts[0][2]}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
        mode = "re-encode"
    res = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(list_path)
    if res.returncode != 0:
        print(res.stderr.strip())
        return res.returncode
    dur = subprocess.run([ff, "-hide_banner", "-i", out], capture_output=True, text=True).stderr
    m = re.search(r"Duration: ([\d:.]+)", dur)
    print(f"merged {len(files)} file(s) ({mode}) → {out}  duration {m.group(1) if m else '?'}  "
          f"{os.path.getsize(out) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
