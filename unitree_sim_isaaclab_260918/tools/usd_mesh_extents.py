#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Print per-mesh and overall extents of a USD asset without launching Isaac Sim.

Binary .usd (usdc) files cannot be read as text, and the project venv has no ``pxr``
outside a running Kit app. Use the bindings shipped with the Isaac Sim source build::

    R=/home/jjy/Desktop/vscode/dexmate/isaacsim/_build/linux-x86_64/release
    EXT=$(ls -d $R/extscache/omni.usd.libs-*/ | head -1)
    PYTHONPATH=$EXT LD_LIBRARY_PATH=$EXT/bin:$R/kit $R/kit/python/bin/python3 \\
        tools/usd_mesh_extents.py assets/objects/workspace_with_chilsung/card_box.usd

Output: stage units, every Mesh prim's bbox in the asset frame (local transforms applied),
the overall bbox, and for each of the four bbox side faces the extent of the vertices lying
on it (i.e. how wide that edge really is). Vertices are grouped per x level as a compact
top-view outline.
"""
import sys
from collections import defaultdict

from pxr import Gf, Usd, UsdGeom


def main(path: str) -> None:
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise SystemExit(f"cannot open {path}")
    print(f"{path}\n  metersPerUnit={UsdGeom.GetStageMetersPerUnit(stage)} upAxis={UsdGeom.GetStageUpAxis(stage)} "
          f"defaultPrim={stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    pts_all: list = []
    print("per-mesh bbox (m):")
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        xf = cache.GetLocalToWorldTransform(prim)
        pts = [xf.Transform(Gf.Vec3d(p)) for p in (UsdGeom.Mesh(prim).GetPointsAttr().Get() or [])]
        if not pts:
            continue
        pts_all += pts
        lo = [min(p[i] for p in pts) for i in range(3)]
        hi = [max(p[i] for p in pts) for i in range(3)]
        print(f"  {str(prim.GetPath()):50s} x [{lo[0]:+.4f},{hi[0]:+.4f}] y [{lo[1]:+.4f},{hi[1]:+.4f}] "
              f"z [{lo[2]:+.4f},{hi[2]:+.4f}]  ({len(pts)} pts)")
    if not pts_all:
        raise SystemExit("no Mesh prims with points found")
    lo = [min(p[i] for p in pts_all) for i in range(3)]
    hi = [max(p[i] for p in pts_all) for i in range(3)]
    print(f"overall bbox: x [{lo[0]:+.4f},{hi[0]:+.4f}] y [{lo[1]:+.4f},{hi[1]:+.4f}] z [{lo[2]:+.4f},{hi[2]:+.4f}]"
          f"  size {hi[0]-lo[0]:.4f} x {hi[1]-lo[1]:.4f} x {hi[2]-lo[2]:.4f} m")
    tol = 5e-4
    for label, axis, value, other in (("-X", 0, lo[0], 1), ("+X", 0, hi[0], 1), ("-Y", 1, lo[1], 0), ("+Y", 1, hi[1], 0)):
        on = [p[other] for p in pts_all if abs(p[axis] - value) < tol]
        print(f"  {label} face ({'xyz'[axis]}={value:+.4f}): {len(on)} vertices, "
              f"{'xyz'[other]} range [{min(on):+.4f},{max(on):+.4f}] -> edge width {max(on)-min(on):.4f} m")
    levels = defaultdict(list)
    for p in pts_all:
        levels[round(p[0], 3)].append(p[1])
    print("top-view outline, vertices grouped by x level: x -> y extent")
    for x in sorted(levels):
        print(f"  x={x:+.3f}: y [{min(levels[x]):+.4f},{max(levels[x]):+.4f}]  ({len(levels[x])} pts)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
