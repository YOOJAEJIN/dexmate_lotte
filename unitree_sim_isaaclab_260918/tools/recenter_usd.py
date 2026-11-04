#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Recenter a USD asset so the root prim's origin sits at the geometry bounding-box center.

Optionally bake a --roll/--pitch/--yaw rotation into the geometry first: mesh
points/normals/extents are rotated directly so every authored transform can stay at
its default (0, 0, 0) value, then the result is recentered.

Mesh points/extents are offset directly (no xformOps are added for meshes), so physics
properties stay intact; the offset is transformed into each mesh's local space, so
transformed hierarchies are supported. Implicit prims (Cube, Sphere, ...) such as
collision geometry cannot be point-offset; those are moved with a prepended
"xformOp:translate:recenter" op so they follow the new geometry. Baking rotations
works for Cube (multiples of 90 deg) and Sphere; other implicit prims are rejected.

Requires a python with USD bindings (pxr), e.g.:
    /home/ubuntu/isaacsim5_1/kit/python/bin/python3 tools/recenter_usd.py <asset.usd>

Usage:
    python3 recenter_usd.py <usd_path> [--prim /Root] [--xy-only] [--dry-run]
    python3 recenter_usd.py <usd_path> --roll=90    # rotate 90 deg about X, bake, recenter
"""

import argparse

from pxr import Usd, UsdGeom, Gf, Sdf


def _xform_point(m, v):
    """row-vector transform of a point: p' = p * m (with translation)"""
    return type(v)(*(sum(v[k] * m[k][j] for k in range(3)) + m[3][j] for j in range(3)))


def _xform_dir(m, v):
    """row-vector transform of a direction: p' = p * m (linear part only)"""
    return type(v)(*(sum(v[k] * m[k][j] for k in range(3)) for j in range(3)))


def _is_axis_permutation(r3, tol=1e-6):
    """true if the rotation maps every coordinate axis onto another (possibly flipped) axis"""
    for i in range(3):
        big = [abs(r3[i][j]) for j in range(3) if abs(r3[i][j]) > tol]
        if len(big) != 1 or abs(big[0] - 1.0) > tol:
            return False
    return True


def bake_rotation(root, r4):
    """Bake a root-space rotation (Gf.Matrix4d) into the geometry so all authored
    transforms can keep their default (0, 0, 0) values."""
    r3 = r4.ExtractRotationMatrix()
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    n = 0
    for prim in Usd.PrimRange(root):
        t = prim.GetTypeName()
        if t == "Mesh":
            m = xc.GetLocalToWorldTransform(prim)
            a = m * r4 * m.GetInverse()                                 # points: p' = p * a
            l = m.ExtractRotationMatrix()
            n3 = l.GetInverse().GetTranspose() * r3 * l.GetTranspose()  # normals: n' = n * n3
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            if not pts:
                continue
            mesh.GetPointsAttr().Set([_xform_point(a, v) for v in pts])
            nattr = mesh.GetNormalsAttr()
            nrms = nattr.Get()
            if nrms:
                out = []
                for v in nrms:
                    r = _xform_dir(n3, v)
                    ln = r.GetLength()
                    out.append(type(v)(r[0] / ln, r[1] / ln, r[2] / ln) if ln > 0 else v)
                nattr.Set(out)
            ext = mesh.GetExtentAttr().Get()
            if ext:
                corners = [_xform_point(a, type(ext[0])(x, y, z))
                           for x in (ext[0][0], ext[1][0])
                           for y in (ext[0][1], ext[1][1])
                           for z in (ext[0][2], ext[1][2])]
                mesh.GetExtentAttr().Set([
                    type(ext[0])(min(c[0] for c in corners), min(c[1] for c in corners), min(c[2] for c in corners)),
                    type(ext[1])(max(c[0] for c in corners), max(c[1] for c in corners), max(c[2] for c in corners))])
            n += 1
        elif t == "Sphere":
            pass  # rotationally symmetric, nothing to bake
        elif t == "Cube":
            if not _is_axis_permutation(r3):
                raise SystemExit(f"{prim.GetPath()}: a Cube can only bake multiples of 90 deg "
                                 "(or convert it to a Mesh first)")
            # rewrite the implicit cube so it represents the rotated (still axis-aligned) box
            m = xc.GetLocalToWorldTransform(prim)
            size = prim.GetAttribute("size").Get() or 2.0
            c = Gf.Vec3d(m[3][0], m[3][1], m[3][2])
            h = [0.5 * size * sum(m[k][j] ** 2 for k in range(3)) ** 0.5 for j in range(3)]
            c2 = _xform_dir(r3, c)                                                # rotated center
            h2 = [abs(sum(h[k] * r3[k][j] for k in range(3))) for j in range(3)]  # rotated half-extents
            xf = UsdGeom.Xformable(prim)
            ops = xf.GetOrderedXformOps()
            scale_ops = [o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeScale]
            if not scale_ops or not ops or ops[0].GetOpType() != UsdGeom.XformOp.TypeTranslate:
                raise SystemExit(f"{prim.GetPath()}: expected a scale and an outermost translate xformOp on the Cube")
            sv = scale_ops[0].GetAttr().Get()
            scale_ops[0].GetAttr().Set(type(sv)(*(h2[j] / (0.5 * size) for j in range(3))))
            # the scale swap shifts the box under the pivot ops, so re-measure the center
            # and move the outermost translate to place it at the rotated center c2
            xc.Clear()
            m2 = xc.GetLocalToWorldTransform(prim)
            c_swap = Gf.Vec3d(m2[3][0], m2[3][1], m2[3][2])
            tv = ops[0].GetAttr().Get()
            ops[0].GetAttr().Set(type(tv)(*(tv[j] + c2[j] - c_swap[j] for j in range(3))))
        elif t in ("Cylinder", "Cone", "Capsule"):
            raise SystemExit(f"{prim.GetPath()}: baking rotations into {t} prims is not supported; "
                             "convert it to a Mesh first")
    print(f"baked rotation into {n} meshes under {root.GetPath()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("usd_path", help="path to the USD file (modified in place unless --dry-run)")
    parser.add_argument("--prim", default=None, help="prim path to recenter (default: stage default prim)")
    parser.add_argument("--xy-only", action="store_true", help="only center x/y, keep z (e.g. bottom face stays z=0)")
    parser.add_argument("--dry-run", action="store_true", help="only print the offset, do not save")
    parser.add_argument("--roll", type=float, default=0.0, help="degrees to bake around the X axis (geometry rotates, transforms stay 0)")
    parser.add_argument("--pitch", type=float, default=0.0, help="degrees to bake around the Y axis")
    parser.add_argument("--yaw", type=float, default=0.0, help="degrees to bake around the Z axis")
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.usd_path)
    root = stage.GetPrimAtPath(args.prim) if args.prim else stage.GetDefaultPrim()
    if not root.IsValid():
        raise SystemExit(f"prim not found: {args.prim or '<default prim>'}")

    # optionally bake a rotation into the geometry first (around the root origin,
    # applied roll -> pitch -> yaw); the recenter below then works on the rotated result
    if args.roll or args.pitch or args.yaw:
        r4 = Gf.Matrix4d(1.0)
        for axis, deg in ((Gf.Vec3d(1, 0, 0), args.roll), (Gf.Vec3d(0, 1, 0), args.pitch), (Gf.Vec3d(0, 0, 1), args.yaw)):
            if deg:
                r4 = r4 * Gf.Matrix4d().SetRotate(Gf.Rotation(axis, deg))
        bake_rotation(root, r4)

    # current bounding box of all geometry under the root (visual/default purpose only,
    # so guide-purpose collision geometry does not skew the centering)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], False)
    bbox = cache.ComputeWorldBound(root).GetRange()
    lo, hi = bbox.GetMin(), bbox.GetMax()
    center = (lo + hi) / 2.0
    offset = Gf.Vec3f(-center[0], -center[1], 0.0 if args.xy_only else -center[2])

    print(f"root prim      : {root.GetPath()}")
    print(f"current bbox   : ({lo[0]:.4f}, {lo[1]:.4f}, {lo[2]:.4f}) ~ ({hi[0]:.4f}, {hi[1]:.4f}, {hi[2]:.4f})")
    print(f"bbox center    : ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    print(f"point offset   : ({offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f})")

    if args.dry_run:
        print("dry-run: not saving")
        return

    # apply the offset to all geometry under the root:
    #  - Mesh prims: points/extents are offset directly; the world-space offset is
    #    transformed into each mesh's local space, so parent transforms are fine
    #  - implicit prims (Cube/Sphere/...): geometry is defined by size/radius attrs, so
    #    they are moved with a prepended translate xformOp instead of point offsets
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    recenter_op = "xformOp:translate:recenter"
    n_meshes = n_implicits = 0
    for prim in Usd.PrimRange(root):
        if prim.GetPath() == root.GetPath():
            continue
        if prim.GetTypeName() == "Mesh":
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            if not pts:
                continue
            d = xc.GetLocalToWorldTransform(prim).GetInverse().TransformDir(offset)
            new_pts = [type(v)(v[0] + d[0], v[1] + d[1], v[2] + d[2]) for v in pts]
            mesh.GetPointsAttr().Set(new_pts)
            ext = mesh.GetExtentAttr().Get()
            if ext:
                mesh.GetExtentAttr().Set([type(ext[0])(ext[0][0] + d[0], ext[0][1] + d[1], ext[0][2] + d[2]),
                                          type(ext[1])(ext[1][0] + d[0], ext[1][1] + d[1], ext[1][2] + d[2])])
            n_meshes += 1
        elif prim.GetTypeName() in ("Cube", "Sphere", "Cylinder", "Cone", "Capsule"):
            xf = UsdGeom.Xformable(prim)
            # a translate op prepended (outermost) shifts the prim by d in its parent's frame
            d = xc.GetParentToWorldTransform(prim).GetInverse().TransformDir(offset)
            attr = prim.GetAttribute(recenter_op)
            if attr and attr.IsValid() and recenter_op in [op.GetName() for op in xf.GetOrderedXformOps()]:
                prev = attr.Get()
                attr.Set(Gf.Vec3d(prev[0] + d[0], prev[1] + d[1], prev[2] + d[2]))
            else:
                attr = prim.CreateAttribute(recenter_op, Sdf.ValueTypeNames.Double3)
                attr.Set(Gf.Vec3d(d[0], d[1], d[2]))
                xf.SetXformOpOrder([UsdGeom.XformOp(attr)] + xf.GetOrderedXformOps())
            n_implicits += 1

    if n_meshes + n_implicits == 0:
        print("warning: no meshes or implicit geometry found under the root; nothing was changed")

    stage.Save()
    print(f"saved: offset applied to {n_meshes} meshes and {n_implicits} implicit prims "
          f"under {root.GetPath()} (origin moved to bbox center)")


if __name__ == "__main__":
    main()
