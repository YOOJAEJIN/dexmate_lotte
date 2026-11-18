"""Box front face from the mask's depth points - independent of where the camera is.

`rect()` (needs gravity): the front face of an upright pile is a VERTICAL plane.
Fit it among the mask points, then take the face's extent from the inlier
density along its horizontal and vertical axes. What that buys (all seen
2026-11-12 with the torso raised and the head looking 31 deg down):
  - the top face is visible and FastSAM masks it together with the front face;
    its points are off the vertical plane, so they no longer add height
    (hull corners + depth gave 50-67 cm for a 45.5 cm pile);
  - a gripper in front of a corner is off the plane too, and the edges come
    from the rows / columns that are still visible;
  - width and height are measured along true horizontal / vertical, so they do
    not change with the camera's height or tilt.

`fit()` / `ray_hit()` (no gravity): free RANSAC plane, used as the fallback.

Run `python face_plane.py` for the self-check.
"""
import numpy as np

INLIER_M = 0.015  # point-to-plane distance counted as on the face
EDGE_FRAC = 0.35  # density (x typical) that still counts as face: an edge may be up to ~65 % hidden


def fit(P, iters=200, sample=3000, seed=0):
    """P: (N,3) points -> (n, c) unit normal + point on plane, or None."""
    if len(P) < 50:
        return None
    return _ransac(P, INLIER_M, iters, sample, seed)


def _ransac(P, tol, iters, sample, seed):
    """Hyperplane in R^k (k = 2: line, k = 3: plane) -> (unit normal, point)."""
    k = P.shape[1]
    rng = np.random.default_rng(seed)
    S = P if len(P) <= sample else P[rng.choice(len(P), sample, replace=False)]
    i = rng.integers(0, len(S), size=(iters, k))
    A = S[i[:, 0]]
    if k == 3:
        N = np.cross(S[i[:, 1]] - A, S[i[:, 2]] - A)
    else:
        D = S[i[:, 1]] - A
        N = np.stack([-D[:, 1], D[:, 0]], axis=1)
    nn = np.linalg.norm(N, axis=1)
    ok = nn > 1e-9
    if not ok.any():
        return None
    N, A = N[ok] / nn[ok, None], A[ok]
    dist = np.abs(S @ N.T - np.sum(N * A, axis=1))  # (len(S), iters)
    best = int(np.argmax((dist < tol).sum(axis=0)))
    inl = P[np.abs((P - A[best]) @ N[best]) < tol]
    if len(inl) < 30:
        return None
    m = inl.mean(axis=0)
    n = np.linalg.svd(inl - m, full_matrices=False)[2][-1]  # least-squares normal
    off = inl @ n
    return n, m + n * (np.median(off) - m @ n)              # median offset


def ray_hit(ray, plane):
    """Point where the camera ray (through the origin) meets the plane, or None."""
    n, c = plane
    den = float(n @ ray)
    if abs(den) < 1e-6:
        return None
    t = float(n @ c) / den
    return ray * t if t > 0 else None


def _span(vals, px_m, frac=EDGE_FRAC):
    """(lo, hi) of the longest stretch where the density of vals stays >= frac x typical.

    Density = 2 mm histogram box-filtered over ~3 pixel footprints (px_m), so
    the pixel grid leaves no empty bins. A clean step edge crosses 50 % of the
    typical density exactly at the edge; `shift` moves the frac crossing there.
    """
    step = 0.002
    w = max(3, int(round(3 * px_m / step)))
    lo = float(vals.min()) - w * step
    nb = int(np.ceil((float(vals.max()) - lo) / step)) + w
    cnt = np.histogram(vals, bins=nb, range=(lo, lo + nb * step))[0]
    dens = np.convolve(cnt, np.ones(w) / w, mode="same")
    idx = np.nonzero(dens > 0)[0]
    typ = np.median(dens[idx[len(idx) // 4]: idx[3 * len(idx) // 4] + 1])
    ok = np.append(dens >= frac * typ, False)
    best, start = (0, 0), None
    for i, o in enumerate(ok):
        if o and start is None:
            start = i
        elif not o and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    shift = (0.5 - frac) * w * step
    return lo + best[0] * step + shift, lo + best[1] * step - shift


def rect(rays, z, up, px_m, tol=INLIER_M, iters=200, sample=3000, seed=0):
    """Front-face rectangle of an upright box.

    rays: (N,3) camera rays of ALL mask pixels, as [x/z, y/z, 1]. z: (N,) their
    depth in metres (<= 0 where invalid). up: world-up in the camera frame.
    px_m: size of one pixel at the box (depth / focal length, metres).

    Depth decides only what a pixel-accurate mask cannot: where the plane is,
    and where the front face ends and the top face begins. The sides and the
    bottom come from the mask rays cut by the plane - stereo depth bleeds for
    several pixels at a silhouette (2026-11-12: 2 cm lost on the left edge).
    -> (corners (4,3) ordered BR, BL, TL, TR, width_m, height_m) or None
    """
    up = np.asarray(up, float) / np.linalg.norm(up)
    fwd = np.array([0.0, 0.0, 1.0]) - up[2] * up  # optical axis flattened onto the ground
    if np.linalg.norm(fwd) < 0.2:                 # looking straight down / up
        return None
    fwd /= np.linalg.norm(fwd)
    down = -up
    right = np.cross(down, fwd)
    P = rays[z > 0] * z[z > 0, None]
    if len(P) < 50:
        return None
    xd = np.stack([P @ right, P @ fwd], axis=1)   # top view: lateral, distance
    line = _ransac(xd, tol, iters, sample, seed)  # vertical plane = line in the top view
    if line is None:
        return None
    n, c = line
    if n[1] < 0:
        n = -n                                    # normal points away from the camera
    off = float(n @ c)
    e = np.array([n[1], -n[0]])                   # along the face, towards image-right
    inl = np.abs(xd @ n - off) < tol
    if inl.sum() < 50:
        return None
    h_top = _span((P @ down)[inl], px_m)[0]       # depth: front face starts here

    # every mask pixel's ray cut by the plane -> (a, h) on the face
    den = rays @ (n[0] * right + n[1] * fwd)
    ok = den > 1e-6
    if ok.sum() < 30:                             # rays (almost) never meet the plane: bad fit
        return None                               # (2026-11-13: empty h crashed seg_overlay)
    hit = rays[ok] * (off / den[ok])[:, None]
    a, h = np.stack([hit @ right, hit @ fwd], axis=1) @ e, hit @ down
    if h_top > 0:                                 # camera above the pile: the mask also holds
        keep = h >= h_top                         # the top face, whose rays land above the edge
        if keep.sum() < 30:
            return None
        a, h = a[keep], h[keep]
        h_bot = _span(h, px_m)[1]
    else:                                         # top edge is a silhouette: the mask is sharper
        h_top, h_bot = _span(h, px_m)
    rows = (h >= h_top) & (h <= h_bot)
    if rows.sum() < 30:
        return None
    a0, a1 = _span(a[rows], px_m)
    if a1 - a0 < 4 * px_m or h_bot - h_top < 4 * px_m:
        return None

    def corner(a_, h_):
        x, d = a_ * e + off * n
        return x * right + h_ * down + d * fwd

    return (np.array([corner(a1, h_bot), corner(a0, h_bot), corner(a0, h_top), corner(a1, h_top)]),
            a1 - a0, h_bot - h_top)


def _demo_box(pitch_deg, yaw_deg=0.0, cam_h=1.6, gripper=True, seed=1):
    """Synthetic 40.5 x 45.5 cm pile (top at 1.375 m) as a camera at cam_h samples it.

    Includes what broke the earlier versions: the top sheet in the mask, a
    gripper 0.3 m in front of the top-left corner, and depth bleeding to the
    background along the left silhouette.
    """
    rng = np.random.default_rng(seed)
    W, Hh, Dp, dist, px = 0.405, 0.455, 0.30, 1.16, 0.005
    yaw = np.radians(yaw_deg)
    ax = np.array([np.cos(yaw), np.sin(yaw), 0.0])            # along the face (world, z up)
    nrm = np.array([-np.sin(yaw), np.cos(yaw), 0.0])          # into the pile
    o = np.array([0.0, dist, 1.375]) - ax * W / 2             # front-top-left corner
    cam = np.array([0.0, 0.0, cam_h])
    th = np.radians(pitch_deg)
    zc = np.array([0.0, np.cos(th), -np.sin(th)]); xc = np.array([1.0, 0, 0]); yc = np.cross(zc, xc)
    R = np.stack([xc, yc, zc])                                # world -> camera (x right, y down, z fwd)
    g = np.mgrid[0:W:px, 0:Hh:px].reshape(2, -1).T + px / 2   # pixel-like grid on the front face
    face = o + g[:, :1] * ax - g[:, 1:] * [0, 0, 1]
    seen = face.copy()                                        # what the depth map reports per pixel
    seen[g[:, 0] < 0.02] += nrm * 0.04                        # left silhouette bleeds towards the wall
    if gripper:
        hid = (g[:, 0] < 0.10) & (g[:, 1] < 0.12)             # finger hides the top-left 10 x 12 cm
        seen[hid] = cam + (face[hid] - cam) * 0.72            # ... and is measured instead
    pts, meas = [face], [seen + rng.normal(0, 0.003, face.shape)]
    if cam_h > 1.375:                                         # top sheet visible from above
        t = np.mgrid[0:W:px, 0:Dp:px * 3].reshape(2, -1).T + px / 2
        top = o + t[:, :1] * ax + t[:, 1:] * nrm
        pts.append(top); meas.append(top + rng.normal(0, 0.003, top.shape))
    pc, mc = (np.vstack(pts) - cam) @ R.T, (np.vstack(meas) - cam) @ R.T
    truth = [o + ax * W - [0, 0, Hh], o - [0, 0, Hh], o, o + ax * W]  # BR, BL, TL, TR
    return pc / pc[:, 2:3], mc[:, 2], R @ [0, 0, 1.0], (np.array(truth) - cam) @ R.T, px


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    hit = lambda p, pl: ray_hit(p / p[2], pl)  # noqa: E731  ray through a true point

    # 1) free plane: face 1.2 m away tilted 30 deg, 40 x 45 cm, gripper 0.3 m in front
    a = np.radians(30)
    ex, ey = np.array([1.0, 0, 0]), np.array([0, np.cos(a), np.sin(a)])
    o = np.array([-0.2, -0.2, 1.2])
    uv = rng.uniform(0, 1, size=(4000, 2)) * [0.40, 0.45]
    face = o + uv[:, :1] * ex + uv[:, 1:] * ey + rng.normal(0, 0.003, (4000, 3))
    grip = o + rng.uniform(0, 0.08, (900, 3)) * [1, 1, 0] - [0, 0, 0.3]
    pl = fit(np.vstack([face, grip]))
    assert abs(abs(pl[0] @ np.cross(ex, ey)) - 1) < 1e-3
    tl, bl = hit(o, pl), hit(o + 0.45 * ey, pl)
    assert np.linalg.norm(tl - o) < 0.01 and abs(np.linalg.norm(bl - tl) - 0.45) < 0.01
    assert fit(face[:10]) is None

    # 2) rect: the same pile from different camera heights / tilts / yaw -> the same box
    for kw in (dict(pitch_deg=31, cam_h=1.60), dict(pitch_deg=0, cam_h=1.20), dict(pitch_deg=45, cam_h=1.90),
               dict(pitch_deg=20, yaw_deg=20, cam_h=1.5), dict(pitch_deg=31, cam_h=1.60, gripper=False)):
        rays, z, up, truth, px = _demo_box(**kw)
        res = rect(rays, z, up, px)
        assert res is not None, kw
        corners, w, h = res
        assert abs(w - 0.405) < 0.008 and abs(h - 0.455) < 0.008, (kw, w, h)
        assert np.abs(corners - truth).max() < 0.012, (kw, np.abs(corners - truth).max())
    assert rect(rays[:10], z[:10], up, px) is None
    print("face_plane self-check OK")
