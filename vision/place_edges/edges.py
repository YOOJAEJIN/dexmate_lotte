"""Place-target edges on one HD1200 frame (Targeting System, EDGE mode).

Same algorithm as target_edges.py (confirmed with the user 2026-11-18), as a function:
  V2 = right edge of the leaning front strip (dark -> bright; V1 = shadow edge, not used)
  H5 = strip top (walk up from inside the strip; the thin groove line above is ignored)
  V3 = target back edge (saturation step to the background)
  D  = edge from the H5 x V2 corner back along the top of the V2..V3 face, moved off the
       lit highlight onto the highlight -> face outline
  K  = D x V3 extended (back-top corner)
detect(im, d, R, t): im BGR 1920x1200, d metric depth (m), R/t = base <- optical camera.
Returns a dict (pixels are HD1200, xyz in the base frame) or {"ok": False, "why": ...}.
Run `python edges.py` for the self-check against saved frames (target_edges.py numbers).
"""
import cv2
import numpy as np

FX, FY, CX, CY = 735.25, 735.141, 992.571, 618.88            # factory HD1200 intrinsics
U0, U1, V0, V1 = 700, 870, 830, 1045  # ponytail: fixed ROI on the confirmed target; auto when the spot moves
STRIP_W = 16  # ponytail: strip width in HD px at ~1.1 m; derive from depth if the distance changes


REF_CORNER_PX = (743.0, 900.0)  # where the H5 x V2 corner sat when the ROI / ranges above were tuned


def detect(im, d, R, t, shift=(0, 0)):
    """shift: the caller translated im / d by (du, dv) px so the target sits in the tuned ROI;
    rays undo it and returned pixels are in the caller's (unshifted) image."""
    ray = lambda u, v: R @ np.array([(u - shift[0] - CX) / FX, (v - shift[1] - CY) / FY, 1.0])  # noqa: E731

    def xyz(u, v, k=2):
        z = d[int(v) - k:int(v) + k + 1, int(u) - k:int(u) + k + 1].ravel(); z = z[(z > 0.3) & (z < 3)]
        return t + np.min(z) * ray(u, v) if z.size else None  # min: the near (target) side of an edge

    def u_at(s, v):
        return s[0] + (s[2] - s[0]) * (v - s[1]) / max(1e-6, s[3] - s[1])

    def line3d(s, n=12):
        P = [p for k in np.linspace(0.1, 0.9, n) if (p := xyz(s[0] + k * (s[2] - s[0]), s[1] + k * (s[3] - s[1]))) is not None]
        P = np.array(P); c = P.mean(0); dvec = np.linalg.svd(P - c)[2][0]; dvec *= np.sign(dvec[2]) or 1
        return c, dvec, P

    def closest(p1, u1, p2, u2):  # closest points of two 3D lines
        w = p1 - p2; a_, b__, c_ = u1 @ u1, u1 @ u2, u2 @ u2; d_, e_ = u1 @ w, u2 @ w; den = a_ * c_ - b__ * b__
        s1, s2 = (b__ * e_ - c_ * d_) / den, (a_ * e_ - b__ * d_) / den
        return p1 + s1 * u1, p2 + s2 * u2

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    L = cv2.createLineSegmentDetector().detect(gray[V0:V1, U0:U1])[0]
    if L is None:
        return {"ok": False, "why": "no line segments in the ROI"}
    segs = []
    for x0, y0, x1, y1 in L.reshape(-1, 4) + [U0, V0, U0, V0]:
        if y1 < y0: x0, y0, x1, y1 = x1, y1, x0, y0
        ln = np.hypot(x1 - x0, y1 - y0); a = np.degrees(np.arctan2(y1 - y0, x1 - x0))  # 90 = vertical
        if ln >= 12: segs.append((x0, y0, x1, y1, ln, a))
    near_v = [s for s in segs if abs(s[5] - 90) < 25]
    G0 = gray.astype(np.float32)

    def polarity(s, k=4):  # mean(right side) - mean(left side) along the segment
        vs = np.linspace(s[1], s[3], 10); us = [u_at(s, v) for v in vs]
        return np.mean([G0[int(v), int(u) + k] - G0[int(v), int(u) - k] for u, v in zip(us, vs)])

    # V2 = gray strip (left) -> brighter face (right); the shadow edge V1 is bright-to-dark
    c_v2 = [s for s in near_v if 715 < u_at(s, 960) < 785 and polarity(s) > 15]
    if not c_v2:
        return {"ok": False, "why": "V2 not found"}
    V2s = max(c_v2, key=lambda s: s[4])
    # V3 = saturation step to the background (the longest LSD line there was printed artwork)
    sat = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32), (5, 5), 0)
    bv, bu = [], []
    for v in range(V0 + 10, V1 - 20, 3):
        row = sat[v, 800:872]; gr = row[6:] - row[:-6]; k = int(np.argmin(gr))
        if gr[k] < -20: bv.append(v); bu.append(800 + k + 3)
    if len(bv) < 6:
        return {"ok": False, "why": "V3 not found"}
    bv, bu = np.array(bv), np.array(bu)
    for _ in range(3):  # line fit u = a v + b, drop outliers
        a, b = np.polyfit(bv, bu, 1); r = np.abs(bu - (a * bv + b)); keep = r < max(2.5, 2 * np.median(r))
        bv, bu = bv[keep], bu[keep]
    V3s = (a * bv[0] + b, bv[0], a * bv[-1] + b, bv[-1], np.hypot(a * (bv[-1] - bv[0]), bv[-1] - bv[0]), np.degrees(np.arctan2(1, a)))
    V2s = extend_v2(cv2.GaussianBlur(G0, (3, 3), 0), V2s, V0 + 8, V1 - 2) or V2s   # full strip edge first
    # H5 = strip top: walk UP from inside the strip until the rows above turn persistently bright
    hv, hu = [], []
    v_in = int(V2s[1]) + 5
    for u in range(int(u_at(V2s, v_in)) - STRIP_W, int(u_at(V2s, v_in)) - 2):
        inside = np.median(G0[v_in:v_in + 10, u])
        for v in range(v_in, V0 + 8, -1):
            if G0[v - 6:v, u].min() > inside + 40: hv.append(v); hu.append(u); break
    H5s = None
    if hu:
        vm = float(np.median(hv)); hu = [u for u, v in zip(hu, hv) if abs(v - vm) <= 2]
        H5s = (min(hu), vm, max(hu), vm, max(hu) - min(hu), 0.0) if hu else None
    if H5s is None:
        return {"ok": False, "why": "H5 not found"}
    V2s = (u_at(V2s, H5s[1]), H5s[1], V2s[2], V2s[3], float(np.hypot(V2s[2] - u_at(V2s, H5s[1]), V2s[3] - H5s[1])), V2s[5])  # top = corner
    c2, d2, _ = line3d(V2s); c3, d3, _ = line3d(V3s)
    vc = H5s[1]; uc = u_at(V2s, vc)
    lean = np.degrees(np.arctan2(np.hypot(*d2[:2]), d2[2]))  # lean follows V2 (user: the accurate one)
    # D: LSD lines rising to the right whose infinite line passes within 4 px of the corner
    Ld = cv2.createLineSegmentDetector().detect(gray[780:960, 720:900])[0]
    Dc = []
    for x0, y0, x1, y1 in (Ld.reshape(-1, 4) + [720, 780, 720, 780]) if Ld is not None else []:
        if x1 < x0: x0, y0, x1, y1 = x1, y1, x0, y0
        ln = np.hypot(x1 - x0, y1 - y0); rise = np.degrees(np.arctan2(y0 - y1, x1 - x0))
        dist = abs((x1 - x0) * (y0 - vc) - (x0 - uc) * (y1 - y0)) / max(ln, 1e-6)
        if ln >= 10 and 15 < rise < 75 and dist < 4 and x0 > uc - 5: Dc.append((ln, rise, (x0, y0, x1, y1)))
    if not Dc:
        return {"ok": False, "why": "D not found"}
    ln, rise, _ = max(Dc)
    ue, ve = max(((q[2][2], q[2][3]) for q in Dc if abs(q[1] - rise) < 10), key=lambda p: p[0])
    # Across D: top boards | dark crease | bright highlight (~4 px) | face -> outline = highlight -> face step
    G1 = cv2.GaussianBlur(G0, (3, 3), 0)
    dl = np.array([ue - uc, ve - vc]); Ld_ = np.linalg.norm(dl); dl /= Ld_
    nf = np.array([dl[1], -dl[0]]) * (-1 if dl[1] * 1 > 0 else 1); nf = nf if nf[1] > 0 else -nf  # toward lower-right
    offs = np.arange(-8.0, 8.01, 0.25); Q4 = 4; st_sd, st_off = [], []
    for sd in np.arange(6, Ld_ + 30, 2):
        p0 = np.array([uc, vc]) + sd * dl
        prof = np.array([cv2.getRectSubPix(G1, (1, 1), tuple(p0 + o * nf))[0, 0] for o in offs])
        i0, i1 = np.searchsorted(offs, -7), np.searchsorted(offs, 0.01)
        cr = i0 + int(np.argmin(prof[i0:i1])); pk = cr + int(np.argmax(prof[cr:cr + 5 * Q4]))
        if pk >= len(prof) - 5 * Q4: continue
        gr = prof[pk + Q4:pk + 5 * Q4] - prof[pk:pk + 4 * Q4]; k = int(np.argmin(gr))
        if prof[pk] - prof[cr] > 40 and gr[k] < -20: st_sd.append(sd); st_off.append(offs[pk + k] + 0.5)
    if len(st_sd) < 5:
        return {"ok": False, "why": "D outline not found"}
    st_sd, st_off = np.array(st_sd), np.array(st_off); keep = np.ones(len(st_sd), bool)
    for _ in range(4):  # offset(sd) = c0 + c1 sd, drop > 1 px
        c1, c0 = np.polyfit(st_sd[keep], st_off[keep], 1); res = np.abs(st_off - (c0 + c1 * st_sd)); keep = res < 1.0
    sd_in = st_sd[keep]; runs = np.split(sd_in, np.where(np.diff(sd_in) > 8)[0] + 1)
    sd_end = max(runs, key=len)[-1]  # end of the longest run (the near-corner stretch has no crease)
    P_at = lambda sd: np.array([uc, vc]) + sd * dl + (c0 + c1 * sd) * nf  # noqa: E731
    q0, q1 = P_at(0.0), P_at(sd_end); dE = (q1 - q0) / np.linalg.norm(q1 - q0)
    A_ = np.array([[dE[0], -(V2s[2] - V2s[0])], [dE[1], -(V2s[3] - V2s[1])]]); stv = np.linalg.solve(A_, np.array([V2s[0], V2s[1]]) - q0)
    uc, vc = q0 + stv[0] * dE; ue, ve = q1
    corner = xyz(uc, vc + 3)
    if corner is None:
        return {"ok": False, "why": "no depth at the corner"}
    P = np.array([p for k in np.linspace(0.1, 0.8, 14) if (p := xyz(uc + k * (ue - uc), vc + k * (ve - vc))) is not None])
    cP = P.mean(0); dD = np.linalg.svd(P - cP)[2][0]; dD *= np.sign(dD[0]) or 1
    # K = D x V3 extended (image), 3D = midpoint of the closest points of the D / V3 3D lines
    a0, a1 = np.array([uc, vc]), np.array([ue, ve]); b0, b1 = np.array(V3s[:2]), np.array(V3s[2:4])
    ta, tb = np.linalg.solve(np.array([a1 - a0, -(b1 - b0)]).T, b0 - a0); K = a0 + ta * (a1 - a0)
    pD, p3 = closest(corner, dD, c3, d3)
    K3 = (pD + p3) / 2
    f = lambda a: [float(x) for x in a]  # noqa: E731
    un = lambda a: [float(x) - shift[i % 2] for i, x in enumerate(a)]  # noqa: E731  back to unshifted px
    # --- box centre: B1 = floor line right of V3's foot, Vb = V3 x B1 on the floor plane,
    # centre = middle of the C..Vb diagonal of the side face + half the stack thickness (H5) inwards
    ctr, B1px, Vbpx = None, None, None
    b1 = floor_line(G0, V3s)
    if b1 is not None:
        (a1, b1_, u_lo, u_hi), zf_px = b1
        zf = []
        for u, v in zf_px:  # median, not xyz()'s nearest: the white arm right below is nearer than the floor
            z = d[int(v) - 1:int(v) + 2, int(u) - 1:int(u) + 2].ravel(); z = z[(z > 0.3) & (z < 3)]
            if z.size: zf.append(float((t + np.median(z) * ray(u, v))[2]))
        if zf:
            zfloor = float(np.median(zf)); p0_, d3i = np.array(V3s[:2]), np.array(V3s[2:4]) - np.array(V3s[:2])
            sv = (a1 * p0_[0] + b1_ - p0_[1]) / (d3i[1] - a1 * d3i[0]); Vpx = p0_ + sv * d3i
            rv = ray(*Vpx); Vb = t + (zfloor - t[2]) / rv[2] * rv
            thick = (H5s[2] - H5s[0]) * (R.T @ (corner - t))[2] / FX
            ctr = box_center(corner, K3, Vb, thick, t)
            cc = R.T @ (ctr["center"] - t)
            ctr["px"] = un([FX * cc[0] / cc[2] + CX + shift[0], FY * cc[1] / cc[2] + CY + shift[1]])
            ctr["floor_z"] = zfloor
            B1px, Vbpx = un([u_lo, a1 * u_lo + b1_, u_hi, a1 * u_hi + b1_]), un(Vpx)
    out = {
        "ok": True,
        "px": {"V2": un(V2s[:4]), "V3": un(V3s[:4]), "H5": un(H5s[:4]), "D": un([uc, vc, ue, ve]), "K": un(K),
               "B1": B1px, "Vb": Vbpx, "center": ctr["px"] if ctr else None},
        "corner": f(corner), "K": f((pD + p3) / 2), "K_gap_m": float(np.linalg.norm(pD - p3)),
        "D_dir": f(dD), "D_yaw_deg": float(np.degrees(np.arctan2(dD[1], dD[0]))),
        "D_pitch_deg": float(np.degrees(np.arcsin(dD[2]))), "D_len_m": float(np.linalg.norm(pD - corner)),
        "lean_deg": float(lean), "lean_src": "V2", "V2_len_px": float(V2s[4]), "top_z": float(corner[2]),
    }
    if ctr:
        out.update(center=f(ctr["center"]), face_center=f(ctr["face_center"]), Vb=f(ctr["Vb"]), floor_z=ctr["floor_z"],
                   thickness_m=ctr["thickness_m"], normal=f(ctr["normal"]))
    else:
        out.update(center=None, center_why="floor line B1 not found (left arm / held box in front?)")
    return out


def floor_line(G0, V3s, min_step=25):
    """B1 = floor contact line right of V3's foot: per column the face -> floor-strip step going
    down (the floor strip is bright but not saturated; the white arm is). ((a, b, u0, u1), floor px)."""
    G = cv2.GaussianBlur(G0, (3, 3), 0)
    u_foot, v_foot = V3s[2], V3s[3]
    pts = []
    for u in range(int(u_foot) + 4, int(u_foot) + 90, 2):
        lo, hi = int(v_foot) - 10, int(v_foot) + 70
        col = G[lo:hi, u]; gr = col[3:] - col[:-3]; k = int(np.argmax(gr))
        if gr[k] > min_step and col[min(k + 4, len(col) - 1)] < 240: pts.append((u, lo + k + 1.5))
    if len(pts) < 10:
        return None
    P = np.array(pts); keep = np.ones(len(P), bool)
    for _ in range(3):
        a, b = np.polyfit(P[keep, 0], P[keep, 1], 1); keep = np.abs(P[:, 1] - (a * P[:, 0] + b)) < 1.5
    if keep.sum() < 10:
        return None
    U = P[keep, 0]
    return (a, b, float(U.min()), float(U.max())), [(u, a * u + b + 4) for u in U[::3]]   # floor = 4 px below


def box_center(C, K, Vb, thickness, cam):
    """C top-front, K top-back, Vb bottom-back corner of the visible side face (a parallelogram):
    face centre = midpoint of the C..Vb diagonal (needs neither the hidden 4th corner nor K's
    height), centre = face centre + thickness/2 along the face normal, away from the camera."""
    C, K, Vb = (np.asarray(x, float) for x in (C, K, Vb))
    fc = (C + Vb) / 2
    n = np.cross(K - C, Vb - K); n /= np.linalg.norm(n)
    if n @ (fc - np.asarray(cam, float)) < 0: n = -n
    return {"center": fc + n * thickness / 2, "face_center": fc, "Vb": Vb, "normal": n, "thickness_m": float(thickness)}


def extend_v2(G, V2s, v_min, v_max, step=3, min_step=25):
    """V2 at its real length: LSD often returns only a piece of the strip's right edge. From the
    piece, walk UP and DOWN along it taking the dark-strip -> bright-face step on each row; each
    walk ends where the edge stops (strip top above, arm / held box below). Refit, or None."""
    a = (V2s[2] - V2s[0]) / max(1e-6, V2s[3] - V2s[1])
    found = {}
    for start, end, dv in ((int(V2s[1]), int(v_min), -step), (int(V2s[1]) + step, int(v_max), step)):
        miss = 0
        for v in range(start, end, dv):
            u0 = V2s[0] + a * (v - V2s[1]); cols = np.arange(int(u0) - 6, int(u0) + 7)
            gr = G[v, cols + 2] - G[v, cols - 2]; k = int(np.argmax(gr))
            if gr[k] > min_step:
                found[v] = cols[k]; miss = 0
            else:
                miss += 1
                if miss >= 3: break
    if len(found) < 8:
        return None
    vs = np.array(sorted(found), float); us = np.array([found[v] for v in sorted(found)], float); keep = np.ones(len(vs), bool)
    for _ in range(3):  # u = a v + b, drop > 1.5 px residuals
        a, b = np.polyfit(vs[keep], us[keep], 1); keep = np.abs(us - (a * vs + b)) < 1.5
    v0, v1 = float(vs[keep][0]), float(vs[keep][-1])
    return (a * v0 + b, v0, a * v1 + b, v1, float(np.hypot(a * (v1 - v0), v1 - v0)), float(np.degrees(np.arctan2(1, a))))


def detect_tracked(im, d, R, t, anchor):
    """Translate the frame so the base-frame anchor (last corner) lands on REF_CORNER_PX, then
    detect(): the target is found wherever the head / torso moved the camera.
    ponytail: translation only - a much nearer / farther target also needs STRIP_W / ROI scaled."""
    c = R.T @ (np.asarray(anchor, float) - t)
    if c[2] <= 0.1:
        return {"ok": False, "why": "anchor behind the camera"}
    u, v = FX * c[0] / c[2] + CX, FY * c[1] / c[2] + CY
    du, dv = int(round(REF_CORNER_PX[0] - u)), int(round(REF_CORNER_PX[1] - v))
    if abs(du) > 400 or abs(dv) > 400:
        return {"ok": False, "why": f"target off the tuned view (shift {du},{dv} px)"}
    M = np.float32([[1, 0, du], [0, 1, dv]]); hh, ww = d.shape[:2]
    r = detect(cv2.warpAffine(im, M, (ww, hh)), cv2.warpAffine(d, M, (ww, hh), flags=cv2.INTER_NEAREST), R, t, (du, dv))
    r["shift"] = [du, dv]
    return r


if __name__ == "__main__":  # self-checks: box centre on a known box, V2 extension on a synthetic strip
    cam = np.array([0.0, 0.0, 1.5])                       # side face at y = 0.4 facing the camera, box 2 cm thick
    bc = box_center([1.1, 0.4, 1.05], [1.45, 0.4, 1.05], [1.45, 0.4, 0.69], 0.02, cam)
    assert np.allclose(bc["center"], [1.275, 0.41, 0.87]), bc["center"]
    G = np.full((1200, 1920), 200.0)
    for v in range(900, 1100):
        G[v, :int(750 + 0.2 * (v - 900))] = 100.0          # dark strip left of the edge, 0.2 px/row lean
    G[1040:, :] = 230.0                                      # "arm" hides the strip below v 1040
    piece = (750 + 0.2 * 80, 980, 750 + 0.2 * 100, 1000, 20.6, 78.7)   # LSD returned only v 980-1000
    G[:900, :] = 200.0                                       # nothing above the strip top
    v2 = extend_v2(G, piece, 838, 1043)
    assert v2 and abs(v2[1] - 900) <= 3 and 1030 <= v2[3] <= 1040, v2           # up to the strip top, down to the arm
    assert abs((v2[2] - v2[0]) / (v2[3] - v2[1]) - 0.2) < 0.02, v2               # same lean
    assert extend_v2(np.full((1200, 1920), 200.0), piece, 838, 1043) is None     # no edge -> keep LSD
    print("self-check ok: V2 extended to v %.0f-%.0f (%.0f px), slope %.3f" % (v2[1], v2[3], v2[4], (v2[2] - v2[0]) / (v2[3] - v2[1])))
