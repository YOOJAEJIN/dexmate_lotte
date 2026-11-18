"""Place target edges on the HD1200 snapshot: V1/V2 = left/right edge of the front strip (leaning),
H5 = strip top between them, V3 = target / white carton boundary. 3D + lean from metric depth."""
import json, sys, cv2, numpy as np
im = cv2.imread('hires.png'); d = np.load('depth_hi.npy').astype(np.float32) / 1000
FX, FY, CX, CY = 735.25, 735.141, 992.571, 618.88
bx = [b for b in json.load(open('/home/dexmate/dexmate_lotte/vision/logs/box_coords.json'))['boxes'] if not b.get('held')]
A = np.array([q['cam'] for q in bx[0]['corners']]); B = np.array([q['xyz'] for q in bx[0]['corners']]); ma, mb = A.mean(0), B.mean(0)
U, _, Vt = np.linalg.svd((A - ma).T @ (B - mb)); R = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T; t = mb - R @ ma
ray = lambda u, v: R @ np.array([(u - CX) / FX, (v - CY) / FY, 1.0])
def xyz(u, v, k=2):
    z = d[int(v)-k:int(v)+k+1, int(u)-k:int(u)+k+1].ravel(); z = z[(z > 0.3) & (z < 3)]
    return t + np.min(z) * ray(u, v) if z.size else None   # min: the near (target) side of an edge
U0, U1, V0, V1 = 700, 870, 830, 1045            # ponytail: fixed ROI on the confirmed target; auto when the spot moves
g = cv2.cvtColor(im[V0:V1, U0:U1], cv2.COLOR_BGR2GRAY)
L = cv2.createLineSegmentDetector().detect(g)[0].reshape(-1, 4) + [U0, V0, U0, V0]
segs = []
for x0, y0, x1, y1 in L:
    if y1 < y0: x0, y0, x1, y1 = x1, y1, x0, y0
    ln = np.hypot(x1 - x0, y1 - y0); a = np.degrees(np.arctan2(y1 - y0, x1 - x0))   # 90 = vertical
    if ln >= 12: segs.append((x0, y0, x1, y1, ln, a))
def u_at(s, v): return s[0] + (s[2] - s[0]) * (v - s[1]) / max(1e-6, s[3] - s[1])
near_v = [s for s in segs if abs(s[5] - 90) < 25]
G0 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
def polarity(s, k=4):                    # mean(right side) - mean(left side) along the segment
    vs = np.linspace(s[1], s[3], 10); us = [u_at(s, v) for v in vs]
    return np.mean([G0[int(v), int(u) + k] - G0[int(v), int(u) - k] for u, v in zip(us, vs)])
# V2 = gray strip (left) -> brighter target face (right): dark-to-bright. The shadow edge V1 is
# bright-to-dark (pile -> strip), so polarity alone tells them apart.
V2s = max([s for s in near_v if 715 < u_at(s, 960) < 785 and polarity(s) > 15], key=lambda s: s[4])
# V3 = tan -> white boundary (colour edge; the longest LSD line there was the printed bottle)
sat = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32), (5, 5), 0)
bv, bu = [], []
for v in range(V0 + 10, V1 - 20, 3):
    row = sat[v, 800:872]; gr = row[6:] - row[:-6]                # saturation drop over 6 px
    k = int(np.argmin(gr))
    if gr[k] < -20: bv.append(v); bu.append(800 + k + 3)
bv, bu = np.array(bv), np.array(bu)
for _ in range(3):                                                 # line fit u = a v + b, drop outliers
    a, b = np.polyfit(bv, bu, 1); r = np.abs(bu - (a * bv + b)); keep = r < max(2.5, 2 * np.median(r))
    bv, bu = bv[keep], bu[keep]
V3s = (a * bv[0] + b, bv[0], a * bv[-1] + b, bv[-1], np.hypot(a * (bv[-1] - bv[0]), bv[-1] - bv[0]), np.degrees(np.arctan2(1, a)))
# H5 = strip top between V1 and V2: walk UP from inside the strip (top of V1/V2) until the
# rows above turn persistently bright -> that row is the top edge. Walking up from inside never
# reaches the thin groove line above the light band, and skips edges that are not the strip's.
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
hv, hu = [], []
STRIP_W = 16                                                # ponytail: strip width in HD px at ~1.1 m; derive from depth if the distance changes
v_in = int(V2s[1]) + 5                                      # a row inside the strip, below V2's top
for u in range(int(u_at(V2s, v_in)) - STRIP_W, int(u_at(V2s, v_in)) - 2):
    inside = np.median(gray[v_in:v_in + 10, u])
    for v in range(v_in, V0 + 8, -1):
        if gray[v - 6:v, u].min() > inside + 40: hv.append(v); hu.append(u); break
H5s = None
if hu:
    vm = float(np.median(hv)); hu = [u for u, v in zip(hu, hv) if abs(v - vm) <= 2]   # inlier columns only
    H5s = (min(hu), vm, max(hu), vm, max(hu) - min(hu), 0.0) if hu else None
def line3d(s, n=12):
    P = [p for k in np.linspace(0.1, 0.9, n) if (p := xyz(s[0] + k * (s[2] - s[0]), s[1] + k * (s[3] - s[1]))) is not None]
    P = np.array(P); c = P.mean(0); dvec = np.linalg.svd(P - c)[2][0]; dvec *= np.sign(dvec[2]) or 1
    return c, dvec, P
out = im[V0 - 40:V1 + 15, U0 - 15:U1 + 15].copy(); o = lambda u, v: (int(u - U0 + 15), int(v - V0 + 40))
for name, s, col in (("V2", V2s, (0, 200, 255)), ("V3", V3s, (255, 0, 255)), ("H5", H5s, (255, 200, 0))):
    if s is None: print(name, "not found"); continue
    c, dv, P = line3d(s, 6 if name == "H5" else 12)
    lean = np.degrees(np.arccos(abs(dv[2])))
    print(f"{name}: px ({s[0]:.0f},{s[1]:.0f})-({s[2]:.0f},{s[3]:.0f}) {s[4]:.0f}px img {s[5]:.1f}deg | base mid {c.round(3)}"
          + (f" | 3D tilt from vertical {lean:.1f}deg (toward {'+y' if dv[1] > 0 else '-y'} {abs(dv[1]):.2f}, {'+x' if dv[0] > 0 else '-x'} {abs(dv[0]):.2f})" if name != "H5" else f" | top z {np.median(P[:, 2]):.3f}"))
    cv2.line(out, o(s[0], s[1]), o(s[2], s[3]), col, 2); cv2.putText(out, name, o(s[2] + 3, s[3] - 4 if name != "H5" else s[3] - 6), 0, 0.5, (0, 0, 0), 3); cv2.putText(out, name, o(s[2] + 3, s[3] - 4 if name != "H5" else s[3] - 6), 0, 0.5, col, 1)
if H5s is None: sys.exit('H5 not found this frame - skip it')
# --- place reference: front-top corner (H5 x V2), back edge V3 at the corner height, face yaw, lean
c2, d2, _ = line3d(V2s); c3, d3, _ = line3d(V3s)
vc = H5s[1]; uc = u_at(V2s, vc); corner = xyz(uc, vc + 3)                 # 3 px below the top edge = on the strip
at_z = lambda c, dv, z: c + dv * (z - c[2]) / dv[2]
p2, p3 = at_z(c2, d2, corner[2]), at_z(c3, d3, corner[2])
yaw = np.degrees(np.arctan2(p3[1] - p2[1], p3[0] - p2[0]))
lean = np.degrees(np.arctan2(np.hypot(*((d2[:2] + d3[:2]) / 2)), (d2[2] + d3[2]) / 2))
print(f"REF corner(H5xV2) {corner.round(3)} | back edge @same z {p3.round(3)} | length {100*np.linalg.norm((p3-p2)[:2]):.1f} cm"
      f" | face yaw {yaw:+.1f} deg | lean {lean:.1f} deg")
# --- D = third edge at the corner (H5 = across, V2 = vertical, D = going back along the top of the
# V2..V3 face). LSD lines rising to the right whose infinite line passes within 4 px of the corner.
Ld = cv2.createLineSegmentDetector().detect(cv2.cvtColor(im[780:960, 720:900], cv2.COLOR_BGR2GRAY))[0].reshape(-1, 4) + [720, 780, 720, 780]
Dc = []
for x0, y0, x1, y1 in Ld:
    if x1 < x0: x0, y0, x1, y1 = x1, y1, x0, y0
    ln = np.hypot(x1 - x0, y1 - y0); rise = np.degrees(np.arctan2(y0 - y1, x1 - x0))
    dist = abs((x1 - x0) * (y0 - vc) - (x0 - uc) * (y1 - y0)) / max(ln, 1e-6)
    if ln >= 10 and 15 < rise < 75 and dist < 4 and x0 > uc - 5: Dc.append((ln, rise, (x0, y0, x1, y1)))
if Dc:
    ln, rise, (x0, y0, x1, y1) = max(Dc)
    ue, ve = max(((q[2][2], q[2][3]) for q in Dc if abs(q[1] - rise) < 10), key=lambda p: p[0])   # farthest end, same direction
    # Refine onto the real outline. Across D: top boards | dark crease (shadow) | bright highlight
    # (lit board top, ~4 px) | face. LSD lands on the highlight; the outline is the highlight -> face
    # step. Measured as an offset (px, + = toward the face) at each station along the LSD line.
    G1 = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0)
    dl = np.array([ue - uc, ve - vc]); Ld_ = np.linalg.norm(dl); dl /= Ld_
    nf = np.array([dl[1], -dl[0]]) * (-1 if dl[1] * 1 > 0 else 1); nf = nf if nf[1] > 0 else -nf   # toward lower-right
    offs = np.arange(-8.0, 8.01, 0.25); Q4 = 4; st_sd, st_off = [], []
    for sd in np.arange(6, Ld_ + 30, 2):
        p0 = np.array([uc, vc]) + sd * dl
        prof = np.array([cv2.getRectSubPix(G1, (1, 1), tuple(p0 + o * nf))[0, 0] for o in offs])
        i0, i1 = np.searchsorted(offs, -7), np.searchsorted(offs, 0.01)
        cr = i0 + int(np.argmin(prof[i0:i1])); pk = cr + int(np.argmax(prof[cr:cr + 5 * Q4]))
        if pk >= len(prof) - 5 * Q4: continue
        gr = prof[pk + Q4:pk + 5 * Q4] - prof[pk:pk + 4 * Q4]; k = int(np.argmin(gr))
        if prof[pk] - prof[cr] > 40 and gr[k] < -20: st_sd.append(sd); st_off.append(offs[pk + k] + 0.5)
    st_sd, st_off = np.array(st_sd), np.array(st_off); keep = np.ones(len(st_sd), bool)
    for _ in range(4):                                                   # offset(sd) = c0 + c1 sd, drop > 1 px
        c1, c0 = np.polyfit(st_sd[keep], st_off[keep], 1); res = np.abs(st_off - (c0 + c1 * st_sd)); keep = res < 1.0
    sd_in = st_sd[keep]; runs = np.split(sd_in, np.where(np.diff(sd_in) > 8)[0] + 1)   # contiguous station runs
    sd_end = max(runs, key=len)[-1]                                      # end = end of the longest run (near-corner stretch has no crease)
    P_at = lambda sd: np.array([uc, vc]) + sd * dl + (c0 + c1 * sd) * nf
    q0, q1 = P_at(0.0), P_at(sd_end); dE = (q1 - q0) / np.linalg.norm(q1 - q0)
    A_ = np.array([[dE[0], -(V2s[2] - V2s[0])], [dE[1], -(V2s[3] - V2s[1])]]); stv = np.linalg.solve(A_, np.array([V2s[0], V2s[1]]) - q0)
    uc2, vc2 = q0 + stv[0] * dE
    print(f"D refine: {keep.sum()}/{len(st_sd)} stations, outline offset {c0 + c1 * 15:+.2f}px near / {c0 + c1 * 70:+.2f}px far (+ = toward face),"
          f" ends {sd_end:.0f}px from corner, meets V2 at ({uc2:.1f},{vc2:.1f}) vs H5xV2 ({uc:.1f},{vc:.1f})")
    cv2.line(out, o(uc, vc), o(ue, ve), (160, 160, 160), 1)            # old (LSD) D, grey
    uc, vc, (ue, ve) = uc2, vc2, q1
    corner = xyz(uc, vc + 3)
    P = np.array([p for k in np.linspace(0.1, 0.8, 14) if (p := xyz(uc + k * (ue - uc), vc + k * (ve - vc))) is not None])   # ends skipped: depth at the tip is unreliable
    cP = P.mean(0); dD = np.linalg.svd(P - cP)[2][0]; dD *= np.sign(dD[0]) or 1
    r = ray(ue, ve); w0 = corner - t; b_ = dD @ r                      # far end = point on the 3D line closest to the end pixel's ray
    far = corner + dD * ((b_ * (r @ w0) - (r @ r) * (dD @ w0)) / ((dD @ dD) * (r @ r) - b_ * b_))
    print(f"D: px ({uc:.0f},{vc:.0f})-({ue:.0f},{ve:.0f}) img rise {rise:.1f}deg | 3D dir {dD.round(3)} yaw {np.degrees(np.arctan2(dD[1], dD[0])):+.1f}deg"
          f" pitch {np.degrees(np.arcsin(dD[2])):+.1f}deg | corner -> far end {100*np.linalg.norm(far - corner):.1f} cm, far {far.round(3)}")
    cv2.line(out, o(uc, vc), o(ue, ve), (0, 0, 255), 2); cv2.putText(out, "D", o(ue + 3, ve), 0, 0.5, (0, 0, 0), 3); cv2.putText(out, "D", o(ue + 3, ve), 0, 0.5, (0, 0, 255), 1)
    # --- extend D and V3 until they meet: the back-top corner (K)
    a0, a1 = np.array([uc, vc]), np.array([ue, ve]); b0, b1 = np.array(V3s[:2]), np.array(V3s[2:4])
    M = np.array([a1 - a0, -(b1 - b0)]).T; ta, tb = np.linalg.solve(M, b0 - a0); K = a0 + ta * (a1 - a0)
    def closest(p1, u1, p2, u2):                                       # closest points of two 3D lines
        w = p1 - p2; a_, b__, c_ = u1 @ u1, u1 @ u2, u2 @ u2; d_, e_ = u1 @ w, u2 @ w; den = a_ * c_ - b__ * b__
        s1, s2 = (b__ * e_ - c_ * d_) / den, (a_ * e_ - b__ * d_) / den; return p1 + s1 * u1, p2 + s2 * u2
    pD, p3 = closest(corner, dD, c3, d3)
    K3 = (pD + p3) / 2
    rK = ray(*K); kD, _ = closest(t, rK, corner, dD)                    # ray through K vs the D line
    print(f"K (D x V3): px ({K[0]:.1f},{K[1]:.1f}), D extended x{ta:.2f}, V3 extended x{tb:.2f} | 3D {K3.round(3)}"
          f" (D/V3 3D lines miss each other by {100*np.linalg.norm(pD - p3):.1f} cm) | D length to K {100*np.linalg.norm(pD - corner):.1f} cm")
    cv2.line(out, o(*a0), o(*K), (0, 0, 255), 2); cv2.line(out, o(*K), o(*b1), (255, 0, 255), 2)
    cv2.circle(out, o(*K), 4, (0, 255, 255), -1); cv2.putText(out, "K", o(K[0] + 5, K[1] - 4), 0, 0.5, (0, 0, 0), 3); cv2.putText(out, "K", o(K[0] + 5, K[1] - 4), 0, 0.5, (0, 255, 255), 1)
    cv2.imwrite('target_edges.png', cv2.resize(out, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC))
else:
    print("D not found")
for name, (u, v) in {"held box": (1040, 650)}.items(): print(name, "base", np.round(xyz(u, v, 4), 3))
cv2.imwrite('target_edges.png', cv2.resize(out, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC))
