"""Observed mesh -> room layout (floor, ceiling, Manhattan walls) in Mode A's Layout schema.

    up axis     camera "up" vectors give a first guess; refined to the dominant horizontal-
                surface normal (floor/ceiling) of the area-weighted mesh normals
    Manhattan   the wall direction is the peak of a 90-degree-periodic histogram of
                horizontal normal angles
    floor/ceil  lowest up-facing and highest down-facing height peaks; a ceiling the video
                never saw is NOT invented silently: it is set from a prior and flagged
    walls       candidate wall planes are peaks of the per-axis histograms of wall-facing
                surface area; they cut the floor into a grid of cells and a cell is inside the
                room when free-space/floor evidence covers most of it (cell-complex selection)
The footprint is written as a Layout at 1 px = 1 cm, so Mode A's extruder and viewer work on
video-derived layouts unchanged ("both modes in one pipeline").
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import trimesh
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, label
from shapely.geometry import Polygon
from shapely.ops import unary_union

from core.layout import Layout, Room, Wall
from core.log import RunContext

PX_PER_M = 100           # Layout pixel = 1 cm
CEILING_PRIOR_M = 2.6    # used only when the ceiling was never observed (flagged)
WALL_T_M = 0.12
WALL_MIN_AREA = 0.8      # m^2 of wall-facing surface for a plane to cut the floor into cells
JOG_M = 0.15             # footprint jogs narrower than 2x this are regularised away


@dataclass
class RoomFrame:
    Rm: np.ndarray            # world -> room frame rotation (room frame: Y up, X/Z Manhattan)
    floor_y: float
    ceil_y: float
    ceil_observed: bool
    polygon: np.ndarray       # (N,2) room footprint in room-frame (X, Z), metres, CCW
    wall_planes: list[dict]   # {"axis": 0|2, "value": m, "normal": +-1, "p0": (x,z), "p1": (x,z)}
    origin_xz: np.ndarray     # room-frame (X,Z) of Layout pixel (0,0)
    stats: dict

    def to_room(self, P: np.ndarray) -> np.ndarray:
        return P @ self.Rm.T


def _face_data(m: trimesh.Trimesh):
    n = m.face_normals
    a = m.area_faces
    c = m.triangles_center
    return n, a, c


def _peak_dir(vecs: np.ndarray, w: np.ndarray, init: np.ndarray, cos_win: float = np.cos(np.radians(20))):
    d = init / np.linalg.norm(init)
    for _ in range(10):
        s = vecs @ d
        sel = np.abs(s) > cos_win
        if not sel.any():
            break
        v = vecs[sel] * np.sign(s[sel])[:, None]
        d_new = (v * w[sel, None]).sum(0)
        d_new /= np.linalg.norm(d_new)
        if np.allclose(d_new, d, atol=1e-6):
            break
        d = d_new
        cos_win = max(cos_win, np.cos(np.radians(8)))
    return d


def extract_planes(m: trimesh.Trimesh, n_max: int = 30, thr: float = 0.03, min_area: float = 0.3,
                   n_samples: int = 40000, iters: int = 400, seed: int = 0) -> list[dict]:
    """Sequential RANSAC on area-weighted face centres. A face is an inlier when it is within
    `thr` of the plane and its normal agrees (cos > 0.8). Planes are refit by SVD on inliers."""
    rng = np.random.default_rng(seed)
    n, a, c = _face_data(m)
    if len(a) == 0:
        return []
    idx = rng.choice(len(a), min(n_samples, len(a)), replace=False, p=a / a.sum())
    P, N = c[idx], n[idx]
    w = np.full(len(idx), a.sum() / len(idx))  # each sample stands for this much area
    alive = np.ones(len(idx), bool)
    planes = []
    for _ in range(n_max):
        ids = np.flatnonzero(alive)
        if len(ids) < 50:
            break
        best, best_in = None, None
        for _ in range(iters):
            k = ids[rng.integers(len(ids))]
            nn, p0 = N[k], P[k]
            inl = ids[(np.abs((P[ids] - p0) @ nn) < thr) & (np.abs(N[ids] @ nn) > 0.8)]
            if best is None or len(inl) > len(best_in):
                best, best_in = (nn, p0), inl
        if best_in is None or w[best_in].sum() < min_area:
            break
        Q = P[best_in]
        mu = Q.mean(0)
        nn = np.linalg.svd(Q - mu)[2][-1]
        if nn @ N[best_in].mean(0) < 0:
            nn = -nn  # orient like the faces (towards the observer)
        inl = ids[(np.abs((P[ids] - mu) @ nn) < thr) & (N[ids] @ nn > 0.8)]
        if w[inl].sum() < min_area:
            alive[best_in] = False
            continue
        planes.append({"n": nn, "p": mu, "area": float(w[inl].sum()), "pts": P[inl]})
        alive[inl] = False
    return planes


def manhattan_frame(m: trimesh.Trimesh, cam_R: list[np.ndarray], cam_C: np.ndarray | None = None,
                    planes: list[dict] | None = None):
    """Up = normal of the dominant floor plane (and ceiling, if seen); X = dominant wall normal."""
    planes = planes if planes is not None else extract_planes(m)
    up0 = -np.mean([R[1] for R in cam_R], 0)
    up0 /= np.linalg.norm(up0)
    cam_h = None if cam_C is None else np.median(cam_C @ up0)
    horiz = [pl for pl in planes if abs(pl["n"] @ up0) > np.cos(np.radians(30))]
    floors = [pl for pl in horiz if pl["n"] @ up0 > 0 and (cam_h is None or pl["p"] @ up0 < cam_h - 0.5)]
    ceils = [pl for pl in horiz if pl["n"] @ up0 < 0 and (cam_h is None or pl["p"] @ up0 > cam_h + 0.2)]
    if floors:
        up = max(floors, key=lambda pl: pl["area"])["n"].copy()
        if ceils:
            cl = max(ceils, key=lambda pl: pl["area"])
            up = up * max(floors, key=lambda pl: pl["area"])["area"] - cl["n"] * cl["area"]
        up /= np.linalg.norm(up)
    else:
        n, a, _ = _face_data(m)
        up = _peak_dir(n, a, up0)
        if up @ up0 < 0:
            up = -up
    walls = [pl for pl in planes if abs(pl["n"] @ up) < np.sin(np.radians(15))]
    e1 = np.cross(up, [1, 0, 0] if abs(up[0]) < 0.9 else [0, 0, 1])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    if walls:
        th = np.array([np.arctan2(pl["n"] @ e2, pl["n"] @ e1) for pl in walls])
        wa = np.array([pl["area"] for pl in walls])
        z = (wa * np.exp(4j * th)).sum()  # 90-degree periodic circular mean
        phi = np.angle(z) / 4
    else:
        n, a, _ = _face_data(m)
        h = n - np.outer(n @ up, up)
        sel = (np.abs(n @ up) < 0.3) & (np.linalg.norm(h, axis=1) > 0.5)
        th = np.arctan2(h[sel] @ e2, h[sel] @ e1)
        phi = np.angle((a[sel] * np.exp(4j * th)).sum()) / 4
    x = np.cos(phi) * e1 + np.sin(phi) * e2
    z = np.cross(x, up)
    return np.stack([x, up, z])


def _height_peaks(y: np.ndarray, w: np.ndarray, bin_m: float = 0.02):
    if len(y) == 0:
        return np.zeros(0), np.zeros(0)
    lo, hi = y.min() - bin_m, y.max() + bin_m
    hist, edges = np.histogram(y, bins=max(3, int((hi - lo) / bin_m)), range=(lo, hi), weights=w)
    hist = np.convolve(hist, [0.25, 0.5, 0.25], "same")
    pk = [k for k in range(1, len(hist) - 1) if hist[k] >= hist[k - 1] and hist[k] >= hist[k + 1] and hist[k] > 0]
    centers = (edges[:-1] + edges[1:]) / 2
    return centers[pk], hist[pk]


def _axis_peaks(v: np.ndarray, w: np.ndarray, bin_m: float = 0.02, min_area: float = 0.25):
    pos, val = _height_peaks(v, w, bin_m)
    keep = val >= min_area  # m^2 of wall-facing surface in the (smoothed) peak bin
    pos, val = pos[keep], val[keep]
    # merge peaks closer than 8 cm (keep the stronger)
    order = np.argsort(-val)
    out = []
    for k in order:
        if all(abs(pos[k] - p) > 0.08 for p, _ in out):
            out.append((pos[k], val[k]))
    return sorted(out)


def estimate(m: trimesh.Trimesh, cam_R: dict, cam_C: dict, vol, ctx: RunContext) -> RoomFrame:
    planes = extract_planes(m)
    Cw = np.array(list(cam_C.values()))
    Rm = manhattan_frame(m, list(cam_R.values()), Cw, planes)
    n, a, c = _face_data(m)
    nr, cr = n @ Rm.T, c @ Rm.T
    C = Cw @ Rm.T
    up = Rm[1]
    hz = [pl for pl in planes if abs(pl["n"] @ up) > np.cos(np.radians(10))]
    # floor / ceiling
    upf = nr[:, 1] > 0.9
    dnf = nr[:, 1] < -0.9
    cam_y = np.median(C[:, 1])
    # floor: the lowest large up-facing plane below the cameras (>= 25% of the biggest one)
    fl = [(float(pl["p"] @ up), pl["area"]) for pl in hz if pl["n"] @ up > 0 and pl["p"] @ up < cam_y - 0.5]
    if fl:
        amax = max(ar for _, ar in fl)
        floor_y = min(y for y, ar in fl if ar >= 0.25 * amax)
    else:
        floor_y = float(np.percentile(cr[:, 1], 1))
        ctx.fallback("layout", "no floor plane found; using the lowest observed surface")
    ce = [(float(pl["p"] @ up), pl["area"]) for pl in hz
          if pl["n"] @ up < 0 and pl["p"] @ up > cam_y + 0.2 and pl["p"] @ up - floor_y > 1.8]
    ceil_obs = False
    if not ce:  # small ceiling patches: >= 0.3 m^2 of down-facing surface within +-5 cm
        sel = dnf & (cr[:, 1] > cam_y + 0.2) & (cr[:, 1] - floor_y > 1.8)
        if sel.any():
            ys, ar = cr[sel, 1], a[sel]
            hist, e = np.histogram(ys, bins=max(1, int(np.ptp(ys) / 0.05) + 1), weights=ar)
            k = int(np.argmax(hist))
            if hist[k] >= 0.3:
                inb = (ys >= e[k]) & (ys <= e[k + 1])
                ce = [(float(np.average(ys[inb], weights=ar[inb])), float(hist[k]))]
    if ce:  # a ceiling of at least 0.3 m^2 was observed
        ceil_y = max(ce, key=lambda t: t[1])[0]
        ceil_obs = True
    else:
        top = np.percentile(cr[:, 1], 99.5)
        ceil_y = max(floor_y + CEILING_PRIOR_M, top + 0.02)
        ctx.fallback("layout", f"ceiling never observed; height set from a {CEILING_PRIOR_M} m prior "
                               f"(or the highest observed surface) and marked generated")
    H = ceil_y - floor_y

    # wall candidates: wall-facing area by axis, between 0.3 m and ceiling - 0.3 m
    band = (cr[:, 1] > floor_y + 0.3) & (cr[:, 1] < ceil_y - 0.3)
    wx = band & (np.abs(nr[:, 0]) > 0.9)
    wz = band & (np.abs(nr[:, 2]) > 0.9)
    xs = _axis_peaks(cr[wx, 0], a[wx])
    zs = _axis_peaks(cr[wz, 2], a[wz])
    for pl in planes:  # RANSAC wall planes are candidates too
        nr_ = pl["n"] @ Rm.T
        if abs(nr_[0]) > 0.95:
            xs.append((float(pl["p"] @ Rm[0]), pl["area"]))
        elif abs(nr_[2]) > 0.95:
            zs.append((float(pl["p"] @ Rm[2]), pl["area"]))

    # inside evidence on a 5 cm grid: observed floor, observed free space, camera positions
    res = 0.05
    P = m.vertices @ Rm.T
    lo = np.minimum(P[:, [0, 2]].min(0), C[:, [0, 2]].min(0)) - 0.3
    hi = np.maximum(P[:, [0, 2]].max(0), C[:, [0, 2]].max(0)) + 0.3
    W, Hh = np.ceil((hi - lo) / res).astype(int) + 1
    ev = np.zeros((Hh, W), bool)

    def mark(xz):
        ij = np.floor((xz - lo) / res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < Hh)
        ev[ij[ok, 1], ij[ok, 0]] = True

    fl = upf & (np.abs(cr[:, 1] - floor_y) < 0.05)
    mark(cr[fl][:, [0, 2]])
    mark(C[:, [0, 2]])
    free_xz = _free_space_xz(vol, Rm, floor_y, ceil_y)
    if len(free_xz):
        mark(free_xz)
    ev = binary_closing(ev, np.ones((5, 5)), iterations=2)
    ev = binary_fill_holes(ev)
    lab, nl = label(ev)
    if nl > 1:  # the component containing the cameras
        cij = np.floor((C[:, [0, 2]] - lo) / res).astype(int)
        ids = lab[np.clip(cij[:, 1], 0, Hh - 1), np.clip(cij[:, 0], 0, W - 1)]
        ids = ids[ids > 0]
        keep = np.bincount(ids).argmax() if len(ids) else np.bincount(lab.ravel())[1:].argmax() + 1
        ev = lab == keep

    # cell complex from wall lines + evidence extent
    ys_, xs_ = np.nonzero(ev)
    ex_lo = lo + np.array([xs_.min(), ys_.min()]) * res
    ex_hi = lo + np.array([xs_.max() + 1, ys_.max() + 1]) * res
    def lines(cands, lo_e, hi_e):
        strong = sorted(float(p) for p, ar in cands if ar >= WALL_MIN_AREA and lo_e - 0.3 < p < hi_e + 0.3)
        span = hi_e - lo_e
        # evidence extent only where no real wall bounds that side
        if not any(p < lo_e + 0.25 * span for p in strong):
            strong.append(lo_e)
        if not any(p > hi_e - 0.25 * span for p in strong):
            strong.append(hi_e)
        return _dedupe(sorted(strong))
    xl = lines(xs, ex_lo[0], ex_hi[0])
    zl = lines(zs, ex_lo[1], ex_hi[1])
    cells = []
    for i in range(len(xl) - 1):
        for j in range(len(zl) - 1):
            a0 = np.floor((np.array([xl[i], zl[j]]) - lo) / res).astype(int)
            a1 = np.ceil((np.array([xl[i + 1], zl[j + 1]]) - lo) / res).astype(int)
            sub = ev[max(a0[1], 0):max(a1[1], 0), max(a0[0], 0):max(a1[0], 0)]
            if sub.size and sub.mean() > 0.5:
                cells.append(Polygon([(xl[i], zl[j]), (xl[i + 1], zl[j]), (xl[i + 1], zl[j + 1]), (xl[i], zl[j + 1])]))
    poly = unary_union(cells) if cells else None
    if poly is None or poly.is_empty:
        ctx.fallback("layout", "no room footprint found; using the evidence bounding box")
        poly = Polygon([(ex_lo[0], ex_lo[1]), (ex_hi[0], ex_lo[1]), (ex_hi[0], ex_hi[1]), (ex_lo[0], ex_hi[1])])
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    poly = Polygon(poly.exterior)
    # remove jogs and spikes narrower than 2*JOG_M; mitred joins keep the corners square
    reg = poly.buffer(-JOG_M, join_style=2).buffer(2 * JOG_M, join_style=2).buffer(-JOG_M, join_style=2)
    if reg.geom_type == "Polygon" and reg.area > 0.5 * poly.area:
        poly = reg
    elif reg.geom_type == "MultiPolygon" and len(reg.geoms):
        big = max(reg.geoms, key=lambda g: g.area)
        poly = big if big.area > 0.5 * poly.area else poly
    poly = Polygon(poly.exterior).simplify(0.01)
    if not poly.exterior.is_ccw:
        poly = Polygon(list(poly.exterior.coords)[::-1])
    ring = np.array(poly.exterior.coords)[:-1]
    walls = []
    for k in range(len(ring)):
        p0, p1 = ring[k], ring[(k + 1) % len(ring)]
        d = p1 - p0
        axis = 0 if abs(d[0]) < abs(d[1]) else 2      # plane x=const has a z-running edge
        # inward normal of a CCW ring in (X,Z): rotate the edge direction by +90 deg
        inward = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
        val = p0[0] if axis == 0 else p0[1]
        walls.append({"axis": axis, "value": float(val), "normal": float(np.sign(inward[0 if axis == 0 else 1])),
                      "p0": p0.tolist(), "p1": p1.tolist(), "length": float(np.linalg.norm(d))})
    xz = ring
    stats = {"floor_y": float(floor_y), "ceiling_y": float(ceil_y), "height_m": float(H),
             "ceiling_observed": ceil_obs, "n_walls": len(walls),
             "footprint_m": [float(np.ptp(xz[:, 0])), float(np.ptp(xz[:, 1]))],
             "area_m2": float(poly.area), "n_x_candidates": len(xs), "n_z_candidates": len(zs)}
    return RoomFrame(Rm, float(floor_y), float(ceil_y), ceil_obs, ring, walls, ring.min(0) - 0.5, stats)


def _dedupe(v, tol=0.05):
    out = []
    for x in sorted(v):
        if not out or x - out[-1] > tol:
            out.append(x)
    return out


def _free_space_xz(vol, Rm, floor_y, ceil_y, stride: int = 2):
    """(X,Z) of observed-empty voxels between floor+0.2 m and ceiling-0.2 m."""
    if vol is None:
        return np.zeros((0, 2))
    sub = (vol.weight[::stride, ::stride, ::stride] > 0) & (vol.tsdf[::stride, ::stride, ::stride] > 0.99)
    ijk = np.argwhere(sub) * stride
    if not len(ijk):
        return np.zeros((0, 2))
    P = (ijk * vol.voxel + vol.origin) @ Rm.T
    ok = (P[:, 1] > floor_y + 0.2) & (P[:, 1] < ceil_y - 0.2)
    return P[ok][:, [0, 2]]


def to_layout(rf: RoomFrame) -> Layout:
    """Room frame (metres) -> Mode A Layout (1 px = 1 cm; plan x = room X, plan y = room Z)."""
    o = rf.origin_xz
    pxy = lambda p: [(p[0] - o[0]) * PX_PER_M, (p[1] - o[1]) * PX_PER_M]
    ext = rf.polygon.max(0) - o + 0.5
    t = WALL_T_M * PX_PER_M
    walls = []
    for k, w in enumerate(rf.wall_planes):
        p0, p1 = np.array(w["p0"]), np.array(w["p1"])
        d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
        inward = np.array([-d[1], d[0]])
        off = -inward * WALL_T_M / 2  # centreline sits half a thickness outside the inner face
        a, b = p0 + off - d * WALL_T_M / 2, p1 + off + d * WALL_T_M / 2
        walls.append(Wall(k, pxy(a), pxy(b), t))
    room = Room(0, [pxy(p) for p in rf.polygon], "Undefined", "video room")
    return Layout(int(np.ceil(ext[0] * PX_PER_M)), int(np.ceil(ext[1] * PX_PER_M)), 1.0 / PX_PER_M, "video_metric",
                  walls, [], [room], [],
                  meta={"method": "video", "height_m": rf.ceil_y - rf.floor_y, "ceiling_observed": rf.ceil_observed,
                        "room_frame": {"Rm": rf.Rm.tolist(), "floor_y": rf.floor_y, "origin_xz": o.tolist()}})


def snap_to_planes(m: trimesh.Trimesh, rf: RoomFrame, tol: float = 0.04, cos_tol: float = 0.85) -> tuple[trimesh.Trimesh, float]:
    """Planar regularisation (ours): vertices of the observed mesh that lie within `tol` of the
    floor, ceiling or a wall plane of the room boundary, with a matching normal, are projected
    onto it. Furniture is untouched (different normal or not on a boundary plane)."""
    P = m.vertices @ rf.Rm.T
    N = m.vertex_normals @ rf.Rm.T
    moved = np.zeros(len(P), bool)
    for y, ny in ((rf.floor_y, 1.0), (rf.ceil_y, -1.0)):
        sel = (np.abs(P[:, 1] - y) < tol) & (N[:, 1] * ny > cos_tol)
        P[sel, 1] = y
        moved |= sel
    for w in rf.wall_planes:
        ax = w["axis"]
        other = 2 if ax == 0 else 0
        lo_, hi_ = sorted([w["p0"][0 if other == 0 else 1], w["p1"][0 if other == 0 else 1]])
        sel = (np.abs(P[:, ax] - w["value"]) < tol) & (N[:, ax] * w["normal"] > cos_tol) & \
              (P[:, other] >= lo_ - tol) & (P[:, other] <= hi_ + tol)
        P[sel, ax] = w["value"]
        moved |= sel
    out = m.copy()
    out.vertices = P @ rf.Rm
    return out, float(moved.mean())
