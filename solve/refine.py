"""Dimension-constrained layout refinement (research contribution).

Unknowns
  horizontal wall: (y, x0, x1)   vertical wall: (x, y0, y1)   every h/v wall: thickness t
  global log-scale  log s   (metres per pixel)
  Orthogonality / parallelism of h/v walls is exact by parametrisation; diagonal
  walls are held fixed.

Soft constraints (each residual is divided by its sigma; robust Huber loss):
  data        coordinates stay near the detection                 sigma = 0.3 t_typ px
  junction    an endpoint snapped onto another wall stays on it    sigma = 0.5 px
  align       nearly collinear parallel walls share a centreline   sigma = 1.0 px
  thickness   walls in the same thickness cluster agree            sigma = 3.0 px
  span        OCR length "7200" == distance between two wall faces sigma = 1% + 2 cm
  room        OCR "a x b" == room inner width / depth              sigma = 1% + 2 cm
  area        OCR area == room / total area (scale only)           sigma = 3% (log)
  scale       log s near the voting estimate                       sigma = 0.15 (log)

Afterwards rooms and openings follow the walls through a monotone piecewise-
linear remap of x and y. Per-constraint residuals are reported, and annotations
still violated after the solve are flagged as inconsistent.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from core.layout import Layout
from core.log import RunContext
from vectorize.walls import orientation


# Constraint families that are switched on (lets studies ablate individual terms).
ENABLED = {"data", "junction", "align", "thickness", "scale", "span", "room", "area", "total_area"}


class _Model:
    def __init__(self, L: Layout):
        self.L = L
        self.idx = {}      # wall id -> (kind, [i_c, i_a, i_b, i_t])
        x0 = []
        for w in L.walls:
            o = orientation(w)
            if w.kind != "wall" or o == "d":
                continue
            if o == "h":
                vals = [w.p0[1], w.p0[0], w.p1[0], w.thickness]
            else:
                vals = [w.p0[0], w.p0[1], w.p1[1], w.thickness]
            self.idx[w.id] = (o, list(range(len(x0), len(x0) + 4)))
            x0 += vals
        self.i_s = len(x0)
        x0.append(np.log(L.meters_per_px))
        self.x0 = np.array(x0, float)

    def coord(self, x, wid, off=0.0):
        """Across-axis coordinate of a wall's centreline + off * t/2."""
        _, ii = self.idx[wid]
        return x[ii[0]] + off * x[ii[3]] / 2


def _thickness_clusters(ts: np.ndarray, rel_gap: float = 0.3) -> np.ndarray:
    order = np.argsort(ts)
    lab = np.zeros(len(ts), int)
    k = 0
    for a, b in zip(order[:-1], order[1:]):
        if ts[b] > ts[a] * (1 + rel_gap):
            k += 1
        lab[b] = k
    return lab


def build_residuals(L: Layout, M: _Model, t_typ: float):
    terms = []   # (name, fn(x) -> residual, info)
    x0 = M.x0
    sd = max(1.0, 0.3 * t_typ)
    for wid, (o, ii) in M.idx.items():
        for j, i in enumerate(ii[:3]):
            terms.append(("data", lambda x, i=i: (x[i] - x0[i]) / sd, None))
        terms.append(("data", lambda x, i=ii[3]: (x[i] - x0[i]) / 2.0, None))

    # junctions: endpoint of an h wall lying on a v wall's centreline (and vice versa)
    walls = {w.id: w for w in L.walls if w.id in M.idx}
    for a in walls.values():
        oa, ia = M.idx[a.id]
        for b in walls.values():
            ob, ib = M.idx[b.id]
            if oa == ob:
                continue
            # a's along-axis endpoints vs b's across coordinate
            bc = x0[ib[0]]
            lo, hi = x0[ib[1]], x0[ib[2]]
            ac = x0[ia[0]]
            if not (lo - b.thickness <= ac <= hi + b.thickness):
                continue
            for j in (1, 2):
                if abs(x0[ia[j]] - bc) < 1.5:
                    terms.append(("junction", lambda x, p=ia[j], q=ib[0]: (x[p] - x[q]) / 0.5, None))
            # b's endpoint touching a (T or L), symmetric case handled when roles swap

    # alignment of nearly collinear parallel walls
    ws = list(walls.values())
    for i, a in enumerate(ws):
        oa, ia = M.idx[a.id]
        for b in ws[i + 1:]:
            ob, ib = M.idx[b.id]
            if oa != ob:
                continue
            if abs(x0[ia[0]] - x0[ib[0]]) < 0.5 * min(a.thickness, b.thickness) and \
                    abs(x0[ia[0]] - x0[ib[0]]) > 1e-6:
                terms.append(("align", lambda x, p=ia[0], q=ib[0]: (x[p] - x[q]) / 1.0, None))

    # thickness consistency within clusters
    if ws:
        ts = np.array([x0[M.idx[w.id][1][3]] for w in ws])
        lab = _thickness_clusters(ts)
        for k in np.unique(lab):
            mem = [M.idx[w.id][1][3] for w, l_ in zip(ws, lab) if l_ == k]
            if len(mem) < 2:
                continue
            for i in mem:
                terms.append(("thickness", lambda x, i=i, mem=mem: (x[i] - np.mean(x[mem])) / 3.0, None))

    # dimension annotations
    s0 = x0[M.i_s]
    terms.append(("scale", lambda x: (x[M.i_s] - s0) / 0.15, None))
    rooms = {r.id: r for r in L.rooms}
    for c in L.meta.get("dim_constraints", []):
        if c["kind"] == "length" and "lo_ref" in c:
            (wa, oa_), (wb, ob_) = c["lo_ref"], c["hi_ref"]
            if wa not in M.idx or wb not in M.idx:
                continue
            m = c["m"]
            sig = 0.01 * m + 0.02

            def f(x, wa=wa, oa_=oa_, wb=wb, ob_=ob_, m=m, sig=sig):
                return ((M.coord(x, wb, ob_) - M.coord(x, wa, oa_)) * np.exp(x[M.i_s]) - m) / sig
            terms.append(("span", f, {"text": c["text"], "m": m}))
        elif c["kind"] == "pair" and c["room"] in rooms:
            r = rooms[c["room"]]
            P = np.array(r.polygon)
            xmin, xmax, ymin, ymax = P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()
            bounds = {}
            for w in ws:
                o, ii = M.idx[w.id]
                cc, lo, hi, t = x0[ii[0]], x0[ii[1]], x0[ii[2]], x0[ii[3]]
                if o == "v" and lo <= (ymin + ymax) / 2 <= hi:
                    for side, face, off in (("l", xmin, 1.0), ("r", xmax, -1.0)):
                        d = abs(cc + off * t / 2 - face)
                        if d < 4 and (side not in bounds or d < bounds[side][0]):
                            bounds[side] = (d, w.id, off)
                if o == "h" and lo <= (xmin + xmax) / 2 <= hi:
                    for side, face, off in (("t", ymin, 1.0), ("b", ymax, -1.0)):
                        d = abs(cc + off * t / 2 - face)
                        if d < 4 and (side not in bounds or d < bounds[side][0]):
                            bounds[side] = (d, w.id, off)
            a, b = c["a"], c["b"]
            s = np.exp(s0)
            wx, wy = (xmax - xmin) * s, (ymax - ymin) * s
            mx, my = (a, b) if abs(np.log(a / wx)) + abs(np.log(b / wy)) <= \
                abs(np.log(b / wx)) + abs(np.log(a / wy)) else (b, a)
            for (s1, s2), m in (((("l"), ("r")), mx), ((("t"), ("b")), my)):
                if s1 in bounds and s2 in bounds:
                    _, w1, o1 = bounds[s1]
                    _, w2, o2 = bounds[s2]
                    sig = 0.01 * m + 0.02

                    def f(x, w1=w1, o1=o1, w2=w2, o2=o2, m=m, sig=sig):
                        return ((M.coord(x, w2, o2) - M.coord(x, w1, o1)) * np.exp(x[M.i_s]) - m) / sig
                    terms.append(("room", f, {"text": c["text"], "m": m, "room": r.id}))
        elif c["kind"] in ("area", "total_area"):
            if c["kind"] == "area" and c.get("room") in rooms:
                from shapely.geometry import Polygon
                A_px = abs(Polygon(rooms[c["room"]].polygon).area)
            else:
                from shapely.geometry import Polygon
                A_px = sum(abs(Polygon(r.polygon).area) for r in L.rooms
                           if len(r.polygon) >= 3 and r.type != "Outdoor")
            if A_px > 0:
                terms.append((c["kind"], lambda x, A=c["area"], A_px=A_px:
                              (np.log(A_px) + 2 * x[M.i_s] - np.log(A)) / 0.03, {"text": c["text"], "m": c["area"]}))
    return terms


def _remap(old: list[float], new: list[float]):
    """Monotone piecewise-linear map from old to new coordinates."""
    o = np.array(old)
    n = np.array(new)
    order = np.argsort(o)
    o, n = o[order], n[order]
    keep = [0]
    for i in range(1, len(o)):
        if o[i] - o[keep[-1]] > 0.5 and n[i] > n[keep[-1]]:
            keep.append(i)
    o, n = o[keep], n[keep]
    if len(o) < 2:
        return lambda v: v

    def f(v):
        # interpolate inside the knots; outside, shift rigidly by the end offset
        return np.where(v < o[0], v + n[0] - o[0],
                        np.where(v > o[-1], v + n[-1] - o[-1], np.interp(v, o, n)))
    return f


def refine(L: Layout, ctx: RunContext) -> Layout:
    try:
        return _refine(L, ctx)
    except Exception as e:  # the solver is an improvement step; never fail the run on it
        ctx.fallback("solver", f"refinement skipped: {e}")
        return L


def _refine(L: Layout, ctx: RunContext) -> Layout:
    M = _Model(L)
    if len(M.idx) < 2:
        ctx.fallback("solver", "too few axis-aligned walls to refine")
        return L
    t_typ = float(np.median([w.thickness for w in L.walls])) if L.walls else 8.0
    terms = [t for t in build_residuals(L, M, t_typ) if t[0] in ENABLED]
    fun = lambda x: np.array([f(x) for _, f, _ in terms])  # noqa: E731
    r0 = fun(M.x0)
    sol = least_squares(fun, M.x0, loss="huber", f_scale=1.0, method="trf", max_nfev=200)
    x = sol.x
    r1 = fun(x)

    # ---- report
    report = {"n_vars": int(len(x)), "n_terms": len(terms), "cost0": float(0.5 * np.sum(r0 ** 2)),
              "cost1": float(0.5 * np.sum(r1 ** 2)), "by_type": {}, "annotations": []}
    for name in sorted({n for n, _, _ in terms}):
        m = np.array([n == name for n, _, _ in terms])
        report["by_type"][name] = {"n": int(m.sum()), "rms_before": float(np.sqrt(np.mean(r0[m] ** 2))),
                                   "rms_after": float(np.sqrt(np.mean(r1[m] ** 2)))}
    for (name, _, info), a, b in zip(terms, r0, r1):
        if info is None:
            continue
        # residuals are in sigma units; convert span/room ones back to metres
        sig = 0.01 * info["m"] + 0.02 if name in ("span", "room") else None
        flag = abs(b) > 3.0
        report["annotations"].append({
            "type": name, "text": info["text"], "value": info["m"],
            "residual_before": float(a * sig) if sig else float(a),
            "residual_after": float(b * sig) if sig else float(b),
            "units": "m" if sig else "sigma", "inconsistent": bool(flag)})
        if flag:
            ctx.fallback("solver", f"annotation '{info['text']}' inconsistent with layout "
                                   f"(residual {b:.1f} sigma after refinement)")
    L.meta["solver"] = report

    # ---- write back walls, remap everything else
    old_x, new_x, old_y, new_y = [], [], [], []
    for w in L.walls:
        if w.id not in M.idx:
            continue
        o, ii = M.idx[w.id]
        c, a, b, t = x[ii]
        oc, oa, ob, ot = M.x0[ii]
        if o == "h":
            old_y += [oc - ot / 2, oc, oc + ot / 2]
            new_y += [c - t / 2, c, c + t / 2]
            old_x += [oa, ob]
            new_x += [a, b]
            w.p0, w.p1 = [a, c], [b, c]
        else:
            old_x += [oc - ot / 2, oc, oc + ot / 2]
            new_x += [c - t / 2, c, c + t / 2]
            old_y += [oa, ob]
            new_y += [a, b]
            w.p0, w.p1 = [c, a], [c, b]
        w.thickness = float(t)
    fx, fy = _remap(old_x, new_x), _remap(old_y, new_y)

    def mp(p):
        return [float(fx(np.array(p[0]))), float(fy(np.array(p[1])))]
    for r in L.rooms:
        r.polygon = [mp(p) for p in r.polygon]
    for f in L.furniture:
        f.polygon = [mp(p) for p in f.polygon]
    for w in L.walls:
        if w.id not in M.idx:  # diagonals / railings follow the remap
            w.p0, w.p1 = mp(w.p0), mp(w.p1)
    wmap = {w.id: w for w in L.walls}
    for o in L.openings:
        o.p0, o.p1 = mp(o.p0), mp(o.p1)
        if o.wall_id in wmap:
            o.thickness = wmap[o.wall_id].thickness
    L.meters_per_px = float(np.exp(x[M.i_s]))
    if L.meta.get("dim_constraints"):
        L.scale_source = "ocr+solver"
    return L
