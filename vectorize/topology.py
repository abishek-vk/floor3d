"""Topology cleanup for wall segments so rooms close into polygons.

1. merge collinear pieces (across door/window gaps, or small noise gaps)
2. snap endpoints to junctions (L / T / X) via centreline intersections
3. extend dangling ends along their direction to the next wall (closes gaps)
4. drop slivers
"""
from __future__ import annotations

import numpy as np

from core.layout import Opening, Wall
from vectorize.walls import orientation


def _canon(w: Wall) -> Wall:
    o = orientation(w)
    if (o == "h" and w.p0[0] > w.p1[0]) or (o == "v" and w.p0[1] > w.p1[1]):
        w.p0, w.p1 = w.p1, w.p0
    if o == "h":
        y = (w.p0[1] + w.p1[1]) / 2
        w.p0, w.p1 = [w.p0[0], y], [w.p1[0], y]
    elif o == "v":
        x = (w.p0[0] + w.p1[0]) / 2
        w.p0, w.p1 = [x, w.p0[1]], [x, w.p1[1]]
    return w


def _axis_params(w: Wall):
    """For h/v walls: (along_lo, along_hi, across_centre)."""
    if orientation(w) == "h":
        return w.p0[0], w.p1[0], w.p0[1]
    return w.p0[1], w.p1[1], w.p0[0]


def _gap_has_opening(o_axis: str, lo: float, hi: float, c: float, t: float, ops: list[Opening]) -> bool:
    for op in ops:
        m = np.mean([op.p0, op.p1], 0)
        a, b = (m[0], m[1]) if o_axis == "h" else (m[1], m[0])
        half = np.linalg.norm(np.subtract(op.p1, op.p0)) / 2
        if abs(b - c) <= t + op.thickness and a + half >= lo - 2 and a - half <= hi + 2:
            return True
    return False


def merge_collinear(walls: list[Wall], ops: list[Opening], gap_max: float, noise_gap: float) -> list[Wall]:
    out = [w for w in walls if orientation(w) == "d"]
    for ax in ("h", "v"):
        segs = [list(_axis_params(w)) + [w.thickness, w.kind] for w in walls if orientation(w) == ax]
        changed = True
        while changed:
            changed = False
            segs.sort(key=lambda s: (s[4], s[2], s[0]))
            i = 0
            while i < len(segs):
                j = i + 1
                while j < len(segs):
                    a, b = segs[i], segs[j]
                    if a[4] != b[4]:
                        j += 1
                        continue
                    if abs(a[2] - b[2]) > 0.6 * max(a[3], b[3]):
                        j += 1
                        continue
                    gap = max(b[0] - a[1], a[0] - b[1])
                    ok = gap <= noise_gap or (
                        gap <= gap_max and _gap_has_opening(ax, min(a[1], b[1]), max(a[0], b[0]),
                                                            (a[2] + b[2]) / 2, max(a[3], b[3]), ops))
                    if ok:
                        la, lb = a[1] - a[0], b[1] - b[0]
                        wsum = max(la + lb, 1e-6)
                        segs[i] = [min(a[0], b[0]), max(a[1], b[1]), (a[2] * la + b[2] * lb) / wsum,
                                   (a[3] * la + b[3] * lb) / wsum, a[4]]
                        segs.pop(j)
                        changed = True
                    else:
                        j += 1
                i += 1
        for lo, hi, c, t, kind in segs:
            p0, p1 = ([lo, c], [hi, c]) if ax == "h" else ([c, lo], [c, hi])
            out.append(Wall(0, p0, p1, t, kind=kind))
    for i, w in enumerate(out):
        w.id = i
    return out


def _intersect(p, r, q, s):
    """Lines p + t r and q + u s. Returns (t, u) or None if parallel."""
    rxs = r[0] * s[1] - r[1] * s[0]
    if abs(rxs) < 1e-9:
        return None
    qp = q - p
    return (qp[0] * s[1] - qp[1] * s[0]) / rxs, (qp[0] * r[1] - qp[1] * r[0]) / rxs


def snap_junctions(walls: list[Wall], extra: float) -> list[Wall]:
    P = [np.array([w.p0, w.p1], float) for w in walls]
    new = [p.copy() for p in P]
    snapped = [[False, False] for _ in walls]
    for i, wa in enumerate(walls):
        a0, a1 = P[i]
        ra = a1 - a0
        la = np.linalg.norm(ra)
        if la < 1e-6:
            continue
        ua = ra / la
        for end in (0, 1):
            e = P[i][end]
            best = None
            for j, wb in enumerate(walls):
                if i == j:
                    continue
                b0, b1 = P[j]
                rb = b1 - b0
                lb = np.linalg.norm(rb)
                if lb < 1e-6 or abs(np.dot(ua, rb / lb)) > 0.5:  # need >60 deg between walls
                    continue
                res = _intersect(a0, ra, b0, rb)
                if res is None:
                    continue
                t, u = res
                X = a0 + t * ra
                d = np.linalg.norm(X - e)
                tol = wb.thickness / 2 + wa.thickness / 2 + extra
                # X must lie on B's extent (allowing B to be extended by A's half thickness + extra)
                slack = (wa.thickness / 2 + extra) / lb
                if d <= tol and -slack <= u <= 1 + slack:
                    # don't snap an end "through" the wall it belongs to (t must stay near the end)
                    if (end == 0 and t > 0.5) or (end == 1 and t < 0.5):
                        continue
                    if best is None or d < best[0]:
                        best = (d, X)
            if best is not None:
                new[i][end] = best[1]
                snapped[i][end] = True
    for w, p, s in zip(walls, new, snapped):
        w.p0, w.p1 = p[0].tolist(), p[1].tolist()
        w.__dict__["_snapped"] = s
    return walls


def extend_dangling(walls: list[Wall], max_ext: float) -> list[Wall]:
    """Ray-cast free ends along the wall direction to the next wall centreline."""
    P = [np.array([w.p0, w.p1], float) for w in walls]
    for i, w in enumerate(walls):
        s = w.__dict__.get("_snapped", [False, False])
        r = P[i][1] - P[i][0]
        L = np.linalg.norm(r)
        if L < 1e-6:
            continue
        u = r / L
        for end in (0, 1):
            if s[end]:
                continue
            e = P[i][end]
            d = u if end == 1 else -u
            best = None
            for j, wb in enumerate(walls):
                if j == i:
                    continue
                b0, b1 = P[j]
                rb = b1 - b0
                lb = np.linalg.norm(rb)
                if lb < 1e-6 or abs(np.dot(u, rb / lb)) > 0.5:
                    continue
                res = _intersect(e, d, b0, rb)
                if res is None:
                    continue
                t, ub = res
                if 0 < t <= max_ext + wb.thickness / 2 and -0.02 <= ub <= 1.02:
                    if best is None or t < best[0]:
                        best = (t, e + t * d)
            if best is not None:
                if end == 0:
                    w.p0 = best[1].tolist()
                else:
                    w.p1 = best[1].tolist()
    return walls


def cleanup(walls: list[Wall], ops: list[Opening], t_typ: float, door_px: float) -> list[Wall]:
    walls = [_canon(w) for w in walls]
    walls = merge_collinear(walls, ops, gap_max=1.6 * door_px, noise_gap=1.5 * t_typ)
    walls = snap_junctions(walls, extra=max(3.0, 0.5 * t_typ))
    walls = extend_dangling(walls, max_ext=1.2 * door_px)
    walls = [_canon(w) for w in walls]
    # a second merge picks up pieces that became collinear-adjacent after snapping
    walls = merge_collinear(walls, ops, gap_max=1.6 * door_px, noise_gap=1.5 * t_typ)
    walls = snap_junctions(walls, extra=max(3.0, 0.5 * t_typ))
    keep = [w for w in walls if np.linalg.norm(np.subtract(w.p1, w.p0)) >= 0.75 * w.thickness]
    for i, w in enumerate(keep):
        w.id = i
        w.__dict__.pop("_snapped", None)
    return keep


def prune_unbounded(walls: list[Wall], rooms, ops: list[Opening], frac: float = 0.1):
    """Drop walls that bound no room (frame lines, dimension lines, furniture edges read as
    walls), and the openings hosted on them. A wall is kept if >= frac of probe points just
    beside it, on either side, fall inside some room."""
    from shapely.geometry import Point, Polygon
    from shapely.prepared import prep
    polys = [prep(Polygon(r.polygon).buffer(0)) for r in rooms if len(r.polygon) >= 3]
    if not polys:
        return walls, ops
    keep = []
    for w in walls:
        if w.kind != "wall":
            keep.append(w)
            continue
        a, b = np.array(w.p0), np.array(w.p1)
        r = b - a
        L = np.linalg.norm(r)
        if L < 1e-6:
            continue
        u = r / L
        n = np.array([-u[1], u[0]])
        d = w.thickness / 2 + 4
        probes = [a + t * r + s * d * n for t in np.linspace(0.1, 0.9, 7) for s in (1, -1)]
        hits = sum(any(p.contains(Point(*q)) for p in polys) for q in probes)
        if hits / len(probes) >= frac:
            keep.append(w)
    ids = {w.id for w in keep}
    return keep, [o for o in ops if o.wall_id is None or o.wall_id in ids]
