"""Layout-vs-layout metrics. Both layouts must be in the same pixel frame."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from core.layout import ROOM_CLASSES, Layout
from extrude.mesh import wall_footprint

ROOM_TYPES = [c for c in ROOM_CLASSES if c not in ("Background", "Wall", "Railing")]


def _fill(shape, rings_list) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for rings in rings_list:
        cv2.fillPoly(m, [np.round(np.asarray(r)).astype(np.int32) for r in rings], 1)
    return m


def room_masks(L: Layout, shape) -> list[np.ndarray]:
    return [_fill(shape, [[r.polygon]]) for r in L.rooms if len(r.polygon) >= 3]


def wall_mask(L: Layout, shape) -> np.ndarray:
    g = wall_footprint(L.walls, 1.0)
    polys = [g] if g.geom_type == "Polygon" else list(getattr(g, "geoms", []))
    rings = [[np.array(p.exterior.coords)] + [np.array(i.coords) for i in p.interiors]
             for p in polys if p.geom_type == "Polygon" and not p.is_empty]
    return _fill(shape, rings)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def class_map(L: Layout, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)  # 0 = background
    for r in sorted(L.rooms, key=lambda r: -abs(Polygon(r.polygon).area) if len(r.polygon) >= 3 else 0):
        if len(r.polygon) >= 3:
            cv2.fillPoly(m, [np.round(np.asarray(r.polygon)).astype(np.int32)],
                         ROOM_TYPES.index(r.type) + 1 if r.type in ROOM_TYPES else len(ROOM_TYPES))
    return m


def rect_dims(poly) -> tuple[float, float]:
    """(short, long) side of the min-area rectangle, in polygon units."""
    (_, _), (w, h), _ = cv2.minAreaRect(np.asarray(poly, np.float32))
    return min(w, h), max(w, h)


def match_rooms(P: Layout, G: Layout, shape):
    pm, gm = room_masks(P, shape), room_masks(G, shape)
    if not pm or not gm:
        return [], np.zeros((len(gm), len(pm))), pm, gm
    # IoU matrix computed only on overlapping bounding boxes (full-image products are slow)
    def bbox(m):
        ys, xs = np.nonzero(m)
        return (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1) if len(ys) else None
    pb, gb = [bbox(m) for m in pm], [bbox(m) for m in gm]
    pa, ga = [int(m.sum()) for m in pm], [int(m.sum()) for m in gm]
    M = np.zeros((len(gm), len(pm)))
    for i, (g, b) in enumerate(zip(gm, gb)):
        if b is None:
            continue
        for j, (p, c) in enumerate(zip(pm, pb)):
            if c is None:
                continue
            y0, y1, x0, x1 = max(b[0], c[0]), min(b[1], c[1]), max(b[2], c[2]), min(b[3], c[3])
            if y0 >= y1 or x0 >= x1:
                continue
            inter = int(np.count_nonzero(g[y0:y1, x0:x1] & p[y0:y1, x0:x1]))
            M[i, j] = inter / (ga[i] + pa[j] - inter) if inter else 0.0
    gi, pi = linear_sum_assignment(-M)
    return list(zip(gi.tolist(), pi.tolist())), M, pm, gm


def opening_pr(P: Layout, G: Layout, kind: str) -> tuple[int, int, int]:
    """Greedy match by center distance (< 0.5 * GT width, min 15 px). Returns tp, n_pred, n_gt."""
    pc = [np.mean([o.p0, o.p1], 0) for o in P.openings if o.type == kind]
    gl = [o for o in G.openings if o.type == kind]
    gc = [np.mean([o.p0, o.p1], 0) for o in gl]
    if not pc or not gc:
        return 0, len(pc), len(gc)
    D = np.linalg.norm(np.array(gc)[:, None] - np.array(pc)[None], axis=2)
    thr = np.array([max(15.0, 0.5 * np.linalg.norm(np.subtract(o.p1, o.p0))) for o in gl])
    gi, pi = linear_sum_assignment(D)
    tp = int(sum(D[g, p] <= thr[g] for g, p in zip(gi, pi)))
    return tp, len(pc), len(gc)


def evaluate(P: Layout, G: Layout, iou_thr: float = 0.5) -> dict:
    shape = (G.height, G.width)
    out = {}

    pairs, M, pm, gm = match_rooms(P, G, shape)
    matched = {g: p for g, p in pairs}
    # Layout IoU: mean over GT rooms of IoU with its matched prediction (0 if unmatched)
    out["room_iou"] = float(np.mean([M[g, matched[g]] if g in matched else 0 for g in range(len(gm))])) if gm else 0.0
    # Pixel-wise room-class mIoU over classes present in GT or prediction
    pc, gc = class_map(P, shape), class_map(G, shape)
    cls = sorted((set(np.unique(gc)) | set(np.unique(pc))) - {0})
    out["class_miou"] = float(np.mean([iou(pc == c, gc == c) for c in cls])) if cls else 0.0
    out["footprint_iou"] = iou(pc > 0, gc > 0)
    out["wall_iou"] = iou(wall_mask(P, shape), wall_mask(G, shape))

    good = [(g, p) for g, p in pairs if M[g, p] >= iou_thr]
    out["n_rooms_gt"], out["n_rooms_pred"] = len(gm), len(pm)
    out["room_recall@.5"] = len(good) / len(gm) if gm else 0.0
    out["room_precision@.5"] = len(good) / len(pm) if pm else 0.0
    out["room_count_abs_err"] = abs(len(pm) - len(gm))

    for kind in ("door", "window"):
        tp, npred, ngt = opening_pr(P, G, kind)
        out[f"{kind}_tp"], out[f"{kind}_npred"], out[f"{kind}_ngt"] = tp, npred, ngt
        out[f"{kind}_precision"] = tp / npred if npred else (1.0 if ngt == 0 else 0.0)
        out[f"{kind}_recall"] = tp / ngt if ngt else 1.0

    # Metric dimensions on matched rooms: each layout uses its OWN scale, so scale errors count.
    sp, sg = P.meters_per_px, G.meters_per_px
    gr = [r for r in G.rooms if len(r.polygon) >= 3]
    prs = [r for r in P.rooms if len(r.polygon) >= 3]
    dim_err, area_err = [], []
    for g, p in good:
        gs, gl_ = rect_dims(gr[g].polygon)
        ps, pl = rect_dims(prs[p].polygon)
        if gs * sg < 0.5:  # skip slivers (<0.5 m) where % error is meaningless
            continue
        dim_err += [abs(ps * sp - gs * sg) / (gs * sg), abs(pl * sp - gl_ * sg) / (gl_ * sg)]
        ga = abs(Polygon(gr[g].polygon).area) * sg ** 2
        pa = abs(Polygon(prs[p].polygon).area) * sp ** 2
        area_err.append(abs(pa - ga) / ga)
    out["dim_mape"] = float(np.mean(dim_err)) if dim_err else np.nan
    out["area_mape"] = float(np.mean(area_err)) if area_err else np.nan
    ta_g = sum(abs(Polygon(r.polygon).area) for r in gr) * sg ** 2
    ta_p = sum(abs(Polygon(r.polygon).area) for r in prs) * sp ** 2
    out["total_area_ape"] = abs(ta_p - ta_g) / ta_g if ta_g else np.nan
    out["scale_ape"] = abs(sp - sg) / sg if sg else np.nan
    out["scale_source"] = P.scale_source
    return out
