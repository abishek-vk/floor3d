"""Our method: fused detection -> vectorize -> topology -> openings -> rooms -> scale -> solver.

Ablation flags (all default True):
    topology   vectorized walls + snapping/merging + rooms from enclosed space
    ocr_scale  metric scale from OCR'd dimension strings
    solver     dimension-constrained least-squares refinement
"""
from __future__ import annotations

import numpy as np

from core import viz
from core.layout import ROOM_CLASSES, Layout, Room, Wall
from core.log import RunContext
from detect.classical import wall_evidence
from detect.cubicasa import SegOutput
from ingest.load import Ingested
from vectorize import naive


def _door_px(ops, t_typ: float) -> float:
    ws = [np.linalg.norm(np.subtract(o.p1, o.p0)) for o in ops if o.type == "door"]
    ws = [w for w in ws if w > 2 * t_typ]
    return float(np.median(ws)) if ws else 6.0 * t_typ


def full_layout(ing: Ingested, seg: SegOutput, ctx: RunContext, abl: dict) -> Layout:
    from vectorize.openings import attach_openings, detect_openings
    from vectorize.rooms import rooms_from_walls
    from vectorize.topology import cleanup
    from vectorize.walls import extract_walls

    h, w = ing.rgb.shape[:2]
    ev = wall_evidence(ing.rgb, seg, fuse=abl.get("fuse", True))
    ctx.save_img("walls_classical", ev.classical * 255)
    ctx.save_img("walls_fused", np.dstack([ev.fused * 255, (ev.learned > 0.5) * 255, ev.classical * 255]).astype(np.uint8))

    if not abl.get("topology", True):
        # "+fusion" ablation: the fused wall mask with the baseline's naive polygonization.
        L = naive.naive_layout(seg, ctx)
        L.walls = [Wall(i, np.mean(e, 0).tolist(), np.mean(e, 0).tolist(), ev.t_typ, e, "wall", hs)
                   for i, (e, hs) in enumerate(naive.mask_polygons(ev.fused, min_area=30))]
        L.meta["method"] = "fusion_only"
    else:
        walls = extract_walls(ev.fused, ev.t_min, ev.t_max)
        ops_raw = detect_openings(seg)
        door_px = _door_px(ops_raw, ev.t_typ)
        ctx.save_img("walls_raw_segments", viz.draw_layout(ing.rgb, Layout(w, h, 1, walls=walls)))
        walls = cleanup(walls, ops_raw, ev.t_typ, door_px)
        if ev.railing.any():
            rails = extract_walls(ev.railing, 2.0, max(4.0, ev.t_typ))
            for r in rails:
                r.kind = "railing"
                r.id = len(walls)
                walls.append(r)
        ops, dropped = attach_openings(ops_raw, walls)
        if dropped:
            ctx.fallback("openings", f"{dropped} opening(s) had no host wall and were dropped")
        rooms, bar = rooms_from_walls(walls, ops, seg, ev.railing, ev.t_typ,
                                      semantic_split=abl.get("semantic_split", True),
                                      learned_fill=abl.get("learned_fill", True))
        ctx.save_img("barrier", bar * 255)
        if abl.get("prune", True) and rooms:
            from vectorize.topology import prune_unbounded
            n0 = len(walls)
            walls, ops = prune_unbounded(walls, rooms, ops)
            if len(walls) < n0:
                ctx.fallback("walls", f"pruned {n0 - len(walls)} wall(s) that bound no room")
        if not rooms:
            ctx.fallback("rooms", "no enclosed rooms; using learned room components")
            rooms = naive.naive_layout(seg, ctx).rooms
        mpp = naive.scale_from_doors(ops)
        src = "door_width"
        if mpp is None:
            mpp, src = naive.DEFAULT_M_PER_PX, "default"
            ctx.fallback("scale", "no doors detected; using default meters/px prior")
        from vectorize.furniture import detect_furniture
        L = Layout(w, h, mpp, src, walls, ops, rooms, detect_furniture(seg),
                   meta={"method": "full", "wall_t_px": [ev.t_min, ev.t_typ, ev.t_max]})

    if abl.get("ocr_scale", True):
        from scale.ocr import estimate_scale
        L = estimate_scale(ing, L, ctx)
    if abl.get("solver", True) and abl.get("topology", True):
        from solve.refine import refine
        L = refine(L, ctx)
    if abl.get("levels", True) and abl.get("topology", True):
        from vectorize.levels import detect_levels
        L = detect_levels(L, ctx, ing.rgb)
    return L
