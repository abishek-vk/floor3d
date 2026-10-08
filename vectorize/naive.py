"""BASELINE: raw CubiCasa5K argmax masks -> naive polygonization.

Deliberately simple (this is what we must beat):
- walls   = contours of the Wall/Railing mask, extruded as-is
- rooms   = one approxPolyDP contour per connected component of each room class
- openings= minAreaRect of each Door/Window icon component
- scale   = median detected door width := 0.85 m (else a fixed default)
"""
from __future__ import annotations

import cv2
import numpy as np

from core.layout import ROOM_CLASSES, Layout, Opening, Room, Wall
from core.log import RunContext
from detect.cubicasa import SegOutput

WALL, RAILING = 2, 8
ROOM_IDS = [i for i, n in enumerate(ROOM_CLASSES) if n not in ("Background", "Wall", "Railing")]
DOOR_M = 0.78  # median drawn door opening, calibrated on the val split only
DEFAULT_M_PER_PX = 0.0125  # last-resort prior; recalibrated from val-set GT later


def _ring(c: np.ndarray) -> list[list[float]]:
    return c.reshape(-1, 2).astype(float).tolist()


def mask_polygons(mask: np.ndarray, min_area: float, eps: float = 1.5):
    """External contours with holes, as (exterior, [holes]) tuples."""
    cs, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    if hier is None:
        return out
    hier = hier[0]
    for i, c in enumerate(cs):
        if hier[i][3] != -1 or cv2.contourArea(c) < min_area:
            continue
        ext = cv2.approxPolyDP(c, eps, True)
        if len(ext) < 3:
            continue
        holes = []
        j = hier[i][2]
        while j != -1:
            if cv2.contourArea(cs[j]) >= min_area:
                h = cv2.approxPolyDP(cs[j], eps, True)
                if len(h) >= 3:
                    holes.append(_ring(h))
            j = hier[j][0]
        out.append((_ring(ext), holes))
    return out


def openings_from_icons(icon_label: np.ndarray, min_area: float = 20) -> list[Opening]:
    ops = []
    for cls, name in ((2, "door"), (1, "window")):
        n, lab, stats, _ = cv2.connectedComponentsWithStats((icon_label == cls).astype(np.uint8))
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < min_area:
                continue
            pts = np.column_stack(np.nonzero(lab == k)[::-1]).astype(np.float32)
            (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
            if w < h:
                w, h, ang = h, w, ang + 90
            d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))]) * w / 2
            c = np.array([cx, cy])
            ops.append(Opening(len(ops), name, (c - d).tolist(), (c + d).tolist(), float(max(h, 1))))
    return ops


def scale_from_doors(ops: list[Opening]) -> float | None:
    ws = [np.hypot(*np.subtract(o.p1, o.p0)) for o in ops if o.type == "door"]
    ws = np.array([w for w in ws if w > 5])
    if len(ws) < 1:
        return None
    # fragmented detections are much narrower than real doors: drop them
    ws = ws[ws >= 0.5 * np.percentile(ws, 90)]
    return DOOR_M / float(np.median(ws))


def naive_layout(seg: SegOutput, ctx: RunContext, min_room_px: float = 400) -> Layout:
    rl = seg.room_label
    h, w = rl.shape

    walls = []
    for kind, cls in (("wall", WALL), ("railing", RAILING)):
        m = (rl == cls).astype(np.uint8)
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        thick = float(2 * np.median(dt[dt > 0])) if (dt > 0).any() else 6.0
        for ext, holes in mask_polygons(m, min_area=30):
            c = np.mean(ext, 0).tolist()
            walls.append(Wall(len(walls), c, c, thick, ext, kind, holes))

    rooms = []
    for cls in ROOM_IDS:
        for ext, _ in mask_polygons(rl == cls, min_area=min_room_px, eps=2.0):
            rooms.append(Room(len(rooms), ext, ROOM_CLASSES[cls]))

    ops = openings_from_icons(seg.icon_label)

    mpp = scale_from_doors(ops)
    src = "door_width"
    if mpp is None:
        mpp, src = DEFAULT_M_PER_PX, "default"
        ctx.fallback("scale", "no doors detected; using default meters/px prior")

    return Layout(w, h, mpp, src, walls, ops, rooms, meta={"method": "baseline"})
