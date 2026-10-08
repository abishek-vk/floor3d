"""Debug drawings for each pipeline stage."""
from __future__ import annotations

import cv2
import numpy as np

from core.layout import Layout

ROOM_PALETTE = np.array([
    [255, 255, 255], [160, 200, 140], [40, 40, 40], [242, 204, 140], [230, 184, 140],
    [178, 190, 235], [153, 217, 230], [217, 209, 191], [120, 120, 130], [199, 191, 178],
    [166, 166, 166], [217, 217, 209]], np.uint8)
ICON_PALETTE = np.array([
    [0, 0, 0], [30, 144, 255], [220, 60, 60], [150, 100, 50], [255, 200, 0], [120, 220, 120],
    [0, 200, 200], [200, 120, 200], [255, 120, 0], [100, 100, 255], [90, 90, 90]], np.uint8)


def seg_overlay(rgb: np.ndarray, room_label: np.ndarray, icon_label: np.ndarray) -> np.ndarray:
    col = ROOM_PALETTE[room_label].copy()
    ic = icon_label > 0
    col[ic] = ICON_PALETTE[icon_label[ic]]
    return cv2.addWeighted(rgb, 0.35, col, 0.65, 0)


def draw_layout(rgb: np.ndarray | None, L: Layout, k: float = 1.0) -> np.ndarray:
    """Draw a layout over the image (or white). k scales layout px -> image px."""
    if rgb is None:
        img = np.full((round(L.height * k), round(L.width * k), 3), 255, np.uint8)
    else:
        img = (rgb.astype(np.float32) * 0.4 + 255 * 0.6).astype(np.uint8)
    P = lambda pts: (np.array(pts) * k).round().astype(np.int32)  # noqa: E731
    over = img.copy()
    for r in L.rooms:
        c = ROOM_PALETTE[min(_room_idx(r.type), 11)].tolist()
        cv2.fillPoly(over, [P(r.polygon)], c)
    img = cv2.addWeighted(img, 0.4, over, 0.6, 0)
    for r in L.rooms:
        cv2.polylines(img, [P(r.polygon)], True, (90, 90, 90), 1)
    for w in L.walls:
        col = (30, 30, 30) if w.kind == "wall" else (120, 120, 200)
        if w.polygon is not None:
            # all rings in one call -> even-odd fill leaves the holes open
            cv2.fillPoly(img, [P(w.polygon)] + [P(h) for h in w.holes or []], col)
        else:
            cv2.line(img, tuple(P(w.p0)), tuple(P(w.p1)), col, max(1, int(w.thickness * k)))
            cv2.circle(img, tuple(P(w.p0)), 2, (0, 0, 255), -1)
            cv2.circle(img, tuple(P(w.p1)), 2, (0, 0, 255), -1)
    for o in L.openings:
        col = (220, 60, 60) if o.type == "door" else (30, 144, 255)
        cv2.line(img, tuple(P(o.p0)), tuple(P(o.p1)), col, max(2, int(o.thickness * k)))
    for r in L.rooms:
        c = P(np.mean(r.polygon, 0))
        txt = r.label or r.type
        cv2.putText(img, txt, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def _room_idx(t: str) -> int:
    from core.layout import ROOM_CLASSES
    return ROOM_CLASSES.index(t) if t in ROOM_CLASSES else 11
