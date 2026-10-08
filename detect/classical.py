"""Classical wall evidence and learned/classical fusion.

Walls are the thickest dark strokes on a plan. A morphological opening with a
kernel just below the thinnest wall removes text, dimension lines, door arcs and
furniture outlines but keeps walls, with pixel-accurate boundaries taken from the
ink itself (the network's masks are blurry at ~4 px output stride).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from detect.cubicasa import SegOutput

WALL, RAILING = 2, 8


@dataclass
class WallEvidence:
    fused: np.ndarray        # uint8 {0,1}: final wall mask
    classical: np.ndarray    # uint8 {0,1}
    learned: np.ndarray      # float wall probability
    railing: np.ndarray      # uint8 {0,1}
    t_min: float             # thin-wall thickness estimate, px
    t_typ: float             # typical wall thickness, px
    t_max: float


def ridge_thickness(mask: np.ndarray) -> np.ndarray:
    """Local thickness samples (2 x distance on the medial ridge)."""
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if not (dt > 0).any():
        return np.array([])
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt > 1)
    return 2 * dt[ridge]


def dark_ink(rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    otsu, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu separates paper from ink, including grey-filled walls. Coloured room
    # fills that slip through are removed by the agreement test in wall_evidence.
    thr = otsu
    return (g < thr).astype(np.uint8)


def wall_evidence(rgb: np.ndarray, seg: SegOutput, fuse: bool = True) -> WallEvidence:
    pw = seg.rooms[WALL] + seg.rooms[RAILING]
    learned_bin = (seg.room_label == WALL).astype(np.uint8)
    rail = (seg.room_label == RAILING).astype(np.uint8)

    ts = ridge_thickness(learned_bin)
    if len(ts) < 20:
        t_min, t_typ, t_max = 4.0, 8.0, 20.0
    else:
        t_min, t_typ, t_max = (float(np.percentile(ts, q)) for q in (10, 50, 95))
    t_min = max(3.0, t_min)

    ink = dark_ink(rgb)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # fill hatching
    k = max(3, int(round(0.6 * t_min)))
    classical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

    if not fuse:
        fused = learned_bin
    else:
        near_learned = cv2.dilate((pw > 0.3).astype(np.uint8), np.ones((5, 5), np.uint8))
        agree = classical & near_learned
        # Keep confident learned walls the ink test misses (outlined/hollow walls).
        fused = agree | (pw > 0.7).astype(np.uint8)
        fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fused &= (1 - rail)
    return WallEvidence(fused, classical, pw, rail, t_min, t_typ, t_max)
