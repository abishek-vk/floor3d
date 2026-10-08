"""Wall mask -> centreline segments with thickness.

Directional decomposition: opening the mask with a long horizontal (vertical)
line kernel keeps only horizontal (vertical) wall bands. Each band is scanned
column-by-column for its centre and thickness, and split where the centre
jumps. Whatever remains (diagonals, stubs, pillars) is fitted with a
min-area rectangle. True diagonals are kept as diagonals.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.layout import Wall


def _band_segments(W: np.ndarray, L: int, min_len: float) -> list[tuple]:
    """Horizontal bands of W -> [(xa, xb, yc, t)] (pixel-edge x extents)."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
    H = cv2.morphologyEx(W, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(H, connectivity=4)
    out = []
    for i in range(1, n):
        x0, y0, w, h, _ = stats[i]
        if w < L:
            continue
        sub = lab[y0:y0 + h, x0:x0 + w] == i
        has = sub.any(0)
        top = np.argmax(sub, 0)
        bot = h - 1 - np.argmax(sub[::-1], 0)
        cen = y0 + (top + bot) / 2.0
        thk = (bot - top + 1).astype(float)
        xs = np.nonzero(has)[0]
        if len(xs) == 0:
            continue
        # split into runs of continuous columns with a stable centre line
        tol = max(2.0, 0.5 * float(np.median(thk[xs])))
        start = 0
        for j in range(1, len(xs) + 1):
            brk = j == len(xs) or xs[j] != xs[j - 1] + 1 or abs(cen[xs[j]] - cen[xs[j - 1]]) > tol
            if brk:
                run = xs[start:j]
                if len(run) >= min_len:
                    out.append((x0 + run[0] - 0.5, x0 + run[-1] + 0.5,
                                float(np.median(cen[run])), float(np.median(thk[run]))))
                start = j
    return out, H


def extract_walls(W: np.ndarray, t_min: float, t_max: float) -> list[Wall]:
    W = W.astype(np.uint8)
    L = int(max(12, round(1.5 * t_max)))
    min_len = max(4.0, t_min)
    walls: list[Wall] = []

    hs, Hm = _band_segments(W, L, min_len)
    for xa, xb, yc, t in hs:
        walls.append(Wall(len(walls), [xa, yc], [xb, yc], t))
    vs, Vm = _band_segments(np.ascontiguousarray(W.T), L, min_len)
    for ya, yb, xc, t in vs:
        walls.append(Wall(len(walls), [xc, ya], [xc, yb], t))

    # Residual: diagonals, short stubs, pillars.
    covered = cv2.dilate(Hm | np.ascontiguousarray(Vm.T), np.ones((3, 3), np.uint8))
    R = W & (1 - covered)
    R = cv2.morphologyEx(R, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(R, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 2 * t_min * t_min:
            continue
        pts = np.column_stack(np.nonzero(lab == i)[::-1]).astype(np.float32)
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
        if w < h:
            w, h, ang = h, w, ang + 90
        a = np.radians(ang)
        # snap near-axis residuals (stubs); keep genuine diagonals
        a_deg = (ang + 180) % 180
        for ax in (0, 90, 180):
            if abs(a_deg - ax) < 8:
                a = np.radians(ax)
        d = np.array([np.cos(a), np.sin(a)]) * w / 2
        c = np.array([cx, cy])
        walls.append(Wall(len(walls), (c - d).tolist(), (c + d).tolist(), float(max(h, t_min))))
    return walls


def orientation(w: Wall) -> str:
    dx, dy = abs(w.p1[0] - w.p0[0]), abs(w.p1[1] - w.p0[1])
    if dy < 1e-6 or dy / max(dx, 1e-6) < 0.02:
        return "h"
    if dx < 1e-6 or dx / max(dy, 1e-6) < 0.02:
        return "v"
    return "d"
