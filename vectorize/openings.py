"""Doors/windows: detect from icon probabilities, then attach to a host wall."""
from __future__ import annotations

import cv2
import numpy as np

from core.layout import Opening, Wall
from detect.cubicasa import SegOutput

DOOR, WINDOW = 2, 1


def detect_openings(seg: SegOutput, min_area: float = 20) -> list[Opening]:
    ops = []
    lab_all = seg.icon_label
    for cls, name in ((DOOR, "door"), (WINDOW, "window")):
        n, lab, stats, _ = cv2.connectedComponentsWithStats((lab_all == cls).astype(np.uint8))
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < min_area:
                continue
            m = lab == k
            pts = np.column_stack(np.nonzero(m)[::-1]).astype(np.float32)
            (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
            if w < h:
                w, h, ang = h, w, ang + 90
            d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))]) * w / 2
            c = np.array([cx, cy])
            conf = float(seg.icons[cls][m].mean())
            ops.append(Opening(len(ops), name, (c - d).tolist(), (c + d).tolist(), float(max(h, 1)),
                               confidence=conf))
    return ops


def attach_openings(ops: list[Opening], walls: list[Wall], tol_extra: float = 4.0) -> tuple[list[Opening], int]:
    """Project each opening onto its host wall. Returns (attached, n_dropped)."""
    out, dropped = [], 0
    for o in ops:
        c = np.mean([o.p0, o.p1], 0)
        ou = np.subtract(o.p1, o.p0)
        ol = np.linalg.norm(ou)
        if ol < 2:
            dropped += 1
            continue
        ou = ou / ol
        best = None
        for w in walls:
            if w.kind != "wall":
                continue
            a, b = np.array(w.p0), np.array(w.p1)
            r = b - a
            L = np.linalg.norm(r)
            if L < 1e-6:
                continue
            u = r / L
            if abs(np.dot(u, ou)) < 0.9:  # within ~25 deg
                continue
            s = np.dot(c - a, u)
            v = c - a
            perp = abs(u[0] * v[1] - u[1] * v[0])
            if s < -ol / 2 or s > L + ol / 2:
                continue
            for mult in (1.0, 2.0):  # strict first, then relaxed
                if perp <= mult * (w.thickness / 2 + o.thickness / 2) + tol_extra:
                    score = perp + (0 if mult == 1 else 1000)
                    if best is None or score < best[0]:
                        best = (score, w, a, u, L, s)
                    break
        if best is None:
            dropped += 1
            continue
        _, w, a, u, L, s = best
        half = min(ol / 2, L / 2)
        s = float(np.clip(s, half, L - half))
        p0, p1 = a + (s - half) * u, a + (s + half) * u
        out.append(Opening(len(out), o.type, p0.tolist(), p1.tolist(), w.thickness, w.id, s, o.confidence))
    # de-duplicate overlapping openings on the same wall (keep the more confident)
    out.sort(key=lambda o: -o.confidence)
    kept = []
    for o in out:
        if any(k.wall_id == o.wall_id and abs(k.offset - o.offset) <
               0.5 * (np.linalg.norm(np.subtract(k.p1, k.p0)) + np.linalg.norm(np.subtract(o.p1, o.p0)))
               for k in kept):
            continue
        kept.append(o)
    for i, o in enumerate(kept):
        o.id = i
    return kept, dropped
