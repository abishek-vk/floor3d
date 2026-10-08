"""CubiCasa5K model.svg -> ground-truth Layout (in F1_scaled.png pixel frame).

Metric scale: each room ("Space") carries a SpaceDimensionsLabel such as
"5.66 m x 3.83 m". Comparing that against the room polygon's extent gives a
metres-per-pixel estimate per room; the GT scale is the median over rooms.
"""
from __future__ import annotations

import os
import re
import sys
from xml.dom import minidom

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.layout import ROOM_CLASSES, Layout, Opening, Room, Wall  # noqa: E402



def _cubicasa_room_map() -> dict:
    """CubiCasa's room-name -> class-index dict, read from house.py without importing
    the floortrans package (its __init__ pulls in lmdb)."""
    import ast
    p = os.path.join(ROOT, "third_party", "CubiCasa5k", "floortrans", "loaders", "house.py")
    for node in ast.parse(open(p, encoding="utf8").read()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "rooms_selected":
            return ast.literal_eval(node.value)
    raise RuntimeError("rooms_selected not found in house.py")


rooms_selected = _cubicasa_room_map()

METRIC_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*m\s*[x×]\s*(\d+(?:[.,]\d+)?)\s*m")


def _poly(e) -> np.ndarray | None:
    pol = next((p for p in e.childNodes if p.nodeName == "polygon"), None)
    if pol is None:
        return None
    pts = [tuple(map(float, a.split(","))) for a in pol.getAttribute("points").split() if "," in a]
    return np.array(pts) if len(pts) >= 3 else None


def _opening(e, kind: str, oid: int) -> Opening | None:
    P = _poly(e)
    if P is None or len(P) < 4:
        return None
    P = P[:4]
    # sides 0-1 / 1-2 of the rectangle: the longer pair is the opening width
    e01, e12 = np.linalg.norm(P[1] - P[0]), np.linalg.norm(P[2] - P[1])
    if e01 >= e12:
        a, b, t = (P[0] + P[3]) / 2, (P[1] + P[2]) / 2, e12
    else:
        a, b, t = (P[0] + P[1]) / 2, (P[3] + P[2]) / 2, e01
    return Opening(oid, kind, a.tolist(), b.tolist(), float(t))


def _texts(e) -> list[str]:
    return ["".join(c.data for c in t.childNodes if c.nodeType == c.TEXT_NODE)
            for t in e.getElementsByTagName("text")]


def load_gt(svg_path: str, width: int, height: int) -> Layout:
    d = minidom.parse(svg_path)
    walls, ops, rooms = [], [], []
    scales = []
    for g in d.getElementsByTagName("g"):
        gid, cls = g.getAttribute("id"), g.getAttribute("class")
        if gid in ("Wall", "Railing"):
            P = _poly(g)
            if P is not None:
                c = P.mean(0).tolist()
                # thickness ~ polygon area / half perimeter-ish (= short side for a rectangle)
                area = 0.5 * abs(np.dot(P[:, 0], np.roll(P[:, 1], 1)) - np.dot(P[:, 1], np.roll(P[:, 0], 1)))
                per = np.sum(np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1))
                t = 2 * area / max(per / 2, 1e-6)
                walls.append(Wall(len(walls), c, c, float(t), P.tolist(), gid.lower()))
        elif gid in ("Door", "Window"):
            o = _opening(g, gid.lower(), len(ops))
            if o is not None:
                ops.append(o)
        elif cls.startswith("Space "):
            P = _poly(g)
            if P is None:
                continue
            name = cls.split(" ")[1] if len(cls.split(" ")) > 1 else "Undefined"
            idx = rooms_selected.get(name, ROOM_CLASSES.index("Undefined"))
            rtype = ROOM_CLASSES[idx] if isinstance(idx, int) and idx < len(ROOM_CLASSES) else "Undefined"
            texts = _texts(g)
            label = next((t for t in texts if t and t.isupper() and not any(ch.isdigit() for ch in t)), None)
            dims = [t for t in texts if METRIC_RE.search(t)]
            rooms.append(Room(len(rooms), P.tolist(), rtype, label, dims))
            for t in dims:
                m = METRIC_RE.search(t)
                a, b = (float(x.replace(",", ".")) for x in m.groups())
                bw, bh = np.ptp(P[:, 0]), np.ptp(P[:, 1])
                if bw < 5 or bh < 5 or a <= 0 or b <= 0:
                    continue
                # label order (w x h vs h x w) is not guaranteed: take the consistent one
                c1 = (a / bw, b / bh)
                c2 = (b / bw, a / bh)
                best = min((c1, c2), key=lambda c: abs(np.log(c[0] / c[1])))
                if abs(np.log(best[0] / best[1])) < 0.25:  # near-rectangular rooms only
                    scales.append(float(np.mean(best)))
    mpp = float(np.median(scales)) if scales else 0.0
    return Layout(width, height, mpp, "gt_labels" if scales else "unknown",
                  walls, ops, rooms, meta={"n_scale_samples": len(scales),
                                           "scale_spread": float(np.std(scales)) if scales else None})


def load_sample(folder: str) -> Layout:
    import cv2
    img = cv2.imread(os.path.join(folder, "F1_scaled.png"))
    h, w = img.shape[:2]
    L = load_gt(os.path.join(folder, "model.svg"), w, h)
    if L.meters_per_px == 0:
        # CubiCasa SVGs are drawn in centimetres (verified: every labelled plan gives
        # 0.0100 m/px), so unlabelled plans fall back to that convention.
        L.meters_per_px, L.scale_source = 0.01, "cubicasa_cm_convention"
    return L


if __name__ == "__main__":
    for f in sys.argv[1:]:
        L = load_sample(f)
        print(f, f"walls={len(L.walls)} openings={len(L.openings)} rooms={len(L.rooms)} "
                 f"m/px={L.meters_per_px:.5f} n={L.meta['n_scale_samples']} sd={L.meta['scale_spread']}")
