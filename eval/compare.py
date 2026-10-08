"""Overlay GT (green rooms, magenta walls) and a predicted layout (red rooms, blue walls).

    python eval/compare.py <cubicasa_folder> <layout.json> <out.png>
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit("eval", 1)[0])
from core.layout import Layout  # noqa: E402
from eval.gt import load_sample  # noqa: E402
from vectorize.rooms import wall_quad  # noqa: E402


def overlay(folder: str, layout_path: str, out: str) -> None:
    G = load_sample(folder)
    P = Layout.load(layout_path)
    img = cv2.imread(folder + "/F1_scaled.png")
    img = (img * 0.5 + 127).astype(np.uint8)
    for w in G.walls:
        cv2.polylines(img, [np.int32(w.polygon)], True, (255, 0, 255), 1)
    for r in G.rooms:
        cv2.polylines(img, [np.int32(r.polygon)], True, (0, 170, 0), 3)
    for w in P.walls:
        q = np.int32(np.round(wall_quad(w))) if w.polygon is None else np.int32(w.polygon)
        cv2.polylines(img, [q], True, (255, 120, 0), 1)
    for r in P.rooms:
        cv2.polylines(img, [np.int32(r.polygon)], True, (0, 0, 255), 1)
        c = np.int32(np.mean(r.polygon, 0))
        cv2.putText(img, r.type[:6], tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
    for o in P.openings:
        cv2.line(img, tuple(np.int32(o.p0)), tuple(np.int32(o.p1)),
                 (0, 0, 255) if o.type == "door" else (255, 0, 0), 3)
    cv2.imwrite(out, img)


if __name__ == "__main__":
    overlay(*sys.argv[1:4])
