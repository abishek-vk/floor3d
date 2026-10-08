"""Furniture / fixture symbols -> oriented footprints (bonus: object-level reconstruction)."""
from __future__ import annotations

import cv2
import numpy as np

from core.layout import Furniture
from detect.cubicasa import SegOutput

# icon class -> (name, proxy height in metres)
PROXY = {3: ("closet", 2.0), 4: ("appliance", 0.9), 5: ("toilet", 0.45), 6: ("sink", 0.85),
         7: ("sauna_bench", 0.9), 8: ("fireplace", 1.2), 9: ("bathtub", 0.55), 10: ("chimney", 2.7)}


def detect_furniture(seg: SegOutput, min_area: float = 60) -> list[Furniture]:
    out = []
    lab_all = seg.icon_label
    for cls in PROXY:
        n, lab, st, _ = cv2.connectedComponentsWithStats((lab_all == cls).astype(np.uint8))
        for k in range(1, n):
            if st[k, cv2.CC_STAT_AREA] < min_area:
                continue
            pts = np.column_stack(np.nonzero(lab == k)[::-1]).astype(np.float32)
            box = cv2.boxPoints(cv2.minAreaRect(pts))
            out.append(Furniture(len(out), PROXY[cls][0], box.tolist()))
    return out


HEIGHT = {name: h for name, h in PROXY.values()}
