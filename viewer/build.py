"""Write a self-contained viewer.html (GLB, layout and plan image embedded as data)."""
from __future__ import annotations

import base64
import json
import os

import cv2
import numpy as np

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html")


def build_viewer(out_html: str, glb_path: str, layout_dict: dict, plan_rgb: np.ndarray | None) -> str:
    html = open(TEMPLATE, encoding="utf8").read()
    glb = base64.b64encode(open(glb_path, "rb").read()).decode()
    plan = ""
    if plan_rgb is not None:
        img = plan_rgb
        k = 2048 / max(img.shape[:2])
        if k < 1:
            img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        plan = base64.b64encode(buf.tobytes()).decode() if ok else ""
    # "</" must not appear inside the inline <script> JSON
    lj = json.dumps(layout_dict).replace("</", "<\\/")
    html = html.replace("__LAYOUT_JSON__", lj).replace("__GLB_B64__", glb).replace("__PLAN_B64__", plan)
    with open(out_html, "w", encoding="utf8") as f:
        f.write(html)
    return out_html
