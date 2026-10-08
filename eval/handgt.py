"""Evaluate on hand-annotated plans with *measured* room dimensions.

Annotation format (one JSON per plan, next to the image, quick to make):

    {
      "image": "flat_03.png",
      "total_area_m2": 74.5,                    # optional
      "rooms": [
        {"name": "Bedroom", "point": [412, 380], "width_m": 3.40, "length_m": 4.10},
        {"name": "Bath",    "point": [620, 190], "width_m": 1.80, "length_m": 2.30}
      ]
    }

`point` is any pixel inside the room on the original image. Width/length order does not
matter (both are sorted). Run:

    python eval/handgt.py --dir data/hand --out results/hand
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys

import numpy as np
from shapely.geometry import Point, Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.layout import Layout  # noqa: E402
from eval.metrics import rect_dims  # noqa: E402


def score(L: Layout, ann: dict) -> dict:
    s = L.meters_per_px
    polys = [(r, Polygon(r.polygon).buffer(0)) for r in L.rooms if len(r.polygon) >= 3]
    errs, area_errs, found, used = [], [], 0, set()
    for a in ann["rooms"]:
        hit = next(((r, p) for r, p in polys if p.contains(Point(*a["point"]))), None)
        if hit is None or hit[0].id in used:
            continue  # missing, or merged with an already-counted room
        r, p = hit
        used.add(r.id)
        found += 1
        ps, pl = sorted(np.array(rect_dims(r.polygon)) * s)
        gs, gl = sorted([a["width_m"], a["length_m"]])
        errs += [abs(ps - gs) / gs, abs(pl - gl) / gl]
        area_errs.append(abs(p.area * s * s - gs * gl) / (gs * gl))
    out = {"rooms_annotated": len(ann["rooms"]), "rooms_found": found,
           "completeness": found / max(1, len(ann["rooms"])),
           "dim_mape": float(np.mean(errs)) if errs else float("nan"),
           "area_mape": float(np.mean(area_errs)) if area_errs else float("nan"),
           "scale_source": L.scale_source}
    if "total_area_m2" in ann:
        tot = sum(p.area for r, p in polys if r.type != "Outdoor") * s * s
        out["total_area_ape"] = abs(tot - ann["total_area_m2"]) / ann["total_area_m2"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "data", "hand"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "hand"))
    ap.add_argument("--methods", nargs="+", default=["baseline", "full"])
    a = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    from run import run
    rows = []
    for jf in sorted(glob.glob(os.path.join(a.dir, "*.json"))):
        ann = json.load(open(jf, encoding="utf8"))
        img = os.path.join(os.path.dirname(jf), ann["image"])
        name = os.path.splitext(os.path.basename(jf))[0]
        for m in a.methods:
            od = os.path.join(a.out, m, name)
            if not os.path.exists(os.path.join(od, "layout.json")):
                run(img, od, m)
            r = score(Layout.load(os.path.join(od, "layout.json")), ann)
            r.update({"plan": name, "method": m})
            rows.append(r)
            print(f"{m:9s} {name:25s} dimMAPE={r['dim_mape']:.3f} areaMAPE={r['area_mape']:.3f} "
                  f"complete={r['completeness']:.2f} scale={r['scale_source']}")
    if not rows:
        print(f"no annotations found in {a.dir}")
        return
    lines = ["| Method | Dim MAPE | Area MAPE | Completeness | Total-area APE |", "|---|---|---|---|---|"]
    for m in a.methods:
        R = [r for r in rows if r["method"] == m]
        f = lambda k: np.nanmean([r.get(k, np.nan) for r in R])  # noqa: E731
        lines.append(f"| {m} | {f('dim_mape'):.3f} | {f('area_mape'):.3f} | {f('completeness'):.3f} | "
                     f"{f('total_area_ape'):.3f} |")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf8") as fh:
        fh.write(f"# Hand-annotated plans ({len(rows) // len(a.methods)})\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
