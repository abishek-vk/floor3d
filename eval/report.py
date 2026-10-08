"""Assemble results/results.md: main table, ablations, solver study, figures.

    python eval/report.py --split test --dir results
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.render import render  # noqa: E402

DATA = os.path.join(ROOT, "data", "cubicasa5k")


def folder_of(sid: str) -> str:
    a, b = sid.rsplit("_", 1)
    return os.path.join(DATA, a, b)


def figure(sid: str, d: str, figdir: str) -> str:
    """plan | baseline 3D | ours 3D, side by side."""
    tiles = []
    plan = cv2.imread(os.path.join(folder_of(sid), "F1_scaled.png"))
    tiles.append(plan)
    for cfg in ("baseline", "full"):
        glb = os.path.join(d, "runs", cfg, sid, "model.glb")
        png = os.path.join(figdir, f"{sid}_{cfg}.png")
        if not os.path.exists(png):
            render(glb, png)
        tiles.append(cv2.imread(png))
    h = 420
    tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
    out = os.path.join(figdir, f"{sid}_side_by_side.png")
    cv2.imwrite(out, np.hstack(tiles))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--n-figs", type=int, default=3)
    a = ap.parse_args()
    d = a.dir
    figdir = os.path.join(d, "figures")
    os.makedirs(figdir, exist_ok=True)

    rows = list(csv.DictReader(open(os.path.join(d, f"per_sample_{a.split}.csv"))))
    by = {}
    for r in rows:
        by.setdefault(r["sample"], {})[r["config"]] = r
    gain = sorted(((float(v["full"]["room_iou"]) - float(v["baseline"]["room_iou"]), s)
                   for s, v in by.items() if "full" in v and "baseline" in v), reverse=True)
    picks = [s for _, s in gain[:a.n_figs]]
    worst = gain[-1][1] if gain else None

    summary = open(os.path.join(d, f"summary_{a.split}.md"), encoding="utf8").read()
    study_p = os.path.join(d, f"solver_study_{a.split}.md")
    study = open(study_p, encoding="utf8").read() if os.path.exists(study_p) else "(not run)"

    def cell(s, c, k):
        return f"{float(by[s][c][k]):.3f}"

    per = ["| Plan | Room IoU base -> ours | Wall IoU base -> ours | Dim MAPE base -> ours | Scale source (ours) |",
           "|---|---|---|---|---|"]
    for _, s in gain:
        per.append(f"| {s} | {cell(s, 'baseline', 'room_iou')} -> {cell(s, 'full', 'room_iou')} | "
                   f"{cell(s, 'baseline', 'wall_iou')} -> {cell(s, 'full', 'wall_iou')} | "
                   f"{cell(s, 'baseline', 'dim_mape')} -> {cell(s, 'full', 'dim_mape')} | "
                   f"{by[s]['full']['scale_source']} |")

    figs = []
    for s in picks + ([worst] if worst and worst not in picks else []):
        p = figure(s, d, figdir)
        tag = "largest loss vs baseline" if s == worst else "largest gains"
        figs.append(f"**{s}** ({tag}): plan | baseline | ours\n\n![{s}](figures/{os.path.basename(p)})\n")

    md = f"""# Results ({a.split} split, held out)

All numbers: CubiCasa5K **{a.split}** plans never used for development (tuning used the val split).
GT = the plans' SVG annotations; metric GT scale comes from the per-room size labels
(every labelled plan gives exactly 1 cm/px, see README).

## Baseline vs ours, with ablations

{summary.split(chr(10), 2)[2] if summary.startswith('#') else summary}

Metric definitions (`eval/metrics.py`):
- **Room IoU**: Hungarian-matched room polygons; mean IoU over GT rooms (unmatched = 0).
- **Class mIoU**: pixel-wise room-type IoU, averaged over classes present.
- **Wall IoU**: rasterised wall footprints.
- **Room R/P@.5**: completeness: GT rooms matched at IoU >= 0.5 / predicted rooms matched.
- **Door/Win P/R**: centre-distance matching (< half the GT width, min 15 px).
- **Dim MAPE**: mean |error| of matched rooms' width and length, in **metres** (each method uses its own scale).
- **Area MAPE / Tot.Area APE**: per-room / total floor area, in m^2.
- **Scale APE**: |m/px - GT m/px| / GT.

Rows: `baseline` = raw CubiCasa5K argmax -> contours (same extruder). `+fusion` = classical
wall evidence fused into the wall mask. `+topology` = vectorised walls, snapping/merging,
rooms from enclosed space split at reflex corners by learned class. `+ocr_scale` = metric
scale by OCR vote. `full` = + dimension-constrained solver.

## Solver study (oracle annotations)

{study}

## Per-plan comparison

{chr(10).join(per)}

## Figures

{chr(10).join(figs)}
"""
    with open(os.path.join(d, "results.md"), "w", encoding="utf8") as fh:
        fh.write(md)
    print(os.path.join(d, "results.md"))


if __name__ == "__main__":
    main()
