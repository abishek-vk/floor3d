"""Run methods/ablations over a CubiCasa split and score them against SVG ground truth.

    python eval/eval.py --split test --configs baseline full --out results/
    python eval/eval.py --split val --n 5 --configs baseline        # quick dev loop

Per-sample rows go to <out>/per_sample.csv, aggregate table to <out>/summary.md.
Runs are cached under <out>/runs/<config>/<sample>/ (use --force to recompute).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.layout import Layout  # noqa: E402
from eval.gt import load_sample  # noqa: E402
from eval.metrics import evaluate  # noqa: E402

# name -> (method, ablation flags). Flags are read by pipeline.full_layout.
CONFIGS = {
    "baseline": ("baseline", {}),
    "+fusion": ("full", {"topology": False, "ocr_scale": False, "solver": False}),
    "+topology": ("full", {"topology": True, "ocr_scale": False, "solver": False}),
    "dev-nofill": ("full", {"topology": True, "ocr_scale": False, "solver": False, "learned_fill": False}),
    "dev-nofuse": ("full", {"topology": True, "ocr_scale": False, "solver": False, "fuse": False}),
    "dev-nosplit": ("full", {"topology": True, "ocr_scale": False, "solver": False, "semantic_split": False}),
    "+ocr_scale": ("full", {"topology": True, "ocr_scale": True, "solver": False}),
    "full": ("full", {"topology": True, "ocr_scale": True, "solver": True}),
}

SUMMARY_COLS = [
    ("room_iou", "Room IoU"), ("class_miou", "Class mIoU"), ("wall_iou", "Wall IoU"),
    ("room_recall@.5", "Room R@.5"), ("room_precision@.5", "Room P@.5"),
    ("door_precision", "Door P"), ("door_recall", "Door R"),
    ("window_precision", "Win P"), ("window_recall", "Win R"),
    ("dim_mape", "Dim MAPE"), ("area_mape", "Area MAPE"), ("total_area_ape", "Tot.Area APE"),
    ("scale_ape", "Scale APE"), ("runtime_s", "Time s"),
]


def samples(split: str, n: int, data: str) -> list[str]:
    folders = []
    for line in open(os.path.join(data, f"{split}.txt")):
        f = os.path.join(data, line.strip().strip("/"))
        if os.path.exists(os.path.join(f, "model.svg")):
            folders.append(f)
    return sorted(folders)[:n] if n else sorted(folders)


def sample_id(folder: str) -> str:
    parts = os.path.normpath(folder).split(os.sep)
    return f"{parts[-2]}_{parts[-1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "cubicasa5k"))
    ap.add_argument("--configs", nargs="+", default=[c for c in CONFIGS if not c.startswith("dev-")])
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-debug", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    from run import run

    rows = []
    folders = samples(a.split, a.n, a.data)
    print(f"{len(folders)} samples from {a.split}")
    for cfg in a.configs:
        method, abl = CONFIGS[cfg]
        for f in folders:
            sid = sample_id(f)
            od = os.path.join(a.out, "runs", cfg, sid)
            lp = os.path.join(od, "layout.json")
            t = None
            if a.force or not os.path.exists(lp):
                t0 = time.perf_counter()
                try:
                    run(os.path.join(f, "F1_scaled.png"), od, method, debug=not a.no_debug, ablation=abl,
                        viewer=cfg in ("baseline", "full"))
                except Exception as e:  # a crash scores as an empty layout, never aborts eval
                    print(f"  !! {cfg} {sid}: {e}")
                t = time.perf_counter() - t0
            G = load_sample(f)
            if os.path.exists(lp):
                P = Layout.load(lp)
                rep = json.load(open(os.path.join(od, "run_report.json")))
                t = t or rep.get("total_s")
            else:
                P = Layout(G.width, G.height, G.meters_per_px, "failed")
            m = evaluate(P, G)
            m.update({"config": cfg, "sample": sid, "runtime_s": t})
            rows.append(m)
            print(f"  {cfg:10s} {sid:35s} roomIoU={m['room_iou']:.3f} wallIoU={m['wall_iou']:.3f} "
                  f"dim={m['dim_mape']:.3f} scale={m['scale_source']}")

    os.makedirs(a.out, exist_ok=True)
    keys = list(rows[0].keys())
    with open(os.path.join(a.out, f"per_sample_{a.split}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    lines = [f"| Config | " + " | ".join(h for _, h in SUMMARY_COLS) + " |",
             "|---" * (len(SUMMARY_COLS) + 1) + "|"]
    for cfg in a.configs:
        R = [r for r in rows if r["config"] == cfg]
        vals = []
        for k, _ in SUMMARY_COLS:
            v = np.array([r[k] for r in R if r[k] is not None], float)
            v = v[~np.isnan(v)]
            vals.append(f"{v.mean():.3f}" if len(v) else "-")
        lines.append(f"| {cfg} | " + " | ".join(vals) + " |")
    table = "\n".join(lines)
    src = {cfg: dict(zip(*np.unique([r["scale_source"] for r in rows if r["config"] == cfg],
                                    return_counts=True))) for cfg in a.configs}
    with open(os.path.join(a.out, f"summary_{a.split}.md"), "w") as fh:
        fh.write(f"# {a.split} split, {len(folders)} plans\n\n{table}\n\nScale sources: "
                 f"{json.dumps({k: {s: int(n) for s, n in v.items()} for k, v in src.items()})}\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
