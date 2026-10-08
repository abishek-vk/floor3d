"""Controlled study of the dimension-constrained solver (oracle annotations).

Real plans differ in how many dimension strings OCR can read, which confounds the
solver's effect. Here every plan gets the same kind of annotation: the GT room
size labels ("5.66 m x 3.83 m", present in CubiCasa SVGs), attached to our room
that best overlaps the GT room. Conditions:

  A  no annotations             scale from door widths, geometry as detected
  B  scale-only                 median(annotation / pixel size); geometry untouched
  C  solver                     joint geometry + scale refinement (ours)
  D  solver, 20% corrupted      as C, but 1 in 5 annotations is off by 25-40%;
                                reports how many corrupted ones get flagged

    python eval/solver_study.py --split test --runs results/runs/+topology --out results/
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.layout import Layout  # noqa: E402
from core.log import RunContext  # noqa: E402
from eval.eval import sample_id, samples  # noqa: E402
from eval.gt import METRIC_RE, load_sample  # noqa: E402
from eval.metrics import evaluate, match_rooms  # noqa: E402
from solve.refine import refine  # noqa: E402

COLS = ["dim_mape", "area_mape", "total_area_ape", "scale_ape", "room_iou", "wall_iou"]


def annotations(P: Layout, G: Layout) -> list[dict]:
    pairs, M, _, _ = match_rooms(P, G, (G.height, G.width))
    gr = [r for r in G.rooms if len(r.polygon) >= 3]
    pr = [r for r in P.rooms if len(r.polygon) >= 3]
    out = []
    for g, p in pairs:
        if M[g, p] < 0.5:
            continue
        for t in gr[g].dims_text:
            m = METRIC_RE.search(t)
            if m:
                a, b = (float(x.replace(",", ".")) for x in m.groups())
                out.append({"kind": "pair", "room": pr[p].id, "a": a, "b": b, "text": t.strip()})
                break
    return out


def scale_only(P: Layout, ann: list[dict]) -> float | None:
    rooms = {r.id: r for r in P.rooms}
    ratios = []
    for c in ann:
        Q = np.array(rooms[c["room"]].polygon)
        wx, wy = np.ptp(Q[:, 0]), np.ptp(Q[:, 1])
        for a, b in ((c["a"], c["b"]), (c["b"], c["a"])):
            if abs(np.log((a / wx) / (b / wy))) < 0.15:
                ratios += [a / wx, b / wy]
                break
    return float(np.median(ratios)) if ratios else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--runs", default=os.path.join(ROOT, "results", "dev", "runs", "+topology"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "dev"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    rng = np.random.default_rng(0)
    rows = {k: [] for k in "ABCD"}
    flags = {"corrupted": 0, "flagged_corrupted": 0, "clean": 0, "flagged_clean": 0}
    n_ann = []
    tmp = os.path.join(a.out, "_solver_tmp")
    for f in samples(a.split, 0, os.path.join(ROOT, "data", "cubicasa5k")):
        lp = os.path.join(a.runs, sample_id(f), "layout.json")
        if not os.path.exists(lp):
            continue
        G = load_sample(f)
        P0 = Layout.load(lp)
        ann = annotations(P0, G)
        if len(ann) < 2:
            continue
        n_ann.append(len(ann))
        rows["A"].append(evaluate(P0, G))

        PB = copy.deepcopy(P0)
        s = scale_only(PB, ann)
        if s:
            PB.meters_per_px = s
        rows["B"].append(evaluate(PB, G))

        PC = copy.deepcopy(P0)
        PC.meters_per_px = s or PC.meters_per_px  # same initial scale as B
        PC.meta["dim_constraints"] = ann
        PC = refine(PC, RunContext(tmp, debug=False))
        rows["C"].append(evaluate(PC, G))

        PD = copy.deepcopy(P0)
        bad = set(rng.choice(len(ann), size=max(1, len(ann) // 5), replace=False).tolist())
        annD = []
        for i, c in enumerate(ann):
            c = dict(c)
            if i in bad:
                k = rng.uniform(1.25, 1.4) ** rng.choice([-1, 1])
                c["a"], c["b"] = c["a"] * k, c["b"] * k
                c["text"] = f"[corrupt] {c['text']}"
            annD.append(c)
        PD.meters_per_px = scale_only(PD, annD) or PD.meters_per_px
        PD.meta["dim_constraints"] = annD
        PD = refine(PD, RunContext(tmp, debug=False))
        rows["D"].append(evaluate(PD, G))
        for r in PD.meta.get("solver", {}).get("annotations", []):
            c = r["text"].startswith("[corrupt]")
            flags["corrupted" if c else "clean"] += 1
            if r["inconsistent"]:
                flags["flagged_corrupted" if c else "flagged_clean"] += 1

    names = {"A": "A. no annotations (door-width scale)", "B": "B. annotations -> scale only",
             "C": "C. constraint solver (ours)", "D": "D. solver, 20% corrupted annotations"}
    lines = [f"Plans: {len(n_ann)} (with >=2 annotated rooms), annotations/plan: {np.mean(n_ann):.1f}", "",
             "| Condition | Dim MAPE | Area MAPE | Tot.Area APE | Scale APE | Room IoU | Wall IoU |",
             "|---|---|---|---|---|---|---|"]
    for k in "ABCD":
        v = [np.nanmean([r[c] for r in rows[k]]) for c in COLS]
        lines.append(f"| {names[k]} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
    # Constraint residuals (by the solver's sigma units) flag inconsistent annotations.
    lines += ["", f"Corrupted annotations flagged: {flags['flagged_corrupted']}/{flags['corrupted']}; "
                  f"clean annotations wrongly flagged: {flags['flagged_clean']}/{flags['clean']}"]
    txt = "\n".join(lines)
    with open(os.path.join(a.out, f"solver_study_{a.split}.md"), "w", encoding="utf8") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
