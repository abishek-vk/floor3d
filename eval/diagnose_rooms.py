"""Why are GT rooms missed? Categorise every GT room not matched at IoU >= 0.5.

    python eval/diagnose_rooms.py --runs results/dev/runs/+topology --split val
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.layout import Layout  # noqa: E402
from eval.eval import sample_id, samples  # noqa: E402
from eval.gt import load_sample  # noqa: E402
from eval.metrics import match_rooms  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    cats = collections.Counter()
    by_type = collections.Counter()
    tot = 0
    for f in samples(a.split, 0, os.path.join(ROOT, "data", "cubicasa5k")):
        lp = os.path.join(a.runs, sample_id(f), "layout.json")
        if not os.path.exists(lp):
            continue
        G, P = load_sample(f), Layout.load(lp)
        pairs, M, pm, gm = match_rooms(P, G, (G.height, G.width))
        matched = {g for g, p in pairs if M[g, p] >= 0.5}
        gr = [r for r in G.rooms if len(r.polygon) >= 3]
        for g in range(len(gm)):
            tot += 1
            if g in matched:
                continue
            ga = gm[g].sum()
            covered = 0.0
            biggest = None
            for j, p in enumerate(pm):
                inter = np.count_nonzero(gm[g] & p)
                if inter == 0:
                    continue
                covered += inter
                if biggest is None or inter > biggest[0]:
                    biggest = (inter, j)
            cov = covered / ga
            if biggest is None or cov < 0.3:
                c = "missing"
            else:
                inter, j = biggest
                pa = pm[j].sum()
                if inter / ga > 0.6 and pa > 1.6 * ga:
                    c = "merged"
                elif inter / ga < 0.6 and cov > 0.6:
                    c = "split"
                else:
                    c = "shifted"
            cats[c] += 1
            by_type[(c, gr[g].type)] += 1
    print(f"GT rooms: {tot}, missed at IoU .5: {sum(cats.values())}")
    for c, n in cats.most_common():
        print(f"  {c:8s} {n}")
    print("by type:", by_type.most_common(12))


if __name__ == "__main__":
    main()
