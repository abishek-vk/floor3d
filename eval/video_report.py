"""rows.json from eval/video_eval.py -> results.md (main comparison, ablation, per scene).

    python eval/video_report.py --dir results_video/test
    python eval/video_report.py --dir results_video/test --merge results_video/test_p1 results_video/test_p2
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

COLS_MAIN = [  # (key, header, fmt, better)
    ("nvs_vertex_psnr", "PSNR ↑ (mesh)", "{:.2f}", "max"),
    ("nvs_ibr_psnr", "PSNR ↑ (IBR)", "{:.2f}", "max"),
    ("nvs_ibr_ssim", "SSIM ↑ (IBR)", "{:.3f}", "max"),
    ("nvs_ibr_lpips", "LPIPS ↓ (IBR)", "{:.3f}", "min"),
    ("sim3_chamfer_cm", "Chamfer-L1 cm ↓", "{:.2f}", "min"),
    ("sim3_fscore", "F@5cm ↑", "{:.3f}", "max"),
    ("metric_chamfer_cm", "Chamfer metric cm ↓", "{:.2f}", "min"),
    ("dim_err_layout_cm", "Dim err cm ↓ (layout)", "{:.1f}", "min"),
    ("dim_err_plane_cm", "Dim err cm ↓ (planes)", "{:.1f}", "min"),
    ("scale_err_abs", "|Scale err| % ↓", "{:.1f}", "min"),
    ("unseen_recall_all", "Unseen recall@10cm ↑", "{:.3f}", "max"),
]


def _mean(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return float(np.mean(v)) if v else None


def table(rows, configs, cols):
    head = "| Config | " + " | ".join(h for _, h, _, _ in cols) + " |"
    sep = "|---|" + "---:|" * len(cols)
    means = {c: {k: _mean([r for r in rows if r["config"] == c], k) for k, _, _, _ in cols} for c in configs}
    best = {}
    for k, _, _, b in cols:
        vals = [means[c][k] for c in configs if means[c][k] is not None]
        if vals:
            best[k] = (max if b == "max" else min)(vals)
    lines = [head, sep]
    for c in configs:
        cells = []
        for k, _, f, _ in cols:
            v = means[c][k]
            s = "–" if v is None else f.format(v)
            cells.append(f"**{s}**" if v is not None and best.get(k) is not None and abs(v - best[k]) < 1e-9 else s)
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_video/test")
    ap.add_argument("--merge", nargs="*", help="combine rows.json of these dirs into --dir first")
    a = ap.parse_args()
    if a.merge:
        rows = sum((json.load(open(os.path.join(d, "rows.json"))) for d in a.merge), [])
        os.makedirs(a.dir, exist_ok=True)
        json.dump(rows, open(os.path.join(a.dir, "rows.json"), "w"), indent=1)
    rows = json.load(open(os.path.join(a.dir, "rows.json")))
    for r in rows:
        if r.get("scale_err_pct") is not None:
            r["scale_err_abs"] = abs(r["scale_err_pct"])
    configs = list(dict.fromkeys(r["config"] for r in rows))
    scenes = list(dict.fromkeys(r["scene"] for r in rows))
    md = [f"# Mode B results ({', '.join(scenes)})", "",
          f"Replica, {len(scenes)} scene(s); 150 keyframes per video, every 8th held out for novel-view "
          f"evaluation; intrinsics self-calibrated. Means over scenes; best per column in bold.", "",
          "## Ablation (cumulative)", "", table(rows, configs, COLS_MAIN), ""]
    extra = [("ate_cm", "ATE cm", "{:.2f}", "min"), ("focal_err_pct", "Focal err %", "{:+.1f}", "min"),
             ("sim3_acc_cm", "Acc cm", "{:.2f}", "min"), ("sim3_comp_cm", "Comp cm", "{:.2f}", "min"),
             ("nvs_vertex_lpips", "LPIPS (mesh)", "{:.3f}", "min"), ("nvs_ibr_psnr_obs", "PSNR obs px", "{:.2f}", "max"),
             ("nvs_ibr_psnr_gen", "PSNR gen px", "{:.2f}", "max"), ("nvs_vertex_hole", "Empty px", "{:.3f}", "min"),
             ("unseen_recall_obs", "Unseen recall (obs only)", "{:.3f}", "max"),
             ("gen_acc_cm", "Generated acc cm", "{:.1f}", "min"), ("gen_precision10", "Generated within 10cm", "{:.3f}", "max"),
             ("gen_frac_shell", "Shell generated", "{:.3f}", "min")]
    md += ["## Secondary metrics", "", table(rows, configs, extra), ""]
    md += ["## Per scene (full vs baseline)", ""]
    keys = ["nvs_ibr_psnr", "nvs_vertex_psnr", "sim3_chamfer_cm", "dim_err_layout_cm", "scale_err_pct",
            "gt_width_m", "layout_width_m", "gt_depth_m", "layout_depth_m", "gt_height_m", "layout_height_m"]
    md.append("| Scene | Config | " + " | ".join(keys) + " |")
    md.append("|---|---|" + "---:|" * len(keys))
    for s in scenes:
        for c in configs:
            r = next((r for r in rows if r["scene"] == s and r["config"] == c), None)
            if r is None or c not in ("baseline", "full", "full+depth_ba"):
                continue
            md.append(f"| {s} | {c} | " + " | ".join("–" if r.get(k) is None else f"{r[k]:.2f}" for k in keys) + " |")
    open(os.path.join(a.dir, "results.md"), "w", encoding="utf8").write("\n".join(md) + "\n")
    print(f"wrote {os.path.join(a.dir, 'results.md')}")


if __name__ == "__main__":
    main()
