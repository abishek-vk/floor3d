"""Mode B: room video -> 3D scene (GLB/OBJ/PLY), room layout (Mode A schema) and viewer.

    python run_video.py --input room.mp4 --out out/room [--method full|baseline]
                        [--intrinsics fx fy cx cy | --fov 70] [--keyframes 150]

Outputs in --out:
    scene.glb             observed scan + generated shell as separate nodes
                          (node extras.provenance = observed | generated)
    scene.obj, *.ply      same geometry; one PLY per provenance class
    provenance.json       per shell surface: observed / opening / generated fractions
    layout.json           room layout in Mode A's schema (walls, room polygon, metres via meters_per_px)
    layout_model.glb      Mode A's extruder run on that layout (clean architectural shell)
    cameras.json          intrinsics + every keyframe pose in the export frame
    viewer.html           self-contained viewer (orbit, walk, highlight/hide generated geometry)
    debug/NN_*.png        keyframes, depth, top-down layout, shell provenance
    run_report.json       per-stage timings and every fallback that fired
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from core.log import RunContext, log, seed_everything  # noqa: E402


def export_frame_pose(R, t, T):
    """World-frame camera (x_c = R x_w + t) -> export frame, where x_e = A x_w + b."""
    A, b = T[:3, :3], T[:3, 3]
    Re = R @ A.T
    return Re, t - Re @ b


def run(video: str, out_dir: str, method: str = "full", ablation: dict | None = None, holdout: int = 0,
        intrinsics=None, fov_deg=None, n_key: int = 150, max_side: int = 640, voxel: float = 0.02,
        debug: bool = True, viewer: bool = True, cache_dir: str | None = None, progress=None) -> tuple[dict, dict]:
    seed_everything(0)
    os.makedirs(out_dir, exist_ok=True)
    ctx = RunContext(out_dir, debug=debug, progress=progress)
    t0 = time.perf_counter()
    from video_pipeline import BASELINE, FULL, reconstruct
    abl = {**(BASELINE if method == "baseline" else FULL), **(ablation or {})}
    res = reconstruct(video, ctx, abl, cache_dir=cache_dir, n_key=n_key, max_side=max_side, holdout=holdout,
                      intrinsics=intrinsics, fov_deg=fov_deg, voxel=voxel)
    from video.export import build_scene, build_viewer, export_all, to_export_frame
    rf, L = res["rf"], res["layout"]
    T = to_export_frame(rf)
    paths = {}
    with ctx.stage("export"):
        sc, info = build_scene(res["mesh"], res["generated"], T)
        paths.update(export_all(sc, out_dir))
        L.meta.update({"input": os.path.abspath(video), "method": method, "fallbacks": ctx.fallbacks,
                       "wall_height_m": rf.ceil_y - rf.floor_y})
        L.save(os.path.join(out_dir, "layout.json"))
        paths["layout"] = os.path.join(out_dir, "layout.json")
        try:  # Mode A's extruder on the video layout (both modes share one representation)
            from extrude.mesh import build_scene as build_layout_scene, export as export_layout
            ls = build_layout_scene(L, wall_height=rf.ceil_y - rf.floor_y)
            paths["layout_model"] = export_layout(ls, os.path.join(out_dir, "layout_model"))["glb"]
        except Exception as e:
            ctx.fallback("export", f"layout model not written: {e}")
        cams = []
        for i in sorted(res["R"]):
            Re, te = export_frame_pose(res["R"][i], res["T"][i], T)
            cams.append({"image": int(i), "video_frame": int(res["frames"].index[i]),
                         "split": "test" if i in res["frames"].test else "train",
                         "R": Re.tolist(), "t": te.tolist(), "C": (-Re.T @ te).tolist(), "fwd": Re[2].tolist()})
        K = res["K"]
        h, w = res["frames"].images[0].shape[:2]
        cam_json = {"K": K.tolist(), "width": w, "height": h, "frames": cams,
                    "K_estimated": not res["frames"].K_given}
        json.dump(cam_json, open(os.path.join(out_dir, "cameras.json"), "w"), indent=1)
        json.dump(res["provenance"], open(os.path.join(out_dir, "provenance.json"), "w"), indent=1)
        paths["cameras"] = os.path.join(out_dir, "cameras.json")
        paths["provenance"] = os.path.join(out_dir, "provenance.json")
    poly = rf.polygon - rf.origin_xz
    room = {"polygon": poly.tolist(), "height_m": rf.ceil_y - rf.floor_y, "ceiling_observed": rf.ceil_observed,
            "width_m": float(np.ptp(poly[:, 0])), "depth_m": float(np.ptp(poly[:, 1])),
            "area_m2": rf.stats["area_m2"]}
    if viewer:
        with ctx.stage("viewer"):
            try:
                sst = res["scale_stats"]
                vinfo = {"room": room, "provenance": res["provenance"].get("_total", {}),
                         "cameras": [{"C": c["C"], "fwd": c["fwd"]} for c in cams if c["split"] == "train"],
                         "n_frames": len(res["frames"].images), "n_registered": len(res["R"]),
                         "scale": f"{'FOV-canonical' if sst.get('canonical_crop') else 'full-frame'} metric depth prior, "
                                  f"{sst.get('n_frames', 0)} frames"}
                paths["viewer"] = build_viewer(os.path.join(out_dir, "viewer.html"), paths["glb"], vinfo)
            except Exception as e:
                ctx.fallback("viewer", f"viewer.html not written: {e}")
    rep = ctx.report()
    rep.update({"outputs": paths, "wall_clock_s": round(time.perf_counter() - t0, 2), "room": room,
                "sfm": res["rec"].stats, "scale": res["scale_stats"], "mvs_refined_frac": res["mvs_frac"],
                "snapped_frac": res["snapped_frac"], "provenance": res["provenance"].get("_total"),
                "layout": rf.stats, "ablation": abl, "export": info})
    with open(os.path.join(out_dir, "run_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    return rep, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", required=True, help="video file (mp4/mov/...) or a folder of frames")
    ap.add_argument("--out", default="out/video")
    ap.add_argument("--method", choices=["full", "baseline"], default="full")
    ap.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"),
                    help="camera intrinsics in source-video pixels (default: self-calibrated)")
    ap.add_argument("--fov", type=float, help="horizontal field of view in degrees, if known")
    ap.add_argument("--keyframes", type=int, default=150)
    ap.add_argument("--max-side", type=int, default=640, help="processing resolution (long side)")
    ap.add_argument("--voxel", type=float, default=0.02, help="TSDF voxel size in metres")
    ap.add_argument("--holdout", type=int, default=0, help="hold out every k-th keyframe as a test view")
    ap.add_argument("--no-debug", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    if not os.path.exists(a.input):
        print(f"error: input not found: {a.input}", file=sys.stderr)
        sys.exit(2)
    try:
        rep, _ = run(a.input, a.out, a.method, holdout=a.holdout, intrinsics=a.intrinsics, fov_deg=a.fov,
                     n_key=a.keyframes, max_side=a.max_side, voxel=a.voxel, debug=not a.no_debug)
    except Exception as e:  # last line of defence: report, don't stack-dump at the user
        log.error("pipeline failed: %s", e)
        traceback.print_exc()
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "run_report.json"), "w") as f:
            json.dump({"error": str(e), "trace": traceback.format_exc()}, f, indent=1)
        sys.exit(1)
    print(json.dumps({k: rep[k] for k in ("room", "provenance", "wall_clock_s", "fallbacks")}, indent=1, default=float))
    print(f"\nOpen {rep['outputs'].get('viewer', rep['outputs']['glb'])} in a browser.")


if __name__ == "__main__":
    main()
