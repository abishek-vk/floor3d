"""Floor plan image -> structured layout -> 3D model (GLB/OBJ/JSON).

    python run.py --input plan.png --out out/ [--wall-height 2.7] [--method full|baseline]
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

from core.layout import Layout  # noqa: E402
from core.log import RunContext, log, seed_everything  # noqa: E402
from core import viz  # noqa: E402


def run(input_path: str, out_dir: str, method: str = "full", wall_height: float = 2.7,
        tta: bool = False, debug: bool = True, ceiling: bool = False,
        ablation: dict | None = None, viewer: bool = True, progress=None) -> dict:
    seed_everything(0)
    os.makedirs(out_dir, exist_ok=True)
    ctx = RunContext(out_dir, debug=debug, progress=progress)
    t0 = time.perf_counter()

    from ingest.load import ingest
    from detect import cubicasa
    from vectorize.naive import naive_layout
    from extrude.mesh import build_scene, export

    with ctx.stage("ingest"):
        ing = ingest(input_path, ctx)

    with ctx.stage("detect"):
        seg = cubicasa.predict(ing.rgb, tta=tta)
        ctx.save_img("seg_overlay", viz.seg_overlay(ing.rgb, seg.room_label, seg.icon_label))

    with ctx.stage("vectorize"):
        if method == "baseline":
            L = naive_layout(seg, ctx)
        else:
            from pipeline import full_layout  # our method
            L = full_layout(ing, seg, ctx, ablation or {})
        ctx.save_img("layout", viz.draw_layout(ing.rgb, L))

    # Back to input-image pixel frame so layouts are comparable to annotations.
    if ing.scale != 1.0:
        L = L.scaled(1.0 / ing.scale)
    L.meta.update({"input": os.path.abspath(input_path), "method": method,
                   "skew_deg": ing.skew_deg, "wall_height_m": wall_height,
                   "fallbacks": ctx.fallbacks})

    with ctx.stage("extrude"):
        scene = build_scene(L, wall_height=wall_height, ceiling=ceiling)
        if len(scene.geometry) == 0:
            import trimesh
            ctx.fallback("extrude", "no structure detected; exporting an empty ground plate")
            w_m, h_m = L.width * L.meters_per_px, L.height * L.meters_per_px
            plate = trimesh.creation.box([w_m, 0.02, h_m])
            plate.apply_translation([w_m / 2, -0.01, h_m / 2])
            scene.add_geometry(plate, node_name="ground", geom_name="ground")
        paths = export(scene, os.path.join(out_dir, "model"))
    L.save(os.path.join(out_dir, "layout.json"))
    paths["layout"] = os.path.join(out_dir, "layout.json")
    if viewer:
        with ctx.stage("viewer"):
            try:
                from ingest.load import read_any
                from viewer.build import build_viewer
                paths["viewer"] = build_viewer(os.path.join(out_dir, "viewer.html"), paths["glb"],
                                               L.to_dict(), read_any(input_path))
            except Exception as e:
                ctx.fallback("viewer", f"viewer.html not written: {e}")
    rep = ctx.report()
    rep.update({"outputs": paths, "wall_clock_s": round(time.perf_counter() - t0, 2)})
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", required=True, help="PNG/JPG/PDF floor plan")
    ap.add_argument("--out", default="out")
    ap.add_argument("--wall-height", type=float, default=2.7)
    ap.add_argument("--method", choices=["full", "baseline"], default="full")
    ap.add_argument("--tta", action="store_true", help="4-rotation test-time augmentation (slower)")
    ap.add_argument("--ceiling", action="store_true")
    ap.add_argument("--no-debug", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    if not os.path.isfile(a.input):
        print(f"error: input file not found: {a.input}", file=sys.stderr)
        sys.exit(2)
    try:
        rep = run(a.input, a.out, a.method, a.wall_height, a.tta, not a.no_debug, a.ceiling)
    except Exception as e:  # last line of defence: report, don't stack-dump at the user
        log.error("pipeline failed: %s", e)
        traceback.print_exc()
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "run_report.json"), "w") as f:
            json.dump({"error": str(e), "trace": traceback.format_exc()}, f, indent=1)
        sys.exit(1)
    print(json.dumps(rep, indent=1))
    print(f"\nOpen {rep['outputs'].get('viewer', rep['outputs']['glb'])} in a browser.")


if __name__ == "__main__":
    main()
