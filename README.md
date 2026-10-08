# floorplan3d — floor plan image → metric 3D model

Hackathon track HNX26EPS06, **Mode A**: architectural floor plan (PNG/JPG/PDF) → structured
layout (walls, doors, windows, rooms, metric dimensions) → extruded 3D model (GLB/OBJ/JSON)
→ browser viewer with orbit + first-person walk.

Results vs the baseline: [`results/results.md`](results/results.md) · Write-up: [`WRITEUP.md`](WRITEUP.md) ·
Demo script: [`DEMO.md`](DEMO.md)

## Web UI (upload a plan or pick a demo)

```bash
python app/server.py          # or double-click start_ui.bat on Windows
# open http://127.0.0.1:8000
```

Pick one of six demo blueprints, or upload your own **PNG, JPG/JPEG or PDF** (first page;
max 25 MB). The page shows progress per stage, then the 3D model (orbit/walk/2D overlay),
a room table with metric sizes and the scale source, every analysis stage image, and
downloads (GLB, OBJ+MTL, layout JSON, standalone viewer). Files are checked by content, not
just extension. Jobs are stored in `app_data/jobs/`.

## One-command demo

```bash
python run.py --input data/cubicasa5k/high_quality_architectural/5927/F1_scaled.png --out out/demo
# open out/demo/viewer.html  (self-contained; double-click works, needs internet for three.js)
```

Outputs in `--out`:

| File | What |
|---|---|
| `model.glb` | primary 3D model; named nodes `walls`, `railings`, `door_<i>`, `window_<i>`, `sill_<i>`, `floor_<i>_<type>` |
| `model.obj` + `material.mtl` | same scene as OBJ |
| `layout.json` | structured layout: walls (centreline + thickness), openings (host wall, offset, width), rooms (polygon, type, OCR label), `meters_per_px`, solver report |
| `viewer.html` | single-file viewer: orbit, walk (WASD + mouse, wall collision), room labels with metric size, 2D plan overlay toggle, ceiling toggle |
| `debug/NN_*.png` | every stage: input, binary, segmentation, classical walls, fused walls, raw segments, barrier, final layout |
| `run_report.json` | per-stage timings and **every fallback that fired** |

Options: `--wall-height 2.7`, `--method baseline|full`, `--tta` (4-rotation test-time
augmentation), `--ceiling`, `--no-debug`, `-v`.

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate        # (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
git clone --depth 1 https://github.com/CubiCasa/CubiCasa5k third_party/CubiCasa5k
gdown 1gRB7ez1e4H7a9Y09lLqRuna0luZO5VRK -O data/weights/model_best_val_loss_var.pkl
python data/fetch_cubicasa.py --split test --n 30      # held-out eval plans
python data/fetch_cubicasa.py --split val  --n 10      # development plans
```

`data/fetch_cubicasa.py` pulls individual plans out of the 5.4 GB Zenodo zip with HTTP range
requests (~10 s/plan), so the full archive is never downloaded.

## Evaluation

```bash
python eval/eval.py --split test --out results            # baseline + all ablations, cached per plan
python eval/solver_study.py --split test --runs results/runs/+topology --out results
python eval/report.py --split test --dir results          # -> results/results.md + figures
```

Ablation rows: `baseline | +fusion | +topology | +ocr_scale | full`. Flags live in
`eval/eval.py:CONFIGS` and are read by `pipeline.full_layout`.

**Hand-annotated plans with measured dimensions.** Put each image in `data/hand/` with a JSON
next to it: one entry per room with any pixel inside the room and the measured width/length
(format in [`eval/handgt.py`](eval/handgt.py)). Takes ~2 minutes per plan. Then:

```bash
python eval/handgt.py --dir data/hand --out results/hand   # baseline vs ours: metric dim/area error, completeness
```

## Pipeline

```
ingest/      load PNG/JPG/PDF (pypdfium2), deskew (Hough), Otsu binarise, denoise scans, normalise to 800-1800 px
detect/      cubicasa.py   pretrained CubiCasa5K multi-task net (rooms, icons, junction heatmaps)
             classical.py  thick-stroke wall extraction (morphological opening) + fusion with learned walls
vectorize/   walls.py      directional decomposition -> centreline segments + thickness (diagonals kept)
             topology.py   collinear merge across door gaps, junction snapping, dangling-end extension,
                           pruning of walls that bound no room
             openings.py   door/window detection + host-wall attachment (offset, width)
             rooms.py      rooms = enclosed free space, split at reflex corners where the net sees different rooms;
                           unenclosed spaces (balconies, open areas) filled from the net's room mask
             naive.py      the BASELINE polygonisation
scale/       ocr.py        RapidOCR (PaddleOCR models, ONNX) + robust metric-scale voting
solve/       refine.py     dimension-constrained robust least squares (the research contribution)
extrude/     mesh.py       banded extrusion with door/window cut-outs, floors, materials -> GLB/OBJ
viewer/      template.html three.js viewer; build.py embeds GLB + layout + plan into viewer.html
eval/        gt.py (SVG -> layout), metrics.py, eval.py, solver_study.py, report.py, render.py, compare.py
pipeline.py  our method; run.py the CLI
```

## Choices and deviations (noted as the brief asked)

- **Python 3.14, not 3.11.** Windows Application Control on the dev laptop blocks the unsigned
  Python 3.11 that uv downloads; the venv uses the installed, signed 3.14. The code uses nothing
  newer than 3.11 syntax.
- **OCR = RapidOCR** (PaddleOCR detection/recognition models on onnxruntime): pip-only, no
  Tesseract install, CPU-fast (~4.5 s per pass, two passes for vertical text).
- **PDF = pypdfium2** instead of pdf2image (no Poppler dependency). First page is used.
- **CPU inference** at <= 1024 px long side; whole pipeline ~10-20 s per plan on a laptop CPU.
- **Dev/test discipline.** Every threshold was tuned on the CubiCasa *val* split (10 plans);
  the 30 *test* plans were only used for the reported numbers. One disclosure: while
  smoke-testing the hand-annotation evaluator on test plan 2090, we found an OCR parser bug
  (the appliance code `APK4,5` read as a 4.5 m² room area and trusted on its own). The fix
  (area numbers must be space-separated; a lone bare number cannot set the scale) is generic
  and was made before the final test run; no thresholds were changed.
- **Scale acceptance (chosen on val).** OCR scale needs >= 2 independent agreeing annotations
  (or one `a x b` size pair) and must agree with the door-width estimate within 28%; on val, OCR
  beat the door cue on 10/13 plans and every gross OCR failure violated that gate.
- **Door-width fallback = 0.78 m**, the median drawn door opening on the val split (the
  textbook 0.85 m overestimates drawn openings). The baseline uses the same helper, so the
  comparison isolates our contributions.
- **CubiCasa GT is 1 cm/px.** Every test/val plan with room size labels gives exactly
  0.0100 m/px in the SVG frame; unlabelled plans use that convention for GT. Our pipeline does
  **not** use this prior: the last-resort default (`0.0125 m/px`) is deliberately not tuned to
  it, and `eval` reports which scale source each plan used.
- **Orthogonality** of horizontal/vertical walls is exact by parametrisation in the solver
  rather than a soft term; diagonal walls are detected and kept but held fixed in the solve.
- **Openings are cut without CSG**: walls are extruded in three height bands (below sill,
  sill-to-head, above head) with different 2D footprints. Robust and exact for straight walls.
- Defaults: wall height 2.7 m, doors 0-2.1 m, windows 0.9-2.1 m, floors 2 cm, door frames 5 cm.

## Never crash

Each stage degrades instead of failing, and records it in `run_report.json -> fallbacks`:
deskew applied, OCR missing or too little support (→ door-width scale → default prior),
openings without a host wall (dropped), no enclosed rooms (→ learned room components),
solver failure (→ unrefined layout), empty scene (→ ground plate), viewer build failure.
Tested on a PDF, a 3°-skewed noisy JPEG scan, a blank page and a 160 px thumbnail.

## Licences

CubiCasa5K data and model: CC BY-NC 4.0 (Kalervo et al., 2019). three.js: MIT.
