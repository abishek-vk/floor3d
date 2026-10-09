# floorplan3d

**Turn a floor plan image or a room video into a measured, walkable 3D model.**

floorplan3d reads an architectural floor plan (PNG, JPG or PDF) and rebuilds it as structured,
metric geometry: walls with centrelines and thickness, doors and windows attached to their host
walls, and rooms as typed polygons. It then extrudes that into a 3D model. Real-world scale comes
from the dimensions written on the drawing. A second mode turns a handheld video of a room into a
metric 3D scene and labels every surface as either observed or generated.

Results open in a self-contained browser viewer with orbit and first-person walk. They export to
GLB, OBJ and JSON for Blender, three.js, game engines and BIM workflows.

---

## Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Web app](#web-app)
  - [Command line: floor plans](#command-line-floor-plans)
  - [Command line: room videos](#command-line-room-videos)
- [Outputs](#outputs)
- [Project structure](#project-structure)
- [Robustness and fallbacks](#robustness-and-fallbacks)
- [Evaluation](#evaluation)
- [Design decisions](#design-decisions)
- [Limitations](#limitations)
- [Further documentation](#further-documentation)
- [Licences and acknowledgements](#licences-and-acknowledgements)

---

## Features

**Floor plan to 3D (Mode A)**
- Accepts PNG, JPG/JPEG and PDF (first page) up to 25 MB. Files are validated by content, not just extension.
- Handles clean CAD exports, colour plans, scans, hollow/outlined walls, hand-drafted pencil drawings and multi-storey sheets.
- Detects walls, doors, windows, rooms (with type and OCR'd label) and fixtures/furniture.
- **Metric scale** is read from dimension strings (`3600`, `3.6 m`, `12'6"`, `3.4 x 4.1 m`) and room areas (`12.5 m²`). If the drawing has none, it falls back to door widths.
- **Dimension-constrained refinement**: a robust least-squares solver adjusts the walls so that they agree with the written dimensions. Annotations that conflict with the rest of the drawing are flagged.
- Extrudes walls to a configurable height and cuts exact door and window openings.
- Processes a plan in roughly 10–30 seconds on a laptop CPU.

**Room video to 3D (Mode B)**
- Accepts MP4/MOV up to 300 MB.
- Recovers camera motion with built-in structure-from-motion, including focal-length self-calibration.
- Combines metric monocular depth with multi-view stereo and fuses the result into a surface.
- Fits floor, ceiling and walls, and outputs the room in the same layout format as Mode A.
- Completes unseen surfaces, and marks every generated surface explicitly (per-surface provenance, separate mesh nodes, and a viewer toggle).

**Viewing and export**
- A web app with demo blueprints, drag-and-drop upload, live progress, room measurements, every intermediate analysis image, and downloads.
- A single-file `viewer.html` with orbit, first-person walk (WASD + mouse, wall collision), metric room labels, a 2D plan overlay and a ceiling toggle.
- Open formats: GLB, OBJ + MTL, PLY and a documented JSON layout.

---

## How it works

### Mode A: floor plan

```
 image / PDF ─▶ Ingest ─▶ Detect ─▶ Vectorise ─▶ Measure ─▶ Refine ─▶ Extrude ─▶ GLB · OBJ · JSON · viewer
```

1. **Ingest.** The input is loaded (PDFs via pypdfium2), deskewed with a Hough transform, binarised with Otsu's method, denoised, and normalised to 800–1800 px.
2. **Hybrid detection.** The pretrained CubiCasa5K multi-task network predicts room, wall and icon probabilities. A classical extractor applies morphological opening just below the thinnest wall, which keeps only thick strokes with pixel-exact edges. The final walls are ink walls the network agrees with, plus confident learned walls the ink test misses, such as hollow walls.
3. **Vectorisation.** Directional decomposition turns the wall mask into centreline segments with thickness, and diagonals are kept. Topology cleanup then:
   - merges collinear pieces across door gaps
   - snaps endpoints to junctions
   - extends dangling ends
   - prunes walls that bound no room
4. **Rooms and openings.** Rooms are the enclosed free space between walls, split only along rays from reflex corners where the network sees different room types. Spaces walls cannot enclose, such as balconies and open areas, come from the network's room mask. Doors and windows are attached to a host wall with an offset and a width. Furniture symbols become parametric proxies.
5. **Metric scale by voting.** RapidOCR reads dimension strings and areas. Each reading proposes metres-per-pixel hypotheses, and the scale is the peak of a kernel density in log space. The OCR scale is accepted only if:
   - at least two independent annotations agree (or there is a single `a × b` size pair), and
   - it agrees with the door-width cue within 28%.

   Otherwise the door-width estimate is used, and failing that, a default prior.
6. **Dimension-constrained refinement.** The unknowns are each axis-aligned wall's position, extent and thickness, plus a global log-scale. They are solved with `scipy.optimize.least_squares` under a Huber loss, with these soft constraints:
   - data fidelity
   - junction coincidence
   - collinearity
   - thickness consistency
   - OCR spans
   - room sizes and areas
   - a scale prior

   Orthogonality is exact by parametrisation. The solver reports every residual and flags annotations that stay inconsistent (> 3σ).
7. **Extrusion.** Walls are extruded in three height bands (below sill, sill to head, above head), so door and window openings are cut exactly without CSG. The result is separate named meshes with materials, exported as GLB and OBJ, together with an embedded three.js viewer.

### Mode B: room video

```
 video ─▶ Keyframes ─▶ SfM ─▶ Metric depth + MVS ─▶ TSDF fusion ─▶ Layout ─▶ Completion ─▶ scene + provenance
```

1. **Keyframes and SfM.** The pipeline picks the sharpest frame per time bin, then matches with:
   - RootSIFT features
   - sequential and bag-of-words loop pairs
   - fundamental-matrix verification

   Focal length is self-calibrated with the Mendonça–Cipolla criterion. Cameras are then registered with incremental PnP, triangulated, and refined with robust sparse bundle adjustment. A temporal-jump check rejects frames that "teleport".
2. **Metric scale.** Depth Anything V2 (metric) runs on centre crops resampled to its training field of view, which removes the scale bias it has on wide-angle cameras.
3. **Dense depth.** Monocular depth is aligned per frame to the SfM points with a smooth scale field. A plane sweep then runs in a narrow band around that prior (multi-view NCC). Where stereo is confident, it overrides the monocular depth. A multi-view consistency filter follows.
4. **Fusion and layout.** Depth is fused with TSDF (torch) and meshed with marching cubes. RANSAC planes give a Manhattan frame, floor and ceiling heights, and a room footprint chosen on a cell complex of wall planes. The footprint is written as a Mode A layout.
5. **Provenance-aware completion.** Each shell surface is rasterised into 3 cm texels. Each texel is labelled **observed**, **opening** (the camera saw through it, so it is never filled) or **generated** (never seen). Only generated texels are filled, and they are exported as separate geometry tagged `extras.provenance = "generated"`.

---

## Requirements

| | |
|---|---|
| OS | Windows, Linux or macOS |
| Python | 3.11 or newer (developed on 3.14) |
| Hardware | CPU only; no GPU required |
| Disk | About 1 GB for models and dependencies; Mode B evaluation data is about 0.5 GB per Replica scene |
| Network | Needed at install time. The web app and viewer load three.js (and the web app its fonts) from a CDN. |

Main dependencies (pinned in [`requirements.txt`](requirements.txt)): PyTorch (CPU), OpenCV, NumPy,
SciPy, Shapely, trimesh, manifold3d, scikit-image, pypdfium2, RapidOCR + onnxruntime. Mode B also
needs onnx and lpips.

---

## Installation

```bash
# 1. Virtual environment
python -m venv .venv
.venv/Scripts/activate              # Linux/macOS: source .venv/bin/activate

# 2. Python dependencies (CPU build of PyTorch)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 3. CubiCasa5K model code and weights (required for Mode A)
git clone --depth 1 https://github.com/CubiCasa/CubiCasa5k third_party/CubiCasa5k
gdown 1gRB7ez1e4H7a9Y09lLqRuna0luZO5VRK -O data/weights/model_best_val_loss_var.pkl
```

**Mode B (optional)**: Depth Anything V2 code and metric weights. The ONNX graphs are exported automatically on first use.

```bash
git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2 third_party/Depth-Anything-V2
curl -L -o data/weights/depth_anything_v2_metric_hypersim_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth
```

**Evaluation data (optional)**

```bash
python data/fetch_cubicasa.py --split test --n 30     # held-out floor plans
python data/fetch_cubicasa.py --split val  --n 10     # development floor plans
python data/fetch_replica.py --scenes office0 room0 room1 room2 office2 office3   # room videos
```

`fetch_cubicasa.py` downloads individual plans from the 5.4 GB CubiCasa5K archive with HTTP range
requests (about 10 s per plan), so the full archive is never downloaded.

`third_party/`, `data/weights/`, `out/` and `app_data/` are git-ignored.

---

## Usage

### Web app

```bash
python app/server.py                       # or double-click start_ui.bat on Windows
python app/server.py --host 0.0.0.0 --port 8080   # optional: different host/port
```

Open http://127.0.0.1:8000. The page has a product overview and a **Studio** section where you can:

1. Pick one of six demo blueprints, or drag and drop your own plan or room video.
2. Set the wall height (2–6 m, default 2.7 m) and start the analysis. Progress is shown per stage.
3. Explore the result in four tabs:
   - **3D model:** orbit, walk and 2D overlay
   - **Rooms & measurements:** metric sizes and where the scale came from
   - **Analysis stages:** every intermediate image
   - **Downloads**

Processing runs locally. Jobs and their outputs are stored in `app_data/jobs/`.

### Command line: floor plans

```bash
python run.py --input plan.png --out out/plan
# then open out/plan/viewer.html
```

| Option | Default | Description |
|---|---|---|
| `--input` | required | PNG, JPG or PDF floor plan |
| `--out` | `out` | Output directory |
| `--wall-height` | `2.7` | Wall height in metres |
| `--method` | `full` | `full` (this method) or `baseline` (reference polygonisation) |
| `--tta` | off | 4-rotation test-time augmentation (slower, sometimes more robust) |
| `--ceiling` | off | Add ceilings to the model |
| `--no-debug` | off | Skip writing intermediate stage images |
| `-v` | off | Verbose logging |

### Command line: room videos

```bash
python run_video.py --input room.mp4 --out out/room             # camera self-calibrated
python run_video.py --input room.mp4 --out out/room --fov 70    # known horizontal FOV
# then open out/room/viewer.html
```

| Option | Default | Description |
|---|---|---|
| `--input` | required | Video file (MP4/MOV/…) or a folder of frames |
| `--out` | `out/video` | Output directory |
| `--method` | `full` | `full` or `baseline` |
| `--intrinsics FX FY CX CY` | estimated | Known camera intrinsics in pixels |
| `--fov` | estimated | Horizontal field of view in degrees |
| `--keyframes` | `150` | Number of keyframes to extract |
| `--max-side` | `640` | Processing resolution (long side, px) |
| `--voxel` | `0.02` | TSDF voxel size in metres |
| `--holdout` | `0` | Hold out every k-th keyframe as a test view |
| `--no-debug`, `-v` | off | As above |

A room video typically takes 5–15 minutes on a laptop CPU.

---

## Outputs

### Floor plan (`run.py`)

| File | Contents |
|---|---|
| `model.glb` | Primary 3D model. Named nodes: `walls`, `railings`, `door_<i>`, `window_<i>`, `sill_<i>`, `floor_<i>_<type>` |
| `model.obj` + `material.mtl` | The same scene as OBJ |
| `layout.json` | Structured layout: walls (centreline + thickness), openings (host wall, offset, width), rooms (polygon, type, OCR label), `meters_per_px`, `scale_source`, solver report |
| `viewer.html` | Self-contained viewer (open by double-clicking) |
| `debug/NN_*.png` | Every stage: input, binarised ink, network segmentation, classical walls, fused walls, wall centrelines, closed boundaries, final layout |
| `run_report.json` | Per-stage timings and every fallback that fired |

### Room video (`run_video.py`)

| File | Contents |
|---|---|
| `scene.glb` | Observed scan (`observed_scan`) and generated shell (`generated_shell`) as separate nodes, each carrying `extras.provenance` |
| `scene.obj`, `observed_scan.ply`, `generated_shell.ply` | The same geometry; one PLY per provenance class |
| `provenance.json` | Per shell surface (floor, ceiling, each wall): observed / opening / generated fractions |
| `layout.json`, `layout_model.glb` | The room as a Mode A layout (1 px = 1 cm) and its clean extruded model, in the same frame as `scene.glb` |
| `cameras.json` | Camera intrinsics and every keyframe pose |
| `viewer.html` | Orbit / walk viewer; the **Generated** button cycles shown → highlighted → hidden |
| `run_report.json` | Timings, room dimensions, provenance summary and fallbacks |

---

## Project structure

```
app/                 Web app: server.py (stdlib HTTP server + job queue), static/index.html, demos/
ingest/              Loading (PNG/JPG/PDF), deskew, binarisation, denoising, normalisation
detect/              cubicasa.py: CubiCasa5K network · classical.py: morphological walls + fusion
vectorize/           walls.py: centrelines + thickness · topology.py: merge/snap/extend/prune
                     openings.py: doors/windows on host walls · rooms.py: room polygons
                     naive.py: baseline polygonisation
scale/               ocr.py: RapidOCR reading + robust metric-scale voting
solve/               refine.py: dimension-constrained robust least squares
extrude/             mesh.py: banded extrusion, openings, floors, materials → GLB/OBJ
viewer/              template.html (three.js) · build.py embeds model + layout + plan into viewer.html
core/                Shared Layout schema, run context / fallback log, visualisation helpers
video/               Mode B: frames, sfm, depth, mvs, fuse, layout, complete, render, export
eval/                Ground truth, metrics, ablations, solver study, reports, hand-annotation evaluator
data/                Dataset fetchers, robustness test inputs, model weights (git-ignored)
pipeline.py          Mode A pipeline           run.py         Mode A CLI
video_pipeline.py    Mode B pipeline           run_video.py   Mode B CLI
start_ui.bat         Windows launcher for the web app
```

---

## Robustness and fallbacks

No stage is allowed to crash the run. Each stage degrades gracefully and records what it did in
`run_report.json → fallbacks`:

| Situation | Behaviour |
|---|---|
| Skewed scan | Deskew applied and logged |
| OCR unavailable or too little support | Door-width scale (0.78 m), then a default prior |
| Opening with no host wall | Opening dropped |
| No enclosed rooms | Rooms taken from the network's room components |
| Solver failure | Unrefined layout used |
| Empty scene | Ground plate exported |
| Viewer build failure | Model files still written |

The web app reports the scale source in plain language and warns when sizes are estimated rather
than measured. The pipeline is tested on a PDF, a 3°-skewed noisy JPEG scan, a blank page and a 160 px
thumbnail (`data/robust/`).

---

## Evaluation

### Floor plans (CubiCasa5K)

```bash
python eval/eval.py --split test --out results                                   # baseline + ablations, cached per plan
python eval/solver_study.py --split test --runs results/runs/+topology --out results
python eval/report.py --split test --dir results                                 # → results/results.md + figures
```

Ablation configurations: `baseline | +fusion | +topology | +ocr_scale | full` (defined in
`eval/eval.py:CONFIGS`, read by `pipeline.full_layout`).

**Hand-annotated plans with measured dimensions.** Put each image in `data/hand/`, next to a JSON
file with one entry per room: any pixel inside the room, plus its measured width and length (format in
[`eval/handgt.py`](eval/handgt.py), about 2 minutes per plan). Then run:

```bash
python eval/handgt.py --dir data/hand --out results/hand    # metric dimension/area error, completeness
```

**Protocol.** All thresholds were tuned on the 10-plan validation split. The 30 test plans are used
only for reported numbers.

### Room videos (Replica)

```bash
python eval/video_eval.py --scenes room1 room2 office2 office3 --out results_video/test
python eval/video_report.py --dir results_video/test
```

`--partial 0.5` uses only the first half of each video, which leaves large parts of the room unseen. The
metrics are:
- novel-view PSNR/SSIM/LPIPS on held-out frames
- Chamfer distance and F-score
- room-dimension error
- scale error
- recall on unseen regions

On the development scene `office0`, the full method improves on the baseline as follows:

| Metric | Baseline | Full |
|---|---:|---:|
| Novel-view PSNR (image-based) | 25.2 dB | 32.1 dB |
| Chamfer distance at its own metric scale | 55 cm | 9.6 cm |
| Recall of unseen regions within 10 cm | 0.24 | 0.85 |

These are development-scene numbers; held-out test results are not yet available. See
[`WRITEUP_MODE_B.md`](WRITEUP_MODE_B.md) for the full table and caveats.

---

## Design decisions

- **OCR: RapidOCR** (PaddleOCR detection and recognition models on onnxruntime). It installs with pip only, needs no Tesseract, and is fast on CPU (about 4.5 s per pass; two passes to catch vertical text).
- **PDF: pypdfium2** rather than pdf2image, so there is no Poppler dependency. The first page is used.
- **CPU inference** at ≤ 1024 px on the long side keeps a full run at about 10–30 s per plan on a laptop.
- **Strict scale acceptance.** OCR scale needs at least two independent agreeing annotations (or one `a × b` pair) and must agree with the door-width estimate within 28%. On validation, OCR beat the door cue on 10 of 13 plans, and every gross OCR failure violated this gate.
- **Door-width fallback of 0.78 m**: the median drawn door opening on the validation split. The textbook 0.85 m overestimates drawn openings.
- **Default scale prior of 0.0125 m/px** is deliberately not tuned to the 1 cm/px convention of the CubiCasa ground truth. The evaluation reports which scale source each plan used.
- **Orthogonality by parametrisation.** Horizontal and vertical walls are exactly orthogonal in the solver. Diagonal walls are kept but held fixed during refinement.
- **Openings without CSG.** Banded extrusion gives exact, robust cut-outs for straight walls.
- **Model defaults:** wall height 2.7 m, doors 0–2.1 m, windows 0.9–2.1 m, floor slab 2 cm, door frames 5 cm.
- **Self-contained Mode B stack.** SfM, TSDF fusion and rendering are implemented on NumPy, SciPy, OpenCV and PyTorch rather than COLMAP or Open3D. This keeps installation pip-only and works on machines where those binaries cannot be installed.

---

## Limitations

- Floor plans should be top-down drawings. Photos of plans taken at an angle are not rectified.
- HEIC, TIFF, DWG and SVG inputs are not supported; export to PNG or PDF first.
- Without dimension text, sizes come from door widths and are typically within 10–15%.
- Curved walls are approximated. Diagonal walls are not refined by the solver.
- Mode B accuracy depends on structure-from-motion. Weakly connected videos can drift in scale, and generated surfaces are plausible planar fills, not measurements.
- The web app and viewer need internet access to load three.js from a CDN.

---

## Further documentation

| Document | Contents |
|---|---|
| [`WRITEUP.md`](WRITEUP.md) | Mode A method in detail |
| [`WRITEUP_MODE_B.md`](WRITEUP_MODE_B.md) | Mode B method, evaluation protocol, results and caveats |

---

## Licences and acknowledgements

- **CubiCasa5K** dataset and model: CC BY-NC 4.0 (Kalervo et al., 2019). Because of this licence, Mode A's detection model is for non-commercial use.
- **Depth Anything V2** Metric-Hypersim Small: Apache-2.0 (Yang et al., 2024).
- **Replica** dataset: research licence (Straub et al., 2019); NICE-SLAM renderings.
- **three.js**: MIT. **LPIPS**: BSD-2-Clause. **RapidOCR / PaddleOCR models**: Apache-2.0.
