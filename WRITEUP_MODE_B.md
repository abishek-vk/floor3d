# Room video → honest, metric 3D scene (Mode B)

**Track HNX26EPS06, Mode B** · a handheld room video → posed keyframes → metric dense geometry →
room layout → completed room with every generated surface labelled → GLB + browser viewer.
Runs on a laptop CPU; no COLMAP/Open3D (blocked by Windows Application Control), so SfM, TSDF
fusion and rendering are implemented on numpy/scipy/OpenCV/torch.

## Problem
A video of a room is easy to take, but turning it into a usable 3D scene has three gaps:
monocular reconstructions have **no metric scale**, monocular depth is **soft** (good layout,
poor shape), and a video never sees the whole room. Closing the last gap by "filling in" is
exactly where systems silently hallucinate geometry. We want a scene that is metric, sharp
where it was observed, complete where it was not, and **always says which is which**.

## Baseline
SfM poses (same SfM as ours) + per-frame median-scaled monocular metric depth (Depth Anything V2,
scale read naively from the full frame) + TSDF fusion, vertex-coloured mesh. No layout reasoning,
no completion. Both methods are rendered and evaluated identically.

## Method
1. **Keyframes and SfM.** Sharpest frame per time bin; RootSIFT; sequential + bag-of-words loop
   pairs; fundamental-matrix verification. **Focal self-calibration** by the Mendonça–Cipolla
   criterion over all verified pairs (E = KᵀFK must have two equal singular values), then
   incremental PnP registration, triangulation and robust sparse BA. A video prior rejects
   registrations that "teleport" away from their temporal neighbours.
2. **Metric scale from FOV-canonical crops (ours).** A metric depth network trained at one FOV
   misreads other cameras (a 90° camera makes everything look ~30% farther). We run it on
   centre crops resampled to its training FOV (60°) and take the median depth ratio against SfM
   points inside the crop.
3. **Dense depth (ours).** Monocular depth aligned per frame with a smooth log-scale field fit to
   SfM points; then a **plane sweep in a narrow band around that prior** (multi-view NCC). Where
   stereo is confident it wins, and its dense depths re-anchor the monocular field elsewhere.
   Multi-view consistency filtering, then TSDF fusion.
4. **Layout.** RANSAC planes → Manhattan frame (up = floor normal), floor/ceiling heights, and a
   room footprint chosen on a cell complex of wall planes by free-space evidence. Written in the
   Mode A `Layout` schema, so Mode A's extruder builds a clean architectural model of the room.
   Observed surfaces near layout planes are snapped onto them.
5. **Honest completion (ours).** Each shell surface is rasterised into 3 cm texels, each labelled
   **observed** (reconstructed surface on the plane), **opening** (camera saw *through* the plane
   — never filled) or **generated** (never seen). Only generated texels are filled: plane
   geometry with texture from inpainted low frequencies + high-frequency patches of the same
   surface. Generated geometry is a separate GLB node with `extras.provenance = "generated"`,
   a separate PLY, a per-surface `provenance.json`, and a viewer toggle (show / highlight / hide).

## Evaluation protocol (Replica, NICE-SLAM renderings)
150 keyframes per video, every 8th held out (never used for geometry or appearance; registered
by PnP afterwards). Camera self-calibrated. Sim(3) alignment of train camera centres to GT.
Geometry on GT points visible from train views; unseen GT points scored separately. Room
dimensions from the strongest wall/floor/ceiling planes, same rule on prediction and GT.
Partial-scan protocol: only the first half of each video, so large parts of the room are unseen.

**Disclosures.** Thresholds were set on `office0`. While building the MVS stage we also looked at
GT depth on `room0` frames, so `room0` is treated as development data and excluded from the test
numbers. Test scenes: `room1, room2, office2, office3`. A first test run on `room1` exposed SfM
failures (scale drift, one teleporting frame). The generic temporal-jump check was added after
that and validated on dev only; a depth prior inside BA was tested on dev, gave mixed results,
and is therefore reported as a separate row (`full+depth_ba`), not part of `full`.

## Results

> **Status: development-scene numbers only.** The held-out test run (room1, room2, office2,
> office3) was stopped before completion, so there are no test numbers yet. The table below is
> `office0`, the scene thresholds were tuned on, so treat it as evidence that each component
> works, not as a generalisation claim. It was produced before the final SfM changes
> (temporal-jump check, 60-frame scale estimate). Reproduce or extend with
> `python eval/video_eval.py --scenes <scenes> --out <dir>` and `python eval/video_report.py --dir <dir>`.

Ablation on `office0` (cumulative; IBR = image-based rendering over our geometry, mesh = the
exported vertex-coloured GLB rendered directly):

| Config | PSNR mesh ↑ | PSNR IBR ↑ | SSIM ↑ | LPIPS ↓ | Chamfer cm ↓ | F@5cm ↑ | Chamfer, own metric scale cm ↓ | Dim err cm ↓ | Scale err % | Unseen recall@10cm ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 22.46 | 25.21 | 0.824 | 0.266 | 17.98 | 0.205 | 55.03 | 120.1 | +20.7 | 0.235 |
| +canon_scale | 23.21 | 26.64 | 0.849 | 0.227 | 18.75 | 0.196 | 10.99 | 9.9 | −9.7 | 0.259 |
| +grid_align | 26.12 | 30.40 | 0.916 | 0.127 | 14.24 | 0.234 | 10.27 | 6.2 | −9.7 | 0.247 |
| +mvs | 27.48 | 30.63 | 0.922 | 0.151 | 10.50 | 0.302 | 9.71 | 22.4 | −9.7 | 0.260 |
| +snap | 27.42 | 30.70 | 0.923 | 0.150 | 10.50 | 0.292 | 9.61 | 22.4 | −9.7 | 0.262 |
| **full** | **27.89** | **32.13** | **0.935** | 0.138 | 10.50 | 0.292 | **9.61** | 22.4 | −9.7 | **0.853** |

Generated geometry (full): 15.2 cm mean distance to the GT surface, 51% within 10 cm; 52% of the
room shell (floor, ceiling, walls) is labelled generated on this trajectory, which looks down a lot.

What each part buys (on this scene):
* **FOV-canonical scale** fixes metric size: Chamfer at our own scale 55 → 11 cm, dims 120 → 10 cm.
* **Grid alignment** and **mono-guided MVS** sharpen geometry: Chamfer 18.8 → 10.5 cm, F-score 0.20 → 0.30,
  and novel views +3.5 dB.
* **Completion** is what lets the room be closed: unseen-region recall 0.26 → 0.85, with every
  generated texel labelled; PSNR on generated pixels 31.1 dB.

Honest caveats:
* Room-dimension error rises from 6 cm (`+grid_align`) to 22 cm with MVS. All three dimensions
  shrink by the same ~5%: MVS geometry follows the SfM poses, whose metric scale is 9.7% off on
  this scene, and the grid-alignment row was right partly by a compensating bias. The remaining
  dimension error is a scale-estimation problem, not a layout problem.
* SfM is the weakest link: on test scene `room1` a first run drifted in scale (27 cm ATE) and all
  downstream metrics collapsed. The temporal-jump check fixes isolated bad frames; scale drift
  on weakly connected videos remains open (the depth-prior BA was mixed on dev).
* Generated surfaces are planar shell fills (walls, floor, ceiling). Furniture backs and objects
  that were never seen are not invented.
* CPU runtime is roughly 8–25 min per video on a laptop, with depth + MVS dominating.

