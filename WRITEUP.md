# Metric 3D models from floor plan images with dimension-constrained refinement

**Track HNX26EPS06, Mode A** · floor plan (PNG/JPG/PDF) → walls, doors, windows, rooms, metric scale → GLB + browser walkthrough.

## Problem
Learned floor-plan parsers (e.g. the CubiCasa5K multi-task network) produce good per-pixel masks
but not a usable model: rooms fragment into several blobs, walls are ragged, and nothing is in
metres, because pixels carry no scale. For a navigable 3D model, the geometry has to be
**structured** (wall centrelines with thickness, rooms as polygons, openings attached to walls) and
**metric** (it should agree with the dimensions written on the drawing).

## Method
1. **Hybrid detection.** The pretrained CubiCasa5K network gives room/wall/icon probabilities.
   A classical extractor (Otsu ink → morphological opening just below the thinnest wall) keeps
   only thick strokes with pixel-exact edges. Fusion: ink walls that agree with the learned mask,
   plus confident learned walls the ink test misses (outlined/hollow walls).
2. **Vectorisation.** Directional decomposition (opening with long horizontal/vertical kernels)
   turns the wall mask into centreline segments with thickness; residual blobs become diagonal
   walls or stubs. Topology cleanup merges collinear pieces (across door gaps), snaps endpoints
   to junctions, extends dangling ends, and prunes walls that bound no room.
3. **Rooms.** Enclosed free space between the walls, split only along rays extended from
   reflex corners, where the network sees different room types (so a rectangle is never cut
   and virtual boundaries are straight). Spaces walls cannot enclose (balconies behind thin
   railings, open spaces) come from the network's uncovered room regions.
4. **Openings** are attached to their host wall (offset, width). **Furniture symbols** become
   parametric proxies.
5. **Metric scale by voting.** RapidOCR (PaddleOCR models) reads dimension strings (12'6",
   3.60, 3600, 3.6 m, "a x b"), room and total areas. Each proposes m/px hypotheses (a length
   proposes every wall-to-wall span covering it); the scale is the peak of a kernel density in
   log space. Accepted only with two independent agreeing annotations, and only if it agrees
   with the door-width cue (0.78 m) within 28%.
6. **Research contribution: dimension-constrained refinement.** Unknowns: each axis-aligned
   wall's centreline coordinate, extent and thickness, plus the global log-scale.
   Soft constraints, each normalised by its σ, under a Huber loss
   (`scipy.optimize.least_squares`):
   - data fidelity
   - junction coincidence
   - collinear alignment
   - thickness consistency within clusters
   - OCR spans (a written length equals the distance between two wall faces)
   - room sizes (inner width/depth equals "a x b")
   - room and total areas
   - a scale prior

   Orthogonality and parallelism are exact by parametrisation. Rooms, openings and furniture
   follow through a monotone piecewise-linear remap. The solver reports every constraint's
   residual before and after, and **flags annotations that remain inconsistent** (> 3σ).
7. **Extrusion.** Walls are extruded to 2.7 m in three height bands with different footprints
   (below sill / sill-to-head / above head), which cuts doors (0–2.1 m) and windows
   (0.9–2.1 m) exactly without CSG. Separate named meshes and materials; GLB, OBJ and JSON.
   A self-contained three.js viewer offers orbit, first-person walk with collision, metric
   labels, and a 2D plan overlay.

__RESULTS__
