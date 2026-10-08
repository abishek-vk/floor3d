# 2-minute demo script

Setup before going on stage: venv active, model weights cached (first OCR call loads ~1 s of
ONNX models), a browser window open, `results/results.md` open in a second tab.
Demo plan: an **unseen test plan**, e.g. `data/cubicasa5k/high_quality_architectural/5927/F1_scaled.png`
(has dimension strings `6000`, `7200`, `11 440` and room areas, so OCR scale + solver fire).

| Time | Show | Say |
|---|---|---|
| 0:00-0:15 | The input plan image | "A scanned architectural plan we've never seen. Goal: a metric, walkable 3D model." |
| 0:15-0:30 | Terminal: `python run.py --input <plan> --out out/demo` (~15 s on CPU) | "One command. Runs on a laptop CPU; every fallback it needs is logged." |
| 0:30-0:55 | `out/demo/debug/`: `03_seg_overlay` → `05_walls_fused` → `06_walls_raw_segments` → `07_barrier` → `08_layout` | "A pretrained segmentation net, fused with classical thick-stroke extraction; walls become centrelines with thickness; topology cleanup closes rooms; rooms split only at reflex corners, where the net sees different room types." |
| 0:55-1:10 | `layout.json` → `meta.solver.annotations` | "OCR reads 6000, 7200, the room areas. They vote for a metric scale, then a robust least-squares solver fits the walls to the written dimensions and flags any annotation that disagrees." |
| 1:10-1:35 | `viewer.html`: orbit → **2D plan** toggle → **Walk** through a door | "GLB export with separate walls, doors, windows, floors. Labels show metric size. Toggle the original plan underneath. Walk mode, eye height 1.6 m, walls collide." |
| 1:35-2:00 | `results/results.md` table + side-by-side figure | "On 30 held-out plans vs the raw CubiCasa baseline: (read room precision, dimension error, scale error). Ablation shows what each part buys, and the solver study shows it catches corrupted annotations." |

Fallback if live inference is slow: open the pre-built `results/runs/full/<plan>/viewer.html`
and the matching `debug/` folder.
