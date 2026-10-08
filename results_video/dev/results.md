# Mode B results (office0)

Replica, 1 scene(s); 150 keyframes per video, every 8th held out for novel-view evaluation; intrinsics self-calibrated. Means over scenes; best per column in bold.

## Ablation (cumulative)

| Config | PSNR ↑ (mesh) | PSNR ↑ (IBR) | SSIM ↑ (IBR) | LPIPS ↓ (IBR) | Chamfer-L1 cm ↓ | F@5cm ↑ | Chamfer metric cm ↓ | Dim err cm ↓ (layout) | Dim err cm ↓ (planes) | |Scale err| % ↓ | Unseen recall@10cm ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 22.46 | 25.21 | 0.824 | 0.266 | 17.98 | 0.205 | 55.03 | 120.1 | 122.5 | 20.7 | 0.235 |
| +canon_scale | 23.21 | 26.64 | 0.849 | 0.227 | 18.75 | 0.196 | 10.99 | 9.9 | **17.6** | **9.7** | 0.259 |
| +grid_align | 26.12 | 30.40 | 0.916 | **0.127** | 14.24 | 0.234 | 10.27 | **6.2** | 20.8 | **9.7** | 0.247 |
| +mvs | 27.48 | 30.63 | 0.922 | 0.151 | **10.50** | **0.302** | 9.71 | 22.4 | 38.9 | **9.7** | 0.260 |
| +snap | 27.42 | 30.70 | 0.923 | 0.150 | 10.50 | 0.292 | **9.61** | 22.4 | 25.8 | **9.7** | 0.262 |
| full | **27.89** | **32.13** | **0.935** | 0.138 | 10.50 | 0.292 | **9.61** | 22.4 | 25.8 | **9.7** | **0.853** |

## Secondary metrics

| Config | ATE cm | Focal err % | Acc cm | Comp cm | LPIPS (mesh) | PSNR obs px | PSNR gen px | Empty px | Unseen recall (obs only) | Generated acc cm | Generated within 10cm | Shell generated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | **3.95** | **-2.5** | 21.96 | 13.99 | 0.391 | 31.06 | – | 0.055 | 0.235 | – | – | 0.647 |
| +canon_scale | **3.95** | **-2.5** | 24.08 | 13.41 | 0.366 | 31.07 | – | 0.035 | 0.259 | – | – | 0.613 |
| +grid_align | **3.95** | **-2.5** | 18.00 | 10.49 | 0.284 | 33.73 | – | 0.015 | 0.247 | – | – | 0.581 |
| +mvs | **3.95** | **-2.5** | **12.68** | 8.32 | 0.286 | **35.91** | – | 0.022 | 0.260 | – | – | 0.518 |
| +snap | **3.95** | **-2.5** | 12.71 | **8.30** | 0.288 | 35.88 | – | 0.021 | **0.262** | – | – | **0.518** |
| full | **3.95** | **-2.5** | 12.71 | **8.30** | **0.283** | 35.89 | **31.12** | **0.009** | **0.262** | **15.2** | **0.509** | **0.518** |

## Per scene (full vs baseline)

| Scene | Config | nvs_ibr_psnr | nvs_vertex_psnr | sim3_chamfer_cm | dim_err_layout_cm | scale_err_pct | gt_width_m | layout_width_m | gt_depth_m | layout_depth_m | gt_height_m | layout_height_m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| office0 | baseline | 25.21 | 22.46 | 17.98 | 120.10 | 20.67 | 3.85 | 5.23 | 4.61 | 5.95 | 2.86 | 3.74 |
| office0 | full | 32.13 | 27.89 | 10.50 | 22.44 | -9.65 | 3.93 | 3.72 | 4.59 | 4.32 | 2.86 | 2.68 |
