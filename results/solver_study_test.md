Plans: 22 (with >=2 annotated rooms), annotations/plan: 7.9

| Condition | Dim MAPE | Area MAPE | Tot.Area APE | Scale APE | Room IoU | Wall IoU |
|---|---|---|---|---|---|---|
| A. no annotations (door-width scale) | 0.111 | 0.196 | 0.200 | 0.072 | 0.685 | 0.688 |
| B. annotations -> scale only | 0.065 | 0.099 | 0.166 | 0.006 | 0.685 | 0.688 |
| C. constraint solver (ours) | 0.061 | 0.095 | 0.174 | 0.008 | 0.686 | 0.686 |
| D. solver, 20% corrupted annotations | 0.065 | 0.107 | 0.190 | 0.019 | 0.685 | 0.682 |

Corrupted annotations flagged: 37/40; clean annotations wrongly flagged: 25/243
