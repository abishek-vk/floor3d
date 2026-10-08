# val split, 50 plans

| Config | Room IoU | Class mIoU | Wall IoU | Room R@.5 | Room P@.5 | Door P | Door R | Win P | Win R | Dim MAPE | Area MAPE | Tot.Area APE | Scale APE | Time s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.771 | 0.635 | 0.720 | 0.857 | 0.561 | 0.823 | 0.923 | 0.766 | 0.847 | 0.194 | 0.505 | 0.574 | 0.154 | 4.227 |
| dev-nofill | 0.706 | 0.586 | 0.700 | 0.776 | 0.885 | 0.860 | 0.902 | 0.820 | 0.815 | 0.126 | 0.227 | 0.251 | 0.108 | 6.668 |
| +topology | 0.797 | 0.652 | 0.700 | 0.880 | 0.811 | 0.860 | 0.902 | 0.820 | 0.815 | 0.127 | 0.230 | 0.240 | 0.108 | 6.725 |

Scale sources: {"baseline": {"door_width": 50}, "dev-nofill": {"door_width": 50}, "+topology": {"door_width": 50}}
