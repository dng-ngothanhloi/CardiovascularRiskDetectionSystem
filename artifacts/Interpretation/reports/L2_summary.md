# L2 – Calibration & Reliability Summary

## Aggregate OOF (Pre vs Post)
| Model | Brier (Pre) | ECE (Pre) | Brier (Post) | ECE (Post) | ΔBrier | ΔECE |
|:------|------------:|----------:|-------------:|-----------:|------:|-----:|
| rf | 0.180187 | 0.012951 | 0.180706 | 0.021618 | 0.000519 | 0.008667 |
| tabnet | 0.179997 | 0.009829 | 0.178788 | 0.000000 | -0.001209 | -0.009829 |

## Per-fold
| Model | Fold | Method | Brier (Pre) | ECE (Pre) | Brier (Post) | ECE (Post) | rel_pre | rel_post |
|:------|-----:|:-------|------------:|----------:|-------------:|-----------:|:--------|:---------|
| rf | 4 | Platt | 0.182137 | 0.013074 | 0.182559 | 0.019155 | fold_4/reliability_before.png | fold_4/reliability_after.png |
| rf | 2 | Platt | 0.176433 | 0.014959 | 0.177091 | 0.024824 | fold_2/reliability_before.png | fold_2/reliability_after.png |
| rf | 3 | Platt | 0.180326 | 0.016808 | 0.180820 | 0.022611 | fold_3/reliability_before.png | fold_3/reliability_after.png |
| rf | 1 | Platt | 0.180428 | 0.018717 | 0.181024 | 0.025436 | fold_1/reliability_before.png | fold_1/reliability_after.png |
| rf | 0 | Platt | 0.181612 | 0.010932 | 0.182034 | 0.020038 | fold_0/reliability_before.png | fold_0/reliability_after.png |
| tabnet | 4 | Isotonic | 0.181766 | 0.009996 | 0.180688 | 0.000000 | fold_4/reliability_before.png | fold_4/reliability_after.png |
| tabnet | 2 | Isotonic | 0.176291 | 0.014710 | 0.174885 | 0.000000 | fold_2/reliability_before.png | fold_2/reliability_after.png |
| tabnet | 3 | Isotonic | 0.180244 | 0.015938 | 0.178996 | 0.000000 | fold_3/reliability_before.png | fold_3/reliability_after.png |
| tabnet | 1 | Isotonic | 0.180446 | 0.015929 | 0.179184 | 0.000000 | fold_1/reliability_before.png | fold_1/reliability_after.png |
| tabnet | 0 | Isotonic | 0.181239 | 0.009855 | 0.180187 | 0.000000 | fold_0/reliability_before.png | fold_0/reliability_after.png |

## Overlays
- plots/reliability_overlay_pre.png: reports/plots/reliability_overlay_pre.png
- plots/reliability_overlay_post.png: reports/plots/reliability_overlay_post.png
