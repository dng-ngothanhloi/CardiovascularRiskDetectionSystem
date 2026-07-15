# L2 – Calibration & Reliability Summary

## Aggregate OOF (Pre vs Post)
| Model | Brier (Pre) | ECE (Pre) | Brier (Post) | ECE (Post) | ΔBrier | ΔECE |
|:------|------------:|----------:|-------------:|-----------:|------:|-----:|
| rf | 0.180076 | 0.011526 | 0.180587 | 0.021580 | 0.000511 | 0.010055 |
| tabnet | 0.179978 | 0.008946 | 0.178746 | 0.000000 | -0.001232 | -0.008946 |

## Per-fold
| Model | Fold | Method | Brier (Pre) | ECE (Pre) | Brier (Post) | ECE (Post) | rel_pre | rel_post |
|:------|-----:|:-------|------------:|----------:|-------------:|-----------:|:--------|:---------|
| rf | 4 | Platt | 0.182043 | 0.013590 | 0.182444 | 0.018096 | fold_4/reliability_before.png | fold_4/reliability_after.png |
| rf | 3 | Platt | 0.180026 | 0.013759 | 0.180533 | 0.022139 | fold_3/reliability_before.png | fold_3/reliability_after.png |
| rf | 2 | Platt | 0.176538 | 0.017290 | 0.177194 | 0.027439 | fold_2/reliability_before.png | fold_2/reliability_after.png |
| rf | 0 | Platt | 0.181683 | 0.012643 | 0.182098 | 0.021129 | fold_0/reliability_before.png | fold_0/reliability_after.png |
| rf | 1 | Platt | 0.180090 | 0.016646 | 0.180667 | 0.024169 | fold_1/reliability_before.png | fold_1/reliability_after.png |
| tabnet | 4 | Isotonic | 0.181797 | 0.009307 | 0.180735 | 0.000000 | fold_4/reliability_before.png | fold_4/reliability_after.png |
| tabnet | 3 | Isotonic | 0.180233 | 0.016046 | 0.178964 | 0.000000 | fold_3/reliability_before.png | fold_3/reliability_after.png |
| tabnet | 2 | Isotonic | 0.176340 | 0.015261 | 0.174893 | 0.000000 | fold_2/reliability_before.png | fold_2/reliability_after.png |
| tabnet | 0 | Isotonic | 0.181162 | 0.010359 | 0.180008 | 0.000000 | fold_0/reliability_before.png | fold_0/reliability_after.png |
| tabnet | 1 | Isotonic | 0.180358 | 0.013999 | 0.179132 | 0.000000 | fold_1/reliability_before.png | fold_1/reliability_after.png |

## Overlays
- plots/reliability_overlay_pre.png: reports/plots/reliability_overlay_pre.png
- plots/reliability_overlay_post.png: reports/plots/reliability_overlay_post.png
