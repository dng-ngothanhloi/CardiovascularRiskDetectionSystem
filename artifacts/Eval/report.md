# Stage 4 – Evaluation Report (Cardio Target)

This report summarizes out-of-fold (OOF) CV results and, when available, held-out Test results from **`final_model`**. OOF and Test metrics are reported separately and must not be mixed as one protocol. Evaluation does **not** fit calibrators (ECE only). Primary Test operating point uses the **OOF** threshold in `final_model/threshold_policy.json`; F1@best retuned on Test is a helper only.

## 1. OOF / CV Results

### Aggregate Metrics (OOF)

| Model | PR-AUC | ROC-AUC | F1@best | Thr(best) | Brier | ECE |
|------|--------:|--------:|--------:|----------:|------:|----:|
| rf | 0.7871 | 0.8022 | 0.7417 | 0.350 | 0.1802 | 0.0130 |
| tabnet | 0.7860 | 0.8025 | 0.7426 | 0.350 | 0.1800 | 0.0098 |
| vae_tabnet | - | - | - | - | - | - |

### Pairwise Comparison on OOF (PR-AUC, 95% CI via paired bootstrap)

| Pair | AP_A | AP_B | Δ(AP_A−AP_B) | 95% CI(Δ) |
|------|-----:|-----:|-------------:|:----------|
| RF vs TabNet | 0.7872 | 0.7860 | 0.0012 | [-0.0014, 0.0037] |
| TabNet vs VAE→TabNet | - | - | - | - |
| RF vs VAE→TabNet | - | - | - | - |

### Diagnostic Plots (OOF)

* **PR Curves**: `plots/pr_overlay.png`
* **ROC Curves**: `plots/roc_overlay.png`
* **Reliability Curves**: `plots/reliability_overlay.png`

### Confusion Matrices on OOF (at best threshold)

* **rf**: `plots/confusion_matrix_rf.png`
* **tabnet**: `plots/confusion_matrix_tabnet.png`
* **vae_tabnet**: -

## 2. Held-out Test Results (`final_model` deploy)

Primary Test metrics use **`final_model`** predictions (params/calibrator/threshold selected on Train/OOF only; Test scored once after freeze). See `test_aggregation_meta.json`.

### Aggregate Metrics (Test — deploy @ OOF threshold) **primary**

| Model | PR-AUC | ROC-AUC | Thr(OOF) | F1@OOF | Acc@OOF | Sens@OOF | Spec@OOF | Brier | ECE |
|------|--------:|--------:|---------:|-------:|--------:|---------:|---------:|------:|----:|
| rf | 0.7737 | 0.7978 | 0.360 | 0.7392 | 0.7166 | 0.8036 | 0.6298 | 0.1826 | 0.0202 |
| tabnet | 0.7755 | 0.7979 | 0.350 | 0.7404 | 0.7108 | 0.8254 | 0.5964 | 0.1825 | 0.0183 |
| vae_tabnet | - | - | - | - | - | - | - | - | - |

Threshold from `final_model/threshold_policy.json` (selected on Train OOF; **not** re-tuned on Test).

### Helper Metrics (Test — F1@best retuned on Test) **secondary**

| Model | F1@best* | Thr(best)* |
|------|--------:|----------:|
| rf | 0.7409 | 0.350 |
| tabnet | 0.7404 | 0.350 |
| vae_tabnet | - | - |

\*Do **not** use as clinical / deploy operating point — optimistic relative to the frozen OOF threshold.

### Pairwise Comparison on Test — final_model (PR-AUC, 95% CI)

| Pair | AP_A | AP_B | Δ(AP_A−AP_B) | 95% CI(Δ) |
|------|-----:|-----:|-------------:|:----------|
| RF vs TabNet | 0.7738 | 0.7756 | -0.0018 | [-0.0066, 0.0029] |
| TabNet vs VAE→TabNet | - | - | - | - |
| RF vs VAE→TabNet | - | - | - | - |

### Confusion Matrices on Test (at **OOF deploy** threshold)

* **rf**: `plots/confusion_matrix_test_rf.png`
* **tabnet**: `plots/confusion_matrix_test_tabnet.png`
* **vae_tabnet**: -

### Test ensemble (diagnostic only)

If present, `test_ensemble_predictions.csv` is the 5-fold mean (nested-CV style). **Not used for deploy.** Compare with `final_model/test_metrics.json`.
