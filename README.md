# CardiovascularRiskDetectionSystem

**Theme:** Rule-Guided Deep Ensemble Learning for Cardiovascular Risk Discovery  

**Pipeline:** EDA → Preprocess → Feature Engineering → Train (L1) → Evaluation → Interpretation (L2) → Discovery (L3)

Chi tiết tham số từng bước: [`docs/Flow-Pipeline.md`](docs/Flow-Pipeline.md)

---

## 1. Project structure

```
CardiovascularRiskDetectionSystem/
├── data/cardio_data.csv
├── configs/
│   ├── default.yaml          # full HPO / production
│   ├── fast.yaml             # fewer trials (dev)
│   ├── no_hpo.yaml           # smoke / debug
│   └── l3_discovery.yaml     # L3 Apriori settings
├── src/
│   ├── data_understanding.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py           # Stratified K-Fold + Optuna HPO
│   ├── hpo.py                # library used by modeling (not a CLI)
│   ├── evaluation.py         # L1: OOF + held-out Test
│   ├── interpretation/       # L2: attention, calibration, importance
│   ├── xai_interpretability.py
│   ├── run_discovery.py      # L3 entry
│   ├── discovery/            # Apriori pipeline
│   ├── models/               # RF, TabNet, VAE, Joint VAE–TabNet
│   └── test_prediction_verification.py  # optional 20-case spotlight
├── artifacts/                # generated outputs
├── docs/
├── tests/
├── requirements.txt
├── Makefile
├── make.sh
└── make.bat
```

---

## 2. Installation

**Prerequisites:** Python ≥ 3.11, pip. On macOS/Linux, `make` is usually available.

```bash
cd /path/to/CardiovascularRiskDetectionSystem
bash local_setup_env.sh
source .venv/bin/activate

chmod +x make.sh   # optional; run make from any dir via ./make.sh
make setup         # install + check data/config
```

Verify:

```bash
make check-data    # expects data/cardio_data.csv
make check-config  # expects configs/default.yaml
make help
```

---

## 3. Pipeline overview

```text
data/cardio_data.csv
        │
        ▼
[1] EDA ──────────────────────────► artifacts/eda/
        │
        ▼
[2] Preprocess ───────────────────► artifacts/preprocessing/
        │
        ▼
[3] FE (rf) + FE (tabnet) ────────► artifacts/fe/{random_forest,tabnet}/
        │                              Train_*.csv / Test_*.csv (80/20)
        ▼
[4] Train --model rf|tabnet|… ────► artifacts/Model/{model}/
        │     HPO inside each fold     oof_predictions.csv
        │     test_scoring + final_model
        ▼
[5] L1 Evaluation ────────────────► artifacts/Eval/
        │     OOF (Train) + Test       report.md, summary.json
        ▼
[6] L2 Interpretation ────────────► artifacts/Interpretation/
        │     attention / calibration / XAI
        ▼
[7] L3 Discovery (Apriori) ───────► artifacts/Discovery/
```

**Validation** is fold-internal on Train (no separate `Val_data_*.csv`).  
**Test** (~20%) is held out at FE; deploy scoring uses **`final_model`** (after freeze) when `modeling.test_scoring` + `modeling.final_model` are true. Fold ensemble is diagnostic only.

**Publication Methods (short):**

- `src.evaluation` compares RF vs TabNet on **OOF (Train)** and **Test–final_model**. It does **not** fit calibrators on Test (ECE only).
- **L1** calibration/threshold = deploy (Train OOF). **L2** `run_calibration` = diagnostic second pass on OOF — not Test.
- Primary Results: OOF / Test PR-AUC + metrics @ **OOF threshold** on Test. Secondary: Test F1@best helpers, ensemble, lifestyle spotlight.
- Full artifact map + known gaps: [`docs/Flow-Pipeline.md`](docs/Flow-Pipeline.md) → *Methods note (publication)*.

| After change #3/#4 | Cần xóa & train lại? |
|--------------------|----------------------|
| Báo cáo deploy chính thức (`final_model/test_*`, `param_strategy=best_fold_val_pr_auc`) | **Có** — overwrite `artifacts/Model/{rf,tabnet}` (rồi chạy lại `evaluate` / L2 phụ thuộc Model). Không cần xóa `fe/` / EDA. |
| Chỉ chạy L2 attention / L3 trên fold weights + OOF cũ | Có thể tiếp tục với artifacts cũ (nhưng TabNet hiện vẫn `median_mode…`, thiếu `final_model/test_*`). |

---

## 4. Quick start (recommended)

Run from project root. Override config with `CONFIG=...`.

### Option A — Make targets

```bash
# Full production-style run (slow: HPO × folds × models)
make all CONFIG=configs/default.yaml

# Faster end-to-end
make all-fast

# Stage by stage
make eda
make preprocess
make fe                 # rf + tabnet
make train              # rf + tabnet (or: make train-rf / train-tabnet)
make evaluate
make interpret          # attention + calibrate + lifestyle-prep + xai
make discover
```

### Option B — Equivalent Python CLI

```bash
CONFIG=configs/default.yaml   # or fast.yaml / no_hpo.yaml

python -m src.data_understanding --config $CONFIG
python -m src.preprocessing --config $CONFIG

python -m src.feature_engineering --config $CONFIG --model-type rf
python -m src.feature_engineering --config $CONFIG --model-type tabnet

python -m src.modeling --config configs/default.yaml --model rf
python -m src.modeling --config configs/default.yaml --model tabnet
# optional: python -m src.modeling --config $CONFIG --model vae_tabnet --joint

# [5] L1 — OOF (Train) + Test final_model (deploy)
python -m src.evaluation \
  --model_root artifacts/Model \
  --out_dir artifacts/Eval \
  --n_boot 2000 --seed 42

# [6a] L2 attention — masks on Train fold val splits; RF importance from fold jobs
python -m src.interpretation.run_attention_summary \
  --fe_dir_tabnet artifacts/fe/tabnet \
  --model_root artifacts/Model/tabnet \
  --out_root artifacts/Interpretation \
  --rf_model_root artifacts/Model/rf \
  --fe_dir_rf artifacts/fe/random_forest

# [6b] L2 calibration — re-fit on OOF (Train), not Test
python -m src.interpretation.run_calibration \
  --model rf --method platt \
  --model_root artifacts/Model \
  --out_root artifacts/Interpretation/calibration --n_bins 15

python -m src.interpretation.run_calibration \
  --model tabnet --method isotonic \
  --model_root artifacts/Model \
  --out_root artifacts/Interpretation/calibration --n_bins 15

# [6c] Prerequisites for lifestyle XAI (Test spotlight — not L1 formal metrics)
python -m src.test_prediction_verification \
  --n_samples 0 \
  --test_prediction src/test_prediction_input.txt
python -m src.create_lifestyle_enriched_predictions \
  --fe_dir artifacts/fe \
  --predictions_dir artifacts/Interpretation/test_prediction_result \
  --output_dir artifacts/Interpretation/test_prediction_result

# [6d] Lifestyle charts (needs *_with_lifestyle.csv) + OOF reliability charts
python -m src.xai_interpretability --output-dir artifacts/Interpretation/reports

# [7] L3 — TabNet attention itemsets (Train OOF labels) → Apriori
python -m src.run_discovery --config configs/l3_discovery.yaml
```

### Only L1 (no L2/L3)

```bash
make pipeline-l1 CONFIG=configs/fast.yaml
```

---

## 5. Configs

| File | Use when |
|------|----------|
| `configs/default.yaml` | Final / paper runs (`hpo.n_trials: 30`, TabNet `epochs: 80`) |
| `configs/fast.yaml` | Faster iteration (`n_trials: 10`) |
| `configs/no_hpo.yaml` | Smoke tests (`hpo.enabled: false`) |
| `configs/l3_discovery.yaml` | L3 Apriori / attention→itemsets |

Key modeling flags in YAML:

- `modeling.hpo.enabled` / `n_trials` / `composite_weights`
- `modeling.calibration.method` (`platt` | `isotonic`)
- `modeling.test_scoring: true` — score held-out Test after freeze
- `modeling.final_model: true` — deploy under `final_model/` (+ root `test_predictions.csv` copy)
- `modeling.final_model_param_strategy: best_fold_val_pr_auc`
- `evaluation.cv_folds: 5`

**HPO is not a separate CLI.** Optuna runs inside `src.modeling` when HPO is enabled. There is no `python -m src.hpo`.

---

## 6. Make targets cheat sheet

| Target | What it does |
|--------|----------------|
| `setup` | install + check data/config |
| `eda` | Stage 1 |
| `preprocess` | Stage 2 |
| `fe` / `fe-rf` / `fe-tabnet` | Stage 3 |
| `train` / `train-rf` / `train-tabnet` / `train-vae-tabnet` | Stage 4 |
| `evaluate` | Stage 5 → `artifacts/Eval/` |
| `interpret` | Stage 6 (attention + calibrate + xai) |
| `attention` / `calibrate` / `xai` | L2 pieces alone |
| `discover` / `discover-fast` | Stage 7 |
| `pipeline-l1` | Stages 1–5 |
| `all` / `all-fast` | Stages 1–7 |
| `clean` / `clean-models` / `clean-eda` / `clean-fe` | Wipe artifacts |

Examples:

```bash
make train-rf CONFIG=configs/no_hpo.yaml
make evaluate N_BOOT=200
./make.sh all-fast   # from any subdirectory
```

---

## 7. Data & target

- **Path:** `data/cardio_data.csv`
- **Target:** `cardio` (0/1)

| Field | Type | Description |
|-------|------|-------------|
| age | int (days) | Age |
| height | int (cm) | Height |
| weight | float (kg) | Weight |
| gender | categorical | Gender code |
| ap_hi / ap_lo | int | Systolic / diastolic BP |
| cholesterol | 1–3 | Normal → well above normal |
| gluc | 1–3 | Glucose category |
| smoke / alco / active | binary | Lifestyle |
| cardio | binary | CVD label |

---

## 8. Models (L1)

| Model | CLI | Role |
|-------|-----|------|
| RandomForest | `--model rf` | Strong tabular baseline |
| TabNet | `--model tabnet` | Attentive deep tabular (feeds L2/L3) |
| VAE–TabNet | `--model vae_tabnet [--joint]` | Latent `z` + classifier |

Training protocol:

1. Stratified **5-fold CV on Train**
2. Per-fold HPO → resolved params → fit → calibrate on val fold → OOF
3. Select best-fold params by val PR-AUC → retrain **final_model** on full Train
4. Calibrator / threshold from OOF → predict **held-out Test** once (deploy)
5. Optional diagnostic: 5-fold Test ensemble (`test_ensemble_*.csv`)

L1 evaluation reports **OOF (Train)** and **Test — final_model** separately in `artifacts/Eval/report.md`.

---

## 9. L2 Interpretation & L3 Discovery

**L2** (`make interpret` = attention → calibrate → lifestyle-prep → xai):

- TabNet per-sample attention masks (**Train** fold-val samples)
- RF Gini vs TabNet importance (fold artifacts)
- Post-hoc OOF calibration (Platt / Isotonic) — **Train OOF**
- Lifestyle XAI needs Test spotlight CSVs: `test_prediction_verification` → `create_lifestyle_enriched_predictions` → `xai_interpretability`

**L3** (`make discover`):

- Attention → binary itemsets → Apriori rules (`IF features THEN cardio=1`)
- Config: `configs/l3_discovery.yaml`
- Output: `artifacts/Discovery/`

Design references: [`docs/L2-L3_DetailDesign.md`](docs/L2-L3_DetailDesign.md), [`docs/Design_L3_Tabnet-Apriori.md`](docs/Design_L3_Tabnet-Apriori.md).

---

## 10. Reproducibility

- Global seed: `random_seed: 42` in YAML
- Seeds for NumPy / TensorFlow set in `src/utils.py`
- Fold splits saved under `artifacts/Model/<model>/cv_splits/`

---

## 11. Platform notes

**macOS / Linux / WSL**

```bash
make help
./make.sh eda    # from any subdirectory
```

**Windows**

- Prefer WSL or use `make.bat` for mirrored targets.
- Or run the Python CLI commands in §4 Option B.

**Common failures**

| Symptom | Fix |
|---------|-----|
| `modeling` missing `--model` | Use `make train-rf` / `--model rf` |
| `evaluation` unknown `--config` | Use `--model_root` / `--out_dir` |
| `No module src.interpretability` | Use `make interpret` (attention/calibrate/xai) |
| `No module src.hpo` as CLI | HPO is inside modeling |
| `*_with_lifestyle.csv` FileNotFoundError | Run verification + `create_lifestyle_enriched_predictions` before `xai` |
| Data not found | Ensure `data/cardio_data.csv` and `paths.data` match |

---

## 12. Academic references (selected)

1. Breiman, L. “Random Forests.” *Machine Learning*, 2001.  
2. Arik, S. Ö., & Pfister, T. “TabNet: Attentive Interpretable Tabular Learning.” *AAAI*, 2021.  
3. Kingma, D. P., & Welling, M. “Auto-Encoding Variational Bayes.” *ICLR*, 2014.  
4. Lundberg, S. M., & Lee, S.-I. “A Unified Approach to Interpreting Model Predictions.” *NeurIPS*, 2017.  
5. Agrawal, R., Imieliński, T., & Swami, A. “Mining Association Rules.” *SIGMOD*, 1993.  
6. Dietterich, T. G. “Ensemble Methods in ML.” *Multiple Classifier Systems*, 2000.  
