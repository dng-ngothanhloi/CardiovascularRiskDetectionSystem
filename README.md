# CardiovascularRiskDetectionSystem

**Chủ đề:** Học sâu kết hợp khai thác luật có khả giải thích để dự đoán nguy cơ bệnh tim mạch.  
(*A Three-Tier Explainable Pipeline Combining Deep Learning and Association Rule Mining for Cardiovascular Disease Risk Prediction*)

**Pipeline:** EDA → Tiền xử lý → Feature Engineering → Huấn luyện (L1) → Đánh giá → Diễn giải (L2) → Khám phá (L3)

Chi tiết tham số từng bước: [`docs/Flow-Pipeline.md`](docs/Flow-Pipeline.md)

---

## 1. Cấu trúc dự án

```
CardiovascularRiskDetectionSystem/
├── data/cardio_data.csv
├── configs/
│   ├── default.yaml                      # HPO đầy đủ / production
│   ├── fast.yaml                         # ít trial hơn (dev)
│   ├── no_hpo.yaml                       # smoke / debug
│   └── l3_discovery.yaml                 # cấu hình Apriori L3
├── src/
│   ├── data_understanding.py             # Đánh giá, phân tích dữ liệu
│   ├── preprocessing.py                  # Tiền xử lý dữ liệu (được phát triển dựa vào data_understanding và phù hợp với tập dữ liệu hiện tại
│   ├── feature_engineering.py            # Tái tạo/Tạo mới đặc tính bổ sung cho mô hình học sâu (Tabnet) và mạng neuron (RandomForest) xây dựng cho riêng bộ dữ liệu
│   ├── modeling.py                       # Stratified K-Fold + Optuna HPO
│   ├── hpo.py                            # thư viện dùng bởi modeling (không phải CLI)
│   ├── evaluation.py                     # L1: OOF + Test hold-out
│   ├── interpretation/                   # L2: attention, calibration, importance
│   ├── xai_interpretability.py            
│   ├── run_discovery.py                  # Entry L3
│   ├── discovery/                        # pipeline Apriori
│   ├── models/                           # RF, TabNet, VAE, Joint VAE–TabNet
│   └── test_prediction_verification.py   # spotlight tùy chọn 20 mẫu - được lựa chọn sẵn từ tập Test
├── artifacts/                            # Lưu trữ tất cả kết quả của các công đoạn
├── docs/
├── tests/
├── requirements.txt
├── Makefile
├── make.sh
└── make.bat
```

---

## 2. Cài đặt

**Yêu cầu:** Python ≥ 3.11, pip. Trên macOS/Linux thường đã có `make`.

```bash
cd /path/to/CardiovascularRiskDetectionSystem
bash local_setup_env.sh
source .venv/bin/activate

chmod +x make.sh   # tùy chọn; chạy make từ mọi thư mục qua ./make.sh
make setup         # cài đặt + kiểm tra data/config
```

Kiểm tra:

```bash
make check-data    # cần có data/cardio_data.csv
make check-config  # cần có configs/default.yaml
make help
```

---

## 3. Tổng quan pipeline

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
        │     HPO trong từng fold      oof_predictions.csv
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

**Validation** nằm trong từng fold trên tập Train (không có file `Val_data_*.csv` riêng).  
**Test** (~20%) được giữ lại từ bước FE; là tập dữ kiệu được dùng để kiểm tra với **`final_model`** khi `modeling.test_scoring` + `modeling.final_model` = true. Ensemble theo fold dùng cho chẩn đoán(Prediction) nguy cơ CVD.

---

## 4. Hướng dẫn cài đặt và tái lập Pipeline

Chạy từ thư mục gốc dự án. Ghi đè config bằng `CONFIG=...`.

### Cách A — Make targets

```bash
# Chạy production đầy đủ (chậm: HPO × folds × models)
make all CONFIG=configs/default.yaml

# End-to-end nhanh hơn
make all-fast

# Từng giai đoạn
make eda
make preprocess
make fe                 # rf + tabnet
make train              # rf + tabnet (hoặc: make train-rf / train-tabnet)
make evaluate
make interpret          # attention + calibrate + lifestyle-prep + xai
make discover
```

### Cách B — CLI Python tương đương

```bash
CONFIG=configs/default.yaml   # hoặc fast.yaml / no_hpo.yaml

python -m src.data_understanding --config $CONFIG
#  EDA Chart cho báo cáo.
python  src/EDA-Chart.py

python -m src.preprocessing --config $CONFIG

python -m src.feature_engineering --config $CONFIG --model-type rf
python -m src.feature_engineering --config $CONFIG --model-type tabnet

python -m src.modeling --config configs/default.yaml --model rf
python -m src.modeling --config configs/tabnet_baseline.yaml --model tabnet
# tùy chọn: python -m src.modeling --config $CONFIG --model vae_tabnet --joint

# [5] L1 — OOF (Train) + Test final_model (deploy)
python -m src.evaluation \
  --model_root artifacts/Model \
  --out_dir artifacts/Eval \
  --n_boot 2000 --seed 42

# [6a] L2 attention — mask trên fold-val của Train; importance RF từ fold jobs
python -m src.interpretation.run_attention_summary \
  --fe_dir_tabnet artifacts/fe/tabnet \
  --model_root artifacts/Model/tabnet \
  --out_root artifacts/Interpretation \
  --rf_model_root artifacts/Model/rf \
  --fe_dir_rf artifacts/fe/random_forest

# [6b] L2 calibration — fit lại trên OOF (Train), không phải Test
python -m src.interpretation.run_calibration \
  --model rf --method platt \
  --model_root artifacts/Model \
  --out_root artifacts/Interpretation/calibration --n_bins 15

python -m src.interpretation.run_calibration \
  --model tabnet --method isotonic \
  --model_root artifacts/Model \
  --out_root artifacts/Interpretation/calibration --n_bins 15

# [6c] Điều kiện tiên quyết cho lifestyle XAI (Test spotlight — không phải metric L1 chính thức)
python -m src.test_prediction_verification \
  --n_samples 0 \
  --test_prediction src/test_prediction_input.txt
python -m src.create_lifestyle_enriched_predictions \
  --fe_dir artifacts/fe \
  --predictions_dir artifacts/Interpretation/test_prediction_result \
  --output_dir artifacts/Interpretation/test_prediction_result

# [6d] Biểu đồ lifestyle (cần *_with_lifestyle.csv) + reliability OOF
python -m src.xai_interpretability --output-dir artifacts/Interpretation/reports

# [7] L3 — itemset attention TabNet (nhãn Train OOF) → Apriori
python -m src.run_discovery --config configs/l3_discovery.yaml
```

### Chỉ Train ở L1 (không L2/L3)

```bash
make pipeline-l1 CONFIG=configs/fast.yaml
```

---

## 5. Cấu hình (Configs)

| File | Dùng khi |
|------|----------|
| `configs/default.yaml` | Chạy cuối / paper (RF HPO `n_trials: 30`; TabNet params pinned sparse) |
| `configs/tabnet_baseline.yaml` | Train TabNet: tắt HPO + params sparse baseline fold_0 (~19 luật L3) |
| `configs/fast.yaml` | Lặp nhanh hơn (`n_trials: 10`) |
| `configs/no_hpo.yaml` | Smoke test (`hpo.enabled: false`) |
| `configs/l3_discovery.yaml` | L3 Apriori / attention→itemsets |

Cờ modeling quan trọng trong YAML:

- `modeling.hpo.enabled` / `n_trials` / `composite_weights`
- `modeling.calibration.method` (`platt` | `isotonic`)
- `modeling.test_scoring: true` — chấm điểm Test hold-out sau khi freeze
- `modeling.final_model: true` — deploy dưới `final_model/` (+ bản sao `test_predictions.csv` ở root)
- `modeling.final_model_param_strategy: best_fold_val_pr_auc`
- `evaluation.cv_folds: 5`

**HPO không phải CLI riêng.** Optuna chạy bên trong `src.modeling` khi HPO được bật. Không có `python -m src.hpo`.

---

## 6. Bảng tra Make targets

| Target | Việc thực hiện |
|--------|----------------|
| `setup` | cài đặt + kiểm tra data/config |
| `eda` | Giai đoạn 1 |
| `preprocess` | Giai đoạn 2 |
| `fe` / `fe-rf` / `fe-tabnet` | Giai đoạn 3 |
| `train` / `train-rf` / `train-tabnet` / `train-vae-tabnet` | Giai đoạn 4 |
| `evaluate` | Giai đoạn 5 → `artifacts/Eval/` |
| `interpret` | Giai đoạn 6 (attention + calibrate + xai) |
| `attention` / `calibrate` / `xai` | Từng phần L2 riêng |
| `discover` / `discover-fast` | Giai đoạn 7 |
| `pipeline-l1` | Giai đoạn 1–5 |
| `all` / `all-fast` | Giai đoạn 1–7 |
| `clean` / `clean-models` / `clean-eda` / `clean-fe` | Xóa artifacts |

Ví dụ:

```bash
make train-rf CONFIG=configs/no_hpo.yaml
make evaluate N_BOOT=200
./make.sh all-fast   # từ bất kỳ thư mục con nào
```

---

## 7. Dữ liệu & biến mục tiêu

- **Đường dẫn:** `data/cardio_data.csv`
- **Target:** `cardio` (0/1)

| Trường | Kiểu | Mô tả |
|--------|------|-------|
| age | int (ngày) | Tuổi |
| height | int (cm) | Chiều cao |
| weight | float (kg) | Cân nặng |
| gender | categorical | Mã giới tính |
| ap_hi / ap_lo | int | Huyết áp tâm thu / tâm trương |
| cholesterol | 1–3 | Bình thường → cao rõ |
| gluc | 1–3 | Nhóm glucose |
| smoke / alco / active | binary | Lối sống |
| cardio | binary | Nhãn bệnh tim mạch (CVD) |

---

## 8. Mô hình (L1)

| Mô hình | CLI | Vai trò |
|---------|-----|---------|
| RandomForest | `--model rf` | Baseline tabular mạnh |
| TabNet | `--model tabnet` | Deep tabular có attention (nuôi L2/L3) |
| VAE–TabNet | `--model vae_tabnet [--joint]` | Latent `z` + bộ phân loại |

Giao thức huấn luyện:

1. **5-fold CV phân tầng** trên Train
2. HPO từng fold → params đã resolve → fit → calibrate trên val fold → OOF
3. Chọn params fold tốt nhất theo val PR-AUC → train lại **final_model** trên toàn Train
4. Calibrator / threshold từ OOF → dự đoán **Test hold-out** một lần (deploy)
5. Chẩn đoán tùy chọn: ensemble Test 5-fold (`test_ensemble_*.csv`)

Đánh giá L1 báo cáo riêng **OOF (Train)** và **Test — final_model** trong `artifacts/Eval/report.md`.

---

## 9. L2 Diễn giải & L3 khai phá tập luật với Apriori 

**L2** (`make interpret` = attention → calibrate → lifestyle-prep → xai):

- Attention mask theo mẫu của TabNet (mẫu **fold-val** trên Train)
- Importance RF Gini so với TabNet (artifact theo fold)
- Calibration hậu kỳ trên OOF (Platt / Isotonic) — **Train OOF**
- Lifestyle XAI cần CSV spotlight Test: `test_prediction_verification` → `create_lifestyle_enriched_predictions` → `xai_interpretability`

**L3** (`make discover`):

- Attention → itemset nhị phân → luật Apriori (`IF features THEN cardio=1`)
- Config: `configs/l3_discovery.yaml`
- Đầu ra: `artifacts/Discovery/`

Tài liệu thiết kế: [`docs/HeartDiseasePrediction_DetailDesign.md`](docs/HeartDiseasePrediction_DetailDesign.md), [`docs/L2-L3_DetailDesign.md`](docs/L2-L3_DetailDesign.md), [`docs/Design_L3_Tabnet-Apriori.md`](docs/Design_L3_Tabnet-Apriori.md).

---

## 10. Kết quả tái lập và kiểm chứng (Reproducibility)

- Seed toàn cục: `random_seed: 42` trong YAML
- Seed NumPy / TensorFlow thiết lập trong `src/utils.py`
- Phân chia fold lưu tại `artifacts/Model/<model>/cv_splits/`
- Config deploy TabNet: `configs/tabnet_baseline.yaml` (tắt HPO, params pinned cho attention thưa → L3 ổn định)

### Kết quả từng công đoạn ở `artifacts/` 

| Giai đoạn | Đường dẫn |
|-----------|-----------|
| EDA | [`artifacts/eda/key_insights.md`](artifacts/eda/key_insights.md), [`artifacts/eda/data_recommendations.md`](artifacts/eda/data_recommendations.md) |
| L1 Evaluation | [`artifacts/Eval/report.md`](artifacts/Eval/report.md) |
| L2 Calibration | [`artifacts/Interpretation/reports/L2_summary.md`](artifacts/Interpretation/reports/L2_summary.md) |
| L2 Importance | [`artifacts/Interpretation/reports/L2_importance_summary.md`](artifacts/Interpretation/reports/L2_importance_summary.md), [`artifacts/Interpretation/importance/rf/summary.md`](artifacts/Interpretation/importance/rf/summary.md), [`artifacts/Interpretation/importance/tabnet/summary.md`](artifacts/Interpretation/importance/tabnet/summary.md) |
| L2 Calibration chi tiết | [`artifacts/Interpretation/calibration/rf/summary.md`](artifacts/Interpretation/calibration/rf/summary.md), [`artifacts/Interpretation/calibration/tabnet/summary.md`](artifacts/Interpretation/calibration/tabnet/summary.md) |
| L2 Test spotlight | [`artifacts/Interpretation/test_prediction_result/Test_Prediction_Verification.md`](artifacts/Interpretation/test_prediction_result/Test_Prediction_Verification.md) |
| L3 Discovery | [`artifacts/Discovery/reports/L3_summary.md`](artifacts/Discovery/reports/L3_summary.md) |

### Kết quả chính 
**L1 — Prediction** (`artifacts/Eval/summary.json`)

| Mô hình | OOF PR-AUC | OOF ROC-AUC | Test PR-AUC | Test ROC-AUC | Acc@ngưỡng OOF |
|---------|-----------:|-----------:|------------:|-------------:|---------------:|
| RF | 0.787 | 0.802 | 0.774 | 0.798 | 0.717 |
| TabNet | 0.786 | 0.802 | 0.776 | 0.798 | 0.711 |

RF vs TabNet Test ΔPR-AUC ≈ −0.002 (KTC 95% chứa 0) → **tương đương (parity)**.

**L2 — Interpretation**

| Tín hiệu | Top features |
|----------|----------------|
| RF Gini | `ap_hi`, `bp_stage`, `ap_lo`, `pulse_pressure`, `age_years` |
| TabNet attention | `bmi`, `bmi_group`, `glucose_category`, `height`, `smoke` |
| Calibration (OOF) | TabNet isotonic: ECE → **0.0**; RF Platt: ECE ≈ 0.022 |

**L3 — Discovery / Interpretability** (13 luật; Top-5 theo `rank_score`)

| Hạng | Luật | Support | Confidence | Lift |
|-----:|------|--------:|-----------:|-----:|
| 1 | `{ap_lo, smoke} → cardio` | 0.057 | 0.722 | 1.446 |
| 2 | `{active, ap_lo} → cardio` | 0.042 | 0.729 | 1.458 |
| 3 | `{ap_lo} → cardio` | 0.131 | 0.653 | 1.307 |
| 4 | `{age_group, ap_lo} → cardio` | 0.038 | 0.722 | 1.444 |
| 5 | `{ap_lo, weight} → cardio` | 0.036 | 0.713 | 1.427 |

Diễn giải lâm sàng (7 luật chính): [`docs/L3_clinical_interpretation.md`](docs/L3_clinical_interpretation.md).

**Ghi chú:** Vớicross-platform metric L1/L2 tái lập thay đổi trong biên ~0.001; L3 có thể cho kết quả 12–13 luật với **7 luật chính ổn định**. Luật biên nhạy với seed và các phiên bản khác nhau của OS, thư viện.

---
**Author:**
Ngo Thanh Loi, MCS Student at Duy Tan University, Vietnam
Email: <ngothanhloi@dtu.edu.vn>
LinkedIn: https://www.linkedin.com/in/ngo-thanh-loi/