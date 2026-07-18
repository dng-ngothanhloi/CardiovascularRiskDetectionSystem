# Test Prediction Verification Report

**Ngày tạo:** 2025-11-07

Báo cáo này so sánh dự báo của models với ground truth trên tập test.

---

## 1. Summary Statistics

### RandomForest

- **Total samples:** 20
- **Correct predictions:** 15 (75.0%)
- **Accuracy:** 0.7500
- **Precision:** 0.6667
- **Recall:** 0.7500
- **F1:** 0.7059

### TabNet

- **Total samples:** 20
- **Correct predictions:** 14 (70.0%)
- **Accuracy:** 0.7000
- **Precision:** 0.6000
- **Recall:** 0.7500
- **F1:** 0.6667

## 2. Detailed Predictions - RandomForest

| Sample ID | y_true | y_pred_prob | y_pred_binary | Correct |
|-----------|--------|-------------|---------------|----------|
| 2900 | 1 | 0.8837 | 1 | [OK] |
| 11830 | 0 | 0.8667 | 1 | [X] |
| 13420 | 1 | 0.8485 | 1 | [OK] |
| 10739 | 0 | 0.8465 | 1 | [X] |
| 5282 | 1 | 0.8336 | 1 | [OK] |
| 5733 | 1 | 0.7106 | 1 | [OK] |
| 4649 | 1 | 0.6838 | 1 | [OK] |
| 7310 | 0 | 0.6433 | 1 | [X] |
| 5538 | 1 | 0.5540 | 1 | [OK] |
| 8231 | 0 | 0.4547 | 0 | [OK] |
| 2791 | 0 | 0.3660 | 0 | [OK] |
| 4045 | 0 | 0.3590 | 0 | [OK] |
| 3143 | 0 | 0.3251 | 0 | [OK] |
| 13405 | 0 | 0.2563 | 0 | [OK] |
| 10175 | 0 | 0.2442 | 0 | [OK] |
| 7653 | 1 | 0.2385 | 0 | [X] |
| 8045 | 1 | 0.1590 | 0 | [X] |
| 169 | 0 | 0.1147 | 0 | [OK] |
| 8610 | 0 | 0.0895 | 0 | [OK] |
| 3855 | 0 | 0.0444 | 0 | [OK] |

## 3. Detailed Predictions - TabNet

| Sample ID | y_true | y_pred_prob | y_pred_binary | Correct |
|-----------|--------|-------------|---------------|----------|
| 10739 | 0 | 0.8856 | 1 | [X] |
| 2900 | 1 | 0.8690 | 1 | [OK] |
| 13420 | 1 | 0.8656 | 1 | [OK] |
| 11830 | 0 | 0.8642 | 1 | [X] |
| 5282 | 1 | 0.8364 | 1 | [OK] |
| 5733 | 1 | 0.7001 | 1 | [OK] |
| 4649 | 1 | 0.6322 | 1 | [OK] |
| 7310 | 0 | 0.6317 | 1 | [X] |
| 5538 | 1 | 0.5314 | 1 | [OK] |
| 8231 | 0 | 0.5001 | 1 | [X] |
| 3143 | 0 | 0.4644 | 0 | [OK] |
| 4045 | 0 | 0.3982 | 0 | [OK] |
| 2791 | 0 | 0.3688 | 0 | [OK] |
| 7653 | 1 | 0.2899 | 0 | [X] |
| 8045 | 1 | 0.2614 | 0 | [X] |
| 13405 | 0 | 0.2598 | 0 | [OK] |
| 10175 | 0 | 0.2570 | 0 | [OK] |
| 169 | 0 | 0.1184 | 0 | [OK] |
| 8610 | 0 | 0.0922 | 0 | [OK] |
| 3855 | 0 | 0.0429 | 0 | [OK] |

## 4. Feature Values (Top 10 Samples by Prediction Probability)

### RandomForest

| Sample ID | gender | height | weight | ap_hi | ap_lo | cholesterol | gluc | smoke | alco | active | y_true | y_pred_prob |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2900 | 1.00 | 158.00 | 58.00 | 150.00 | 90.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1 | 0.8837 |
| 11830 | 1.00 | 164.00 | 117.00 | 170.00 | 100.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0 | 0.8667 |
| 13420 | 2.00 | 172.00 | 78.00 | 150.00 | 100.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1 | 0.8485 |
| 10739 | 1.00 | 140.00 | 78.00 | 170.00 | 90.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0 | 0.8465 |
| 5282 | 1.00 | 169.00 | 75.00 | 140.00 | 90.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1 | 0.8336 |
| 5733 | 2.00 | 164.00 | 72.00 | 130.00 | 90.00 | 2.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1 | 0.7106 |
| 4649 | 1.00 | 159.00 | 65.00 | 130.00 | 90.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1 | 0.6838 |
| 7310 | 1.00 | 159.00 | 58.00 | 120.00 | 80.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0 | 0.6433 |
| 5538 | 1.00 | 163.00 | 85.00 | 130.00 | 80.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1 | 0.5540 |
| 8231 | 1.00 | 150.00 | 70.00 | 100.00 | 70.00 | 2.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0 | 0.4547 |

### TabNet

| Sample ID | gender | height | weight | ap_hi | ap_lo | cholesterol | gluc | smoke | alco | active | y_true | y_pred_prob |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10739 | 1.00 | 140.00 | 78.00 | 170.00 | 90.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0 | 0.8856 |
| 2900 | 1.00 | 158.00 | 58.00 | 150.00 | 90.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1 | 0.8690 |
| 13420 | 2.00 | 172.00 | 78.00 | 150.00 | 100.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1 | 0.8656 |
| 11830 | 1.00 | 164.00 | 117.00 | 170.00 | 100.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0 | 0.8642 |
| 5282 | 1.00 | 169.00 | 75.00 | 140.00 | 90.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1 | 0.8364 |
| 5733 | 2.00 | 164.00 | 72.00 | 130.00 | 90.00 | 2.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1 | 0.7001 |
| 4649 | 1.00 | 159.00 | 65.00 | 130.00 | 90.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1 | 0.6322 |
| 7310 | 1.00 | 159.00 | 58.00 | 120.00 | 80.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0 | 0.6317 |
| 5538 | 1.00 | 163.00 | 85.00 | 130.00 | 80.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1 | 0.5314 |
| 8231 | 1.00 | 150.00 | 70.00 | 100.00 | 70.00 | 2.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0 | 0.5001 |

## 5. Analysis

### Correct Predictions

- **RF:** 15/20 (75.0%)
- **TabNet:** 14/20 (70.0%)

### Incorrect Predictions (False Positives & False Negatives)

- **RF False Positives:** 3
- **RF False Negatives:** 2
- **TabNet False Positives:** 4
- **TabNet False Negatives:** 2

## 6. Probability Spectrum Visualization

![Probability Spectrum](probability_spectrum.png)

*Hình 6.1: Biểu đồ phân phối probability cho các nhóm dự đoán. TP (True Positive), TN (True Negative), FP (False Positive), FN (False Negative - sai nghiêm trọng). Đường đứt nét đen: Threshold = 0.5. Đường chấm chấm: Ideal calibration line.*

## 7. Correct Predictions - Detailed Tables

### 7.1. True Positives (Correctly Predicted Disease) - RandomForest

| Sample ID | y_true | RF Prob | TabNet Prob | Key Features | Lifestyle |
|-----------|--------|---------|-------------|--------------|----------|
| 2900 | 1 | 0.8837 | 0.8690 | N/A | N/A |
| 13420 | 1 | 0.8485 | 0.8656 | N/A | N/A |
| 5282 | 1 | 0.8336 | 0.8364 | N/A | N/A |
| 5733 | 1 | 0.7106 | 0.7001 | N/A | N/A |
| 4649 | 1 | 0.6838 | 0.6322 | N/A | N/A |
| 5538 | 1 | 0.5540 | 0.5314 | N/A | N/A |

### 7.2. True Negatives (Correctly Predicted No Disease) - RandomForest

| Sample ID | y_true | RF Prob | TabNet Prob | Key Features | Lifestyle |
|-----------|--------|---------|-------------|--------------|----------|
| 8231 | 0 | 0.4547 | 0.5001 | N/A | N/A |
| 2791 | 0 | 0.3660 | 0.3688 | ap_hi=140, ap_lo=90, age=57, BMI=35.1, chol=1, bp_stage=3 | smoke=1, alco=0, active=1, risk=0.5 |
| 4045 | 0 | 0.3590 | 0.3982 | N/A | N/A |
| 3143 | 0 | 0.3251 | 0.4644 | N/A | N/A |
| 13405 | 0 | 0.2563 | 0.2598 | N/A | N/A |
| 10175 | 0 | 0.2442 | 0.2570 | N/A | N/A |
| 169 | 0 | 0.1147 | 0.1184 | N/A | N/A |
| 8610 | 0 | 0.0895 | 0.0922 | ap_hi=140, ap_lo=90, age=54, BMI=31.1, chol=1, bp_stage=3 | smoke=0, alco=0, active=1, risk=-0.5 |
| 3855 | 0 | 0.0444 | 0.0429 | N/A | N/A |

## 8. Incorrect Predictions - Detailed Tables

### 8.1. False Positives (Predicted: 1, Actual: 0) - RandomForest

| Sample ID | y_true | RF Prob | TabNet Prob | Key Features | Lifestyle |
|-----------|--------|---------|-------------|--------------|----------|
| 11830 | 0 | 0.8667 | 0.8642 | N/A | N/A |
| 10739 | 0 | 0.8465 | 0.8856 | ap_hi=130, ap_lo=90, age=57, BMI=19.8, chol=2 | smoke=0, alco=0, active=1, risk=-0.5 |
| 7310 | 0 | 0.6433 | 0.6317 | N/A | N/A |

### 8.2. False Negatives (Predicted: 0, Actual: 1) - RandomForest

| Sample ID | y_true | RF Prob | TabNet Prob | Key Features | Lifestyle |
|-----------|--------|---------|-------------|--------------|----------|
| 7653 | 1 | 0.2385 | 0.2899 | N/A | N/A |
| 8045 | 1 | 0.1590 | 0.2614 | N/A | N/A |

---

**Báo cáo được tạo tự động từ test predictions.**
