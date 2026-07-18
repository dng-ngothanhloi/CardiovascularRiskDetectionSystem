# L3 Discovery Summary

## 1. Methodology & Settings

### 1.1. Discovery Pipeline Configuration

**Input Data:**
- **Source:** TabNet attention masks từ 5-fold cross-validation
- **Total samples:** 55,993
- **Features:** 19 features (demographic, clinical, lifestyle)
- **Target:** `cardio` (binary: 0=no disease, 1=disease)

**Attention-to-Itemset Conversion:**
- **Aggregation:** Mean across decision steps
- **Normalization:** Per-feature max normalization
- **Threshold (τ):** p80 (percentile 80) - dynamic threshold per feature
- **Min active features:** 1

**Apriori Algorithm Settings:**
- **Min support:** 0.03 (3% of samples)
- **Min confidence:** 0.6 (60%)
- **Max length:** 4 (maximum 4 items per rule)
- **Metric:** Confidence

**Post-processing:**
- **Min lift:** 1.2 (rules must have at least 20% stronger association than random)
- **Redundancy pruning:** Enabled
- **Bootstrap stability:** Disabled (for faster execution)
- **Top-K:** 200 rules retained

### 1.2. Data Quality

**Binarized Itemset:**
- **Shape:** 55,993 samples × 23 columns (19 features + metadata)
- **Cardio distribution:**
  - `cardio=0`: 5,017 (50.2%)
  - `cardio=1`: 4,983 (49.8%)
- **Class balance:** [OK] Perfectly balanced (50/50)

**Feature Coverage:**
- All 19 features successfully binarized
- Threshold values range from 0.07 to 0.65 (p80 percentiles)
- No missing values in binarized itemset

---

## Settings

| Setting | Value |
| --- | --- |
| Tau | p80 |
| Aggregation | mean |
| Normalisation | per_feature_max |
| Min Active Features | 1 |
| Apriori Min Support | 0.03 |
| Apriori Min Confidence | 0.6 |
| Apriori Max Len | 4 |
| Postprocess Min Lift | 1.2 |
| Postprocess Min Confidence | 0.6 |
| Postprocess Min Support | 0.03 |
| Redundancy Pruning | Enabled |
| Bootstrap | Disabled |
| Cohort Mining | Enabled |

## 2. Discovered Rules Analysis

### 2.1. Overall Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Total Rules** | 13 | Rules passing all filters |
| **Mean Support** | 0.050 | 5.0% of samples on average |
| **Mean Confidence** | 0.700 | 70.0% confidence on average |
| **Mean Lift** | 1.401 | 40.1% stronger than random |
| **Min Lift** | 1.218 | All rules have lift > 1.2 |
| **Max Lift** | 1.551 | Strongest rule has 55.1% lift |

## Top Rules

| Antecedents | Consequents | Support | Confidence | Lift | Leverage | Conviction | Rank Score | Stability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ap_lo, smoke | cardio | 0.057 | 0.722 | 1.446 | 0.018 | 1.803 | 0.519 | - |
| active, ap_lo | cardio | 0.042 | 0.729 | 1.458 | 0.013 | 1.844 | 0.349 | - |
| ap_lo | cardio | 0.131 | 0.653 | 1.307 | 0.031 | 1.442 | 0.314 | - |
| age_group, ap_lo | cardio | 0.038 | 0.722 | 1.444 | 0.012 | 1.798 | 0.258 | - |
| ap_lo, weight | cardio | 0.036 | 0.713 | 1.427 | 0.011 | 1.743 | 0.181 | - |
| age_group, smoke | cardio | 0.035 | 0.705 | 1.411 | 0.010 | 1.696 | 0.175 | - |
| pulse_pressure, weight | cardio | 0.034 | 0.670 | 1.341 | 0.009 | 1.516 | 0.106 | - |
| age_years, bp_stage | cardio | 0.033 | 0.686 | 1.373 | 0.009 | 1.595 | 0.087 | - |
| cholesterol, gender | cardio | 0.032 | 0.647 | 1.295 | 0.007 | 1.418 | 0.050 | - |
| bmi, gender | cardio | 0.032 | 0.755 | 1.511 | 0.011 | 2.044 | 0.032 | - |
| bp_stage, weight | cardio | 0.031 | 0.713 | 1.427 | 0.009 | 1.743 | 0.018 | - |
| ap_lo, smoke, weight | cardio | 0.031 | 0.775 | 1.551 | 0.011 | 2.222 | 0.000 | - |
| smoke | cardio | 0.122 | 0.609 | 1.218 | 0.022 | 1.278 | 0.000 | - |

## Top 5 Rules Tốt Nhất

*Được sắp xếp theo rank_score (cao nhất).*

| Rank | Antecedents | Consequents | Support | Confidence | Lift | Conviction |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | ap_lo, smoke | cardio | 0.057 | 0.722 | 1.446 | 1.803 |
| 2 | active, ap_lo | cardio | 0.042 | 0.729 | 1.458 | 1.844 |
| 3 | ap_lo | cardio | 0.131 | 0.653 | 1.307 | 1.442 |
| 4 | age_group, ap_lo | cardio | 0.038 | 0.722 | 1.444 | 1.798 |
| 5 | ap_lo, weight | cardio | 0.036 | 0.713 | 1.427 | 1.743 |

## Top 5 Rules Có Confidence Thấp Nhất (Cần Kiểm Tra Lâm Sàng)

*Các rules này có confidence thấp, cần được xem xét và kiểm tra lâm sàng kỹ lưỡng trước khi áp dụng.*

| Rank | Antecedents | Consequents | Support | Confidence | Lift | Conviction | Ghi Chú |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | smoke | cardio | 0.122 | 0.609 | 1.218 | 1.278 | [WARN] Confidence < 65% - Cần kiểm tra lâm sàng |
| 2 | cholesterol, gender | cardio | 0.032 | 0.647 | 1.295 | 1.418 | [WARN] Confidence < 65% - Cần kiểm tra lâm sàng |
| 3 | ap_lo | cardio | 0.131 | 0.653 | 1.307 | 1.442 | [WARN] Confidence < 70% - Nên xem xét cẩn thận |
| 4 | pulse_pressure, weight | cardio | 0.034 | 0.670 | 1.341 | 1.516 | [WARN] Confidence < 70% - Nên xem xét cẩn thận |
| 5 | age_years, bp_stage | cardio | 0.033 | 0.686 | 1.373 | 1.595 | [WARN] Confidence < 70% - Nên xem xét cẩn thận |

### Thống Kê Tổng Quan

- **Tổng số rules:** 13
- **Mean confidence:** 0.700
- **Min confidence:** 0.609
- **Max confidence:** 0.775
- **Mean lift:** 1.401
- **Rules với confidence < 0.65:** 2
- **Rules với confidence < 0.70:** 5
- **Rules với lift >= 1.20:** 13 (đã filter)

### 2.3. Feature Frequency in Rules

**Most Common Features in Antecedents:**

| Feature | Frequency | Clinical Relevance |
|---------|-----------|-------------------|
| `ap_lo` | 6 rules | *[To be filled manually]* |
| `smoke` | 4 rules | *[To be filled manually]* |
| `weight` | 4 rules | *[To be filled manually]* |
| `age_group` | 2 rules | *[To be filled manually]* |
| `bp_stage` | 2 rules | *[To be filled manually]* |
| `gender` | 2 rules | *[To be filled manually]* |
| `active` | 1 rules | *[To be filled manually]* |
| `pulse_pressure` | 1 rules | *[To be filled manually]* |
| `age_years` | 1 rules | *[To be filled manually]* |
| `cholesterol` | 1 rules | *[To be filled manually]* |
| `bmi` | 1 rules | *[To be filled manually]* |

**Insights:**
- **ap_lo** xuất hiện trong 6/13 rules (46.2%)
- **smoke** xuất hiện trong 4/13 rules (30.8%)
- **3 features** xuất hiện trong 3+ rules: ap_lo, smoke, weight

*Note: Clinical Relevance column requires domain knowledge and should be filled manually.*

## Figures

- [Rule Graph](../figures/rule_graph_topK.png)
- [Feature Support](../figures/feature_support_bar.png)

## Cohort Insights

_No cohort-specific rules generated._
