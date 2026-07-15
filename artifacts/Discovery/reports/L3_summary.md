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
- Threshold values range from 0.05 to 0.74 (p80 percentiles)
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
| **Total Rules** | 12 | Rules passing all filters |
| **Mean Support** | 0.055 | 5.5% of samples on average |
| **Mean Confidence** | 0.702 | 70.2% confidence on average |
| **Mean Lift** | 1.406 | 40.6% stronger than random |
| **Min Lift** | 1.241 | All rules have lift > 1.2 |
| **Max Lift** | 1.559 | Strongest rule has 55.9% lift |

## Top Rules

| Antecedents | Consequents | Support | Confidence | Lift | Leverage | Conviction | Rank Score | Stability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ap_lo, smoke | cardio | 0.066 | 0.779 | 1.559 | 0.024 | 2.262 | 0.724 | - |
| age_years, bp_stage | cardio | 0.052 | 0.726 | 1.452 | 0.016 | 1.824 | 0.458 | - |
| active, ap_lo | cardio | 0.044 | 0.776 | 1.552 | 0.016 | 2.229 | 0.424 | - |
| ap_lo | cardio | 0.135 | 0.675 | 1.351 | 0.035 | 1.541 | 0.400 | - |
| ap_lo, cholesterol_category | cardio | 0.035 | 0.712 | 1.424 | 0.010 | 1.736 | 0.160 | - |
| bp_stage, weight | cardio | 0.035 | 0.703 | 1.407 | 0.010 | 1.684 | 0.159 | - |
| alco, bmi | cardio | 0.038 | 0.657 | 1.314 | 0.009 | 1.458 | 0.159 | - |
| bp_stage, cholesterol | cardio | 0.037 | 0.649 | 1.298 | 0.008 | 1.424 | 0.130 | - |
| cholesterol_category, smoke | cardio | 0.032 | 0.644 | 1.290 | 0.007 | 1.407 | 0.046 | - |
| bmi, gender | cardio | 0.031 | 0.713 | 1.428 | 0.009 | 1.745 | 0.018 | - |
| age_group, ap_lo | cardio | 0.030 | 0.776 | 1.553 | 0.011 | 2.234 | 0.000 | - |
| bp_stage | cardio | 0.124 | 0.620 | 1.241 | 0.024 | 1.317 | 0.000 | - |

## Top 5 Rules Tốt Nhất

*Được sắp xếp theo rank_score (cao nhất).*

| Rank | Antecedents | Consequents | Support | Confidence | Lift | Conviction |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | ap_lo, smoke | cardio | 0.066 | 0.779 | 1.559 | 2.262 |
| 2 | age_years, bp_stage | cardio | 0.052 | 0.726 | 1.452 | 1.824 |
| 3 | active, ap_lo | cardio | 0.044 | 0.776 | 1.552 | 2.229 |
| 4 | ap_lo | cardio | 0.135 | 0.675 | 1.351 | 1.541 |
| 5 | ap_lo, cholesterol_category | cardio | 0.035 | 0.712 | 1.424 | 1.736 |

## Top 5 Rules Có Confidence Thấp Nhất (Cần Kiểm Tra Lâm Sàng)

*Các rules này có confidence thấp, cần được xem xét và kiểm tra lâm sàng kỹ lưỡng trước khi áp dụng.*

| Rank | Antecedents | Consequents | Support | Confidence | Lift | Conviction | Ghi Chú |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | bp_stage | cardio | 0.124 | 0.620 | 1.241 | 1.317 | [WARN] Confidence < 65% - Cần kiểm tra lâm sàng |
| 2 | cholesterol_category, smoke | cardio | 0.032 | 0.644 | 1.290 | 1.407 | [WARN] Confidence < 65% - Cần kiểm tra lâm sàng |
| 3 | bp_stage, cholesterol | cardio | 0.037 | 0.649 | 1.298 | 1.424 | [WARN] Confidence < 65% - Cần kiểm tra lâm sàng |
| 4 | alco, bmi | cardio | 0.038 | 0.657 | 1.314 | 1.458 | [WARN] Confidence < 70% - Nên xem xét cẩn thận |
| 5 | ap_lo | cardio | 0.135 | 0.675 | 1.351 | 1.541 | [WARN] Confidence < 70% - Nên xem xét cẩn thận |

### Thống Kê Tổng Quan

- **Tổng số rules:** 12
- **Mean confidence:** 0.702
- **Min confidence:** 0.620
- **Max confidence:** 0.779
- **Mean lift:** 1.406
- **Rules với confidence < 0.65:** 3
- **Rules với confidence < 0.70:** 5
- **Rules với lift >= 1.20:** 12 (đã filter)

### 2.3. Feature Frequency in Rules

**Most Common Features in Antecedents:**

| Feature | Frequency | Clinical Relevance |
|---------|-----------|-------------------|
| `ap_lo` | 5 rules | *[To be filled manually]* |
| `bp_stage` | 4 rules | *[To be filled manually]* |
| `smoke` | 2 rules | *[To be filled manually]* |
| `cholesterol_category` | 2 rules | *[To be filled manually]* |
| `bmi` | 2 rules | *[To be filled manually]* |
| `age_years` | 1 rules | *[To be filled manually]* |
| `active` | 1 rules | *[To be filled manually]* |
| `weight` | 1 rules | *[To be filled manually]* |
| `alco` | 1 rules | *[To be filled manually]* |
| `cholesterol` | 1 rules | *[To be filled manually]* |
| `gender` | 1 rules | *[To be filled manually]* |
| `age_group` | 1 rules | *[To be filled manually]* |

**Insights:**
- **ap_lo** xuất hiện trong 5/12 rules (41.7%)
- **bp_stage** xuất hiện trong 4/12 rules (33.3%)
- **2 features** xuất hiện trong 3+ rules: ap_lo, bp_stage

*Note: Clinical Relevance column requires domain knowledge and should be filled manually.*

## Figures

- [Rule Graph](../figures/rule_graph_topK.png)
- [Feature Support](../figures/feature_support_bar.png)

## Cohort Insights

_No cohort-specific rules generated._
