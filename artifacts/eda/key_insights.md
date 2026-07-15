## 1. Dataset Overview
- Total records: 70,000
- Features: 19 (excluding target)
- Target variable: cardio (binary)

## 2. Class Distribution
- No Disease: 50.0%
- Disease: 50.0%

## 5. Lifestyle Factors
- Smoking increases risk: 47.5% vs 50.2%
- Alcohol increases risk: 48.4% vs 50.1%
- Physical activity reduces risk: 53.6% vs 49.1%

## 6. Recommendations for Preprocessing & Feature Engineering
- [WARN] Derived variables (age_years, BMI, pulse_pressure) should be created in feature_engineering.py
- [OK] Flag data quality issues: ap_lo > ap_hi, extreme BP values
- [OK] Consider feature interactions: smoke x active, BMI x age
- [OK] Use stratified cross-validation due to potential class imbalance
- [OK] Focus on top MI features: age, BMI, pulse_pressure, cholesterol, glucose