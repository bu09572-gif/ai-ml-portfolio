# Validation report: Card fraud model (XGBoost)

170,885 training rows, 56,961 test rows, 75 test events. Result: 3 FAIL, 2 WARN, 8 PASS.

| Area | Check | Status | Value | Rule |
|---|---|---|---|---|
| data | missing values | **PASS** | max 0.0% in one column | FAIL above 5% in any column |
| data | test rows also in train | **PASS** | 0.00% | WARN above 1%, FAIL above 10%. Some exact matches are real (e.g. unused accounts) |
| data | single-feature leakage | **FAIL** | V14: AUC 0.956 | FAIL if one feature alone has AUC above 0.95 |
| performance | test average precision | **PASS** | 0.768 (95% CI 0.664-0.861) | reported with a bootstrap interval |
| performance | beats the challenger model | **PASS** | difference +0.077 (95% CI +0.017 to +0.153) | FAIL if the challenger is clearly better, WARN if the interval includes 0 |
| performance | overfitting (train minus test) | **FAIL** | train 0.999, gap +0.230 | WARN above 0.03, FAIL above 0.10 |
| calibration | observed / expected events | **PASS** | 1.013 | WARN outside 0.9-1.1, FAIL outside 0.8-1.2 |
| calibration | Hosmer-Lemeshow test | **WARN** | chi2 17.7, p = 0.0237 | WARN if p < 0.05, FAIL if p < 0.001 |
| calibration | calibration error by decile | **PASS** | 0.0001 | WARN above 0.02, FAIL above 0.05 |
| stability | score PSI (train vs test) | **PASS** | 0.050 | WARN above 0.10, FAIL above 0.25 |
| stability | feature PSI (worst feature) | **FAIL** | hour: 4.433; 17 features above 0.10 | WARN above 0.10, FAIL above 0.25 |
| robustness | score change under 1% noise | **PASS** | median 0.0000 | WARN above 0.01, FAIL above 0.05 |
| reproducibility | retrain with 3 seeds | **WARN** | 0.753 to 0.779 | WARN if the spread is above 0.01, FAIL above 0.03 |
