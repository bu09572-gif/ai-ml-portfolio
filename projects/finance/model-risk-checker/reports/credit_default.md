# Validation report: Credit default model (monotone XGBoost)

20,991 training rows, 4,410 test rows, 1,016 test events. Result: 1 FAIL, 3 WARN, 17 PASS.

| Area | Check | Status | Value | Rule |
|---|---|---|---|---|
| data | missing values | **PASS** | max 0.0% in one column | FAIL above 5% in any column |
| data | test rows also in train | **WARN** | 3.17% | WARN above 1%, FAIL above 10%. Some exact matches are real (e.g. unused accounts) |
| data | single-feature leakage | **PASS** | months_delayed: AUC 0.728 | FAIL if one feature alone has AUC above 0.95 |
| performance | test AUC | **PASS** | 0.776 (95% CI 0.757-0.794) | reported with a bootstrap interval |
| performance | beats the challenger model | **PASS** | difference +0.020 (95% CI +0.011 to +0.029) | FAIL if the challenger is clearly better, WARN if the interval includes 0 |
| performance | overfitting (train minus test) | **PASS** | train 0.797, gap +0.021 | WARN above 0.03, FAIL above 0.10 |
| calibration | observed / expected events | **PASS** | 1.048 | WARN outside 0.9-1.1, FAIL outside 0.8-1.2 |
| calibration | Hosmer-Lemeshow test | **FAIL** | chi2 28.6, p = 0.000376 | WARN if p < 0.05, FAIL if p < 0.001 |
| calibration | calibration error by decile | **WARN** | 0.0232 | WARN above 0.02, FAIL above 0.05 |
| stability | score PSI (train vs test) | **PASS** | 0.003 | WARN above 0.10, FAIL above 0.25 |
| stability | feature PSI (worst feature) | **PASS** | paid_may: 0.004; 0 features above 0.10 | WARN above 0.10, FAIL above 0.25 |
| robustness | score change under 1% noise | **WARN** | median 0.0111 | WARN above 0.01, FAIL above 0.05 |
| robustness | monotone in delay_sep (+) | **PASS** | 0.0% of rows break it | FAIL above 5% of rows |
| robustness | monotone in max_delay (+) | **PASS** | 0.0% of rows break it | FAIL above 5% of rows |
| robustness | monotone in utilization_sep (+) | **PASS** | 0.0% of rows break it | FAIL above 5% of rows |
| robustness | monotone in limit (-) | **PASS** | 0.0% of rows break it | FAIL above 5% of rows |
| robustness | monotone in paid_sep (-) | **PASS** | 0.0% of rows break it | FAIL above 5% of rows |
| fairness | adverse impact ratio by sex | **PASS** | 0.953 (lowest: male) | four-fifths rule: FAIL below 0.8, WARN below 0.9 |
| fairness | adverse impact ratio by age_group | **PASS** | 0.923 (lowest: <=25) | four-fifths rule: FAIL below 0.8, WARN below 0.9 |
| fairness | adverse impact ratio by marriage | **PASS** | 0.997 (lowest: married) | four-fifths rule: FAIL below 0.8, WARN below 0.9 |
| reproducibility | retrain with 3 seeds | **PASS** | 0.772 to 0.774 | WARN if the spread is above 0.01, FAIL above 0.03 |
