# Model Risk Validation Toolkit

A validator that reviews any binary classification model the way a bank's model risk team would under the Fed's SR 11-7 guidance. It checks data, performance against a challenger, calibration, stability, robustness, fairness and reproducibility. Each check gets PASS, WARN or FAIL with the measured value and the rule behind it.

I used it on the two finance models in this repo, on their real data. Then I broke the credit model on purpose in six ways to see whether the validator catches each fault.

## Checks

| Area | Check | Rule |
|---|---|---|
| data | missing values | FAIL above 5% in a column |
| data | test rows also in train | WARN above 1%, FAIL above 10% |
| data | single-feature leakage | FAIL if one feature alone has AUC above 0.95 |
| performance | test AUC (or average precision for rare events) | reported with a bootstrap 95% interval |
| performance | beats the challenger | FAIL if the challenger is clearly better, WARN if the interval includes 0 |
| performance | overfitting | train minus test: WARN above 0.03, FAIL above 0.10 |
| calibration | observed / expected events | WARN outside 0.9-1.1, FAIL outside 0.8-1.2 |
| calibration | Hosmer-Lemeshow test | WARN if p < 0.05, FAIL if p < 0.001 |
| calibration | calibration error by decile | WARN above 0.02, FAIL above 0.05 |
| stability | score PSI and worst feature PSI, train vs test | WARN above 0.10, FAIL above 0.25 |
| robustness | score change under 1% input noise | WARN above 0.01, FAIL above 0.05 |
| robustness | monotone in each named feature | FAIL if more than 5% of rows break it |
| fairness | adverse impact ratio for each protected attribute | four-fifths rule: FAIL below 0.8 |
| reproducibility | retrain on 3 bootstrap samples | WARN if the spread is above 0.01, FAIL above 0.03 |

The validator only needs a `ModelUnderTest`: a predict function, the train and test data, and optionally a challenger, expected monotone directions, protected attributes and a retrain function. So it isn't tied to these two models.

## Does it catch real faults?

I broke the credit default model one fault at a time and ran the validator each time. Results are in `results/fault_injection.json`.

| Fault I injected | Check that should catch it | Result |
|---|---|---|
| A column computed from the outcome added as a feature | single-feature leakage | **FAIL** (caught) |
| Test rows copied into the training data | test rows also in train | **FAIL** (caught) |
| Test customers given 60% higher bills and 40% lower limits | feature PSI | **FAIL** (caught). Score PSI only warned |
| Probabilities inflated by 60% | observed / expected events | **FAIL** (caught) |
| Unconstrained model reviewed against monotone expectations | monotone checks | **FAIL** on 4 of 5 features (caught) |
| 0.25 added to the risk of applicants aged 25 and under | adverse impact ratio by age | **FAIL** (caught) |

All 6 of 6 faults were caught. The drift case is the instructive one: score PSI only warned, because the score depends mostly on repayment history, and that didn't move. The validator still caught the fault through the per-feature PSI on bills and limits. A score-only drift monitor would have missed it.

## Findings on the real models

The full reports are in `reports/`.

**Credit default model** (`reports/credit_default.md`): 1 FAIL, 3 WARN, 17 PASS.

- It beats the logistic challenger by +0.020 AUC (95% CI +0.011 to +0.029). Train-test gap is 0.021. Score PSI is 0.003. All 5 monotone checks pass, and the lowest adverse impact ratio is 0.92 (age 25 and under).
- **FAIL:** the Hosmer-Lemeshow test (p = 0.0004). Overall calibration is fine (observed / expected 1.048), but some deciles are off, and the calibration error by decile is 0.023, which is a WARN. My recommendation: recalibrate on the validation set (isotonic or Platt) before the scores are used as probabilities, for example for expected loss.
- **WARN:** 3.2% of test rows exactly match a training row. I checked why: 119 of the 140 matching rows have zero bills and zero payments in all six months, so they are real customers with identical (unused) histories. It's not a leak.
- **WARN:** the median score change under 1% input noise is 0.011, just over the line.

**Card fraud model** (`reports/card_fraud.md`): 3 FAIL, 2 WARN, 8 PASS.

- **FAIL, a real problem:** train average precision is 0.999 against 0.768 on test. The trees memorise the 360 training frauds. Test performance still beats the challenger (+0.077 average precision, 95% CI +0.017 to +0.153), but the model would likely do better with stronger regularisation.
- **FAIL, a real problem:** the `hour` feature has PSI 4.4. The data covers only two days, and the test period is hours 40 to 48, so it sees a different part of the day than training. Hour of day can't be learned from two days, and I'd drop it. 17 features drift above 0.10, which is a reason to retrain often.
- **FAIL, a false alarm I'd overrule:** single-feature leakage, because V14 alone has AUC 0.956. V14 is a PCA component of the bank's own features, and it is also one of the fraud model's own top SHAP reasons. A strong real signal isn't leakage. It shows that a rule like this has to be reviewed by a person.
- **WARN:** retraining on bootstrap samples gives average precision from 0.753 to 0.779. With 75 test frauds, that spread is mostly noise.

So the validator found 2 real problems in a model I had already shipped in this repo (overfitting and the hour feature). It also produced 1 false alarm that a reviewer has to overrule with a reason.

## Running it

```bash
pip install xgboost scikit-learn scipy pandas pyarrow requests
python run.py             # both reports and the fault injection, under a minute on a 4-core CPU
pytest -q test_checks.py  # each check flags its fault, on the real credit data
```
