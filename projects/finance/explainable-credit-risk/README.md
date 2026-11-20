# Explainable Credit Default Model with Fairness Checks

This project predicts which credit card holders will default next month, approves or declines them, and gives every declined applicant plain-language reasons, like an adverse action notice. It runs on real data from a Taiwanese bank. Sex, marital status, age and education are kept out of the model. I then test whether its decisions still differ by sex, age or marital status.

## Data

The UCI "default of credit card clients" dataset (Yeh & Lien, 2009) has 30,000 card holders from April to September 2005, and 22.1% defaulted the next month. Each row has the credit limit, six months of repayment status (on time, or 1 to 9 months late), six monthly bills and six monthly payments, plus sex, education, marital status and age. `data.py` downloads a fixed copy (OpenML 42477, saved as a release asset of this repo) and checks its SHA-256.

The split is fixed and random: 70% train, 15% validation (to set the cutoff), 15% test (4,410 applicants, 23.0% default). The data covers only one period, so there is no later period to hold out.

**Model inputs:** credit limit, the 18 monthly history fields, September utilization, the September payment ratio, months late out of 6, and the worst delay.

## Models

| | What it is |
|---|---|
| logistic | logistic regression on log-scaled inputs: the usual scorecard-style baseline |
| xgb_monotone | XGBoost with monotone constraints: predicted risk can only go up with payment delays and utilization, and only go down with a higher limit or larger payments |
| xgb_free | the same XGBoost without constraints, to see what the constraints cost |

**Decision rule:** approve the 80% of applicants with the lowest predicted risk. The cutoff is set on validation and then fixed for test.

## Results (4,410 test applicants)

From `results/metrics.json`.

| | Logistic | XGBoost monotone | XGBoost free |
|---|---|---|---|
| AUC | 0.755 | 0.776 | **0.777** |
| KS | 0.410 | 0.416 | **0.419** |
| Brier score | 0.1444 | 0.1384 | **0.1383** |
| Calibration error (by decile) | **0.023** | **0.023** | 0.029 |
| Approved | 81.2% | 80.4% | 80.7% |
| Default rate among approved | 15.2% | **14.1%** | 14.1% |
| Share of defaulters declined | 46.4% | **51.0%** | 50.5% |

- **The constraints are almost free.** The monotone model gives up 0.001 AUC against the free model and is better calibrated. In exchange, an applicant can never get a worse score by paying more or being late less often, and that is what a model reviewer will ask about first.
- **Both XGBoost models beat logistic regression.** At the same approval rate they catch about 5 points more of the defaulters.
- **AUC is in the usual range for this dataset.** Most of the signal is in the repayment history. The top features by mean |SHAP| are worst delay (0.317), September status (0.312) and months late (0.225).

## Fairness (XGBoost monotone, test set)

**Adverse impact ratio** = a group's approval rate divided by the highest group's rate. The four-fifths rule flags anything under 0.8.

| Group | Applicants | Approved | Adverse impact ratio | Approved, among people who did not default | Default rate if approved | Predicted vs actual default |
|---|---|---|---|---|---|---|
| female | 2,646 | 81.9% | 1.00 | 90.6% | 12.9% | 21.2% / 21.2% |
| male | 1,764 | 78.0% | 0.95 | 88.4% | 15.8% | 23.1% / 25.7% |
| age 25 and under | 562 | 75.3% | **0.90** | 86.0% | 14.2% | 26.0% / 24.9% |
| age 26-40 | 2,628 | 81.5% | 0.98 | 90.4% | 13.6% | 21.1% / 22.1% |
| age 41-60 | 1,190 | 80.1% | 0.96 | 90.0% | 14.8% | 22.1% / 24.2% |
| married | 2,065 | 80.1% | 0.97 | 89.5% | 14.9% | 21.9% / 23.8% |
| single | 2,282 | 80.5% | 0.98 | 90.1% | 13.1% | 22.1% / 22.3% |

The 60+ group (30 people) and "other" marital status (63 people) are in `metrics.json` but are too small to read.

- **No group falls under 0.8.** The lowest ratio is 0.90, for applicants aged 25 and under.
- **Young applicants are the group to watch.** Their predicted default (26.0%) is a bit above their actual rate (24.9%). Among people who repaid, they were approved less often (86.0% vs 90.4%). Age is not a model input, so this comes from features that differ with age, such as lower limits and higher utilization.
- **Men are under-predicted, not over-predicted.** Their predicted default is 23.1% against an actual 25.7%. Men who are approved also default more (15.8% vs 12.9%). So the model is not harsher on men than their record supports.

## Decline reasons

For each declined applicant, `credit.reasons` takes the features with the largest positive SHAP values. It maps them to plain-language reasons, keeps one reason per family, and gives at most 4. Late-payment features are one family, bills and utilization are another, and so on. Here are three real declined test applicants, from `results/reasons_examples.json`:

> **Applicant 3** (predicted 47.6%, did default): Late payments: 4 of the last 6 months, up to 2 months late · Credit limit is low · Amount paid is low compared with the amount owed · High balance compared with the credit limit (77% used in September)
>
> **Applicant 7** (predicted 33.6%, did default): Little or no card use in September · Amount paid is low compared with the amount owed · Late payments: 1 of the last 6 months, up to 1 month late · Credit limit is low
>
> **Applicant 10** (predicted 47.6%, did not default): Late payments: 1 of the last 6 months, up to 2 months late · Little or no card use in September · Amount paid is low compared with the amount owed

All 866 declined applicants got at least one reason. Late payments was the main reason for 786 of them, and "little or no card use" was the main reason for the other 80. The model learned from this data that an inactive card is a risk signal. That reason is the one I'd want a credit policy person to sign off on before it appears on a real notice.

## Running it

```bash
pip install xgboost scikit-learn pandas pyarrow requests
python run.py              # a few seconds on a laptop CPU
pytest -q test_credit.py   # data checks, monotone behaviour, reason rules, impact ratio
```
