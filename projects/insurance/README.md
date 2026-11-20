# Insurance projects

| Project | What it does | Key result |
|---|---|---|
| [Motor Insurance Pricing with Discrimination-Free Premiums](motor-pricing-fairness) | Frequency and severity pricing models, and what removing driver age really changes | 678k real policies; GBM Gini 0.361 vs GLM 0.279 |
| [Motor Claim Severity Model with Extreme-Value Large-Loss Analysis](claims-severity-large-loss) | Claim size models below a cap, a Pareto tail fit above it, and a large-loss load | 1.8% of claims carry 34.5% of cost; tail shape 1.07 |
| [AI Governance Report for Insurance Models under State AI Rules](ai-governance-report) | Requirements register from the NAIC bulletin, New York CL 7 and Colorado Reg. 10-1-1, and an evidence check of my models against it | 118 requirements, 34 of 40 theme tags right; 30 jurisdictions adopted the NAIC bulletin |
| [Life Underwriting Mortality Model on Real Health Exams](life-underwriting-risk) | 8-year death risk from the application, a nurse visit or a blood test, on NHANES with National Death Index records; what fast-tracking without labs costs | AUC 0.80 application only vs 0.83 with labs; calibration checked by race and income |
| [Subrogation Opportunity Detection from Crash Reports](subrogation-detection) | Flags recoverable auto claims at first notice of loss, on 147,290 claims built from 2023 NYPD crash reports, against the classic parked/rear-ended rule | 96% vs 89% precision at the rule's volume; 63% of recoverable claims found at 30% referred vs 37% |

[Back to all projects](../../README.md)
