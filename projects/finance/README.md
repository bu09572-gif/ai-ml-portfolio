# Finance projects

| Project | What it does | Key result |
|---|---|---|
| [Real-Time Card Fraud Scoring with SHAP Explanations](realtime-fraud-scoring) | XGBoost fraud API that returns a score, an alert and the top 3 reasons | 73% of fraud caught at 80% precision; 9 ms median latency |
| [Explainable Credit Default Model with Fairness Checks](explainable-credit-risk) | Monotone XGBoost with plain-language decline reasons and four-fifths rule checks | AUC 0.776; every adverse impact ratio at or above 0.90 |
| [Model Risk Validation Toolkit](model-risk-checker) | SR 11-7 style validation: leakage, challenger, calibration, drift, monotonicity, fairness | Caught 6 of 6 injected faults and 2 real issues in the fraud model |
| [SEC Filing Analyst](sec-filing-analyst) | Finds each input number on its page and does the math in Python, never in the model | 40% of calculator answers within 1%; 18% overall vs 20% for plain Q&A, with the reasons |
| [Anti-Money-Laundering Detection on a Real Bitcoin Transaction Graph](graph-aml) | Flags illicit transactions with graph features, a GCN, and a test of what happens when criminal behaviour changes | F1 0.826 (precision 0.96); recall falls from 86% to 2% after a dark market shutdown |

[Back to all projects](../../README.md)
