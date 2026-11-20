"""Credit default models and adverse action reasons.

  logistic       standardized logistic regression (log amounts), the usual scorecard-style baseline
  xgb_monotone   XGBoost where risk can only go up with payment delays and utilization, and only
                 go down with credit limit and payments made. Easier to defend in a model review,
                 and a declined applicant can't get a worse score by paying more.
  xgb_free       XGBoost with no constraints, to measure what the constraints cost

Reasons: for each declined applicant, the features with the largest positive SHAP values (they
pushed the risk up the most), grouped into plain-language reasons like an adverse action notice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

MONTH_NAME = {"sep": "September", "aug": "August", "jul": "July", "jun": "June", "may": "May", "apr": "April"}


def monotone(cols: list[str]) -> tuple[int, ...]:
    def sign(c: str) -> int:
        if c.startswith("delay_") or c in ("max_delay", "months_delayed", "utilization_sep"):
            return 1
        if c.startswith("paid_") or c == "limit":
            return -1
        return 0
    return tuple(sign(c) for c in cols)


def _signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def fit(kind: str, x: pd.DataFrame, y: pd.Series, x_val: pd.DataFrame, y_val: pd.Series):
    if kind == "logistic":
        m = make_pipeline(FunctionTransformer(_signed_log), StandardScaler(), LogisticRegression(max_iter=2000))
        return m.fit(x, y)
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "learning_rate": 0.03,
              "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 20, "seed": 0, "nthread": 4}
    if kind == "xgb_monotone":
        params["monotone_constraints"] = str(monotone(list(x.columns)))
    d = xgb.DMatrix(x, label=y)
    dv = xgb.DMatrix(x_val, label=y_val)
    return xgb.train(params, d, 3000, evals=[(dv, "valid")], early_stopping_rounds=100, verbose_eval=False)


def predict(m, x: pd.DataFrame) -> np.ndarray:
    if isinstance(m, xgb.Booster):
        return m.predict(xgb.DMatrix(x), iteration_range=(0, m.best_iteration + 1))
    return m.predict_proba(x)[:, 1]


def shap_values(m: xgb.Booster, x: pd.DataFrame) -> np.ndarray:
    return m.predict(xgb.DMatrix(x), pred_contribs=True, iteration_range=(0, m.best_iteration + 1))[:, :-1]


def reason_text(feature: str, row: pd.Series) -> str:
    """One plain-language reason per feature family (several features can share one reason)."""
    if feature in ("max_delay", "months_delayed") or feature.startswith("delay_"):
        worst, n = int(row["max_delay"]), int(row["months_delayed"])
        if worst > 0:
            return f"Late payments: {n} of the last 6 months, up to {worst} month{'s' if worst > 1 else ''} late"
        return "Recent repayment pattern (minimum payments or revolving balance)"
    if feature.startswith("paid_") or feature == "paid_ratio_sep":
        return "Amount paid is low compared with the amount owed"
    if feature.startswith("bill_") or feature == "utilization_sep":
        month = "sep" if feature == "utilization_sep" else feature.split("_")[1]
        used = row[f"bill_{month}"] / row["limit"]
        if used < 0.05:
            return f"Little or no card use in {MONTH_NAME[month]}"
        return f"High balance compared with the credit limit ({used:.0%} used in {MONTH_NAME[month]})"
    if feature == "limit":
        return "Credit limit is low"
    return feature


def reason_family(text: str) -> str:
    return text.split(":")[0].split(" (")[0].split(" in ")[0]


def reasons(m: xgb.Booster, x: pd.DataFrame, top: int = 4) -> list[list[str]]:
    sv = shap_values(m, x)
    out = []
    for row_sv, (_, row) in zip(sv, x.iterrows()):
        texts, families = [], set()
        for j in np.argsort(-row_sv):
            if row_sv[j] <= 0 or len(texts) == top:
                break
            t = reason_text(x.columns[j], row)
            family = reason_family(t)
            if family not in families:          # one reason per family, the strongest one
                families.add(family)
                texts.append(t)
        out.append(texts)
    return out
