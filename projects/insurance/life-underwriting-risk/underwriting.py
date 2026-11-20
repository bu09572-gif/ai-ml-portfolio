"""8-year mortality models for life underwriting, at three levels of evidence an insurer can collect.

  application   what the applicant tells you: age, sex, smoking, BMI, diagnosed conditions
  paramedical   + a nurse visit: measured blood pressure, pulse, waist
  full_labs     + a blood test: lipids, HbA1c, kidney and liver markers, blood count
Each level is fitted as a logistic regression and a gradient-boosted model. A rules baseline mimics a
traditional preferred / standard / smoker class decision, combined with age.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

APPLICATION = ["age", "female", "bmi", "current_smoker", "former_smoker", "told_high_bp", "told_diabetes",
               "heart_failure", "coronary_disease", "heart_attack", "stroke", "cancer", "emphysema"]
PARAMEDICAL = APPLICATION + ["sbp", "dbp", "pulse", "waist_cm"]
FULL_LABS = PARAMEDICAL + ["total_chol", "hdl", "chol_hdl_ratio", "hba1c", "albumin", "creatinine", "ggt", "alt",
                           "glucose", "alk_phos", "bun", "uric_acid", "wbc", "hemoglobin", "rdw", "mcv", "platelets"]
TIERS = {"application": APPLICATION, "paramedical": PARAMEDICAL, "full_labs": FULL_LABS}
HISTORY = ["told_diabetes", "heart_failure", "coronary_disease", "heart_attack", "stroke", "cancer", "emphysema"]


def rule_class(d: pd.DataFrame) -> pd.Series:
    """A simple traditional class: smoker; else preferred if healthy build, BP under 140/90 and no major history."""
    hist = d[HISTORY].fillna(0).sum(axis=1) > 0
    healthy = d.bmi.between(18.5, 32) & (d.sbp.fillna(0) < 140) & (d.dbp.fillna(0) < 90) & ~hist
    return pd.Series(np.where(d.current_smoker == 1, "smoker", np.where(healthy, "preferred", "standard")), index=d.index)


def logistic():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(max_iter=2000, C=0.5))


def gbm():
    return HistGradientBoostingClassifier(learning_rate=0.04, max_iter=400, max_depth=3, min_samples_leaf=40,
                                          l2_regularization=1.0, early_stopping=True, validation_fraction=0.2,
                                          n_iter_no_change=40, random_state=0)


def rules_matrix(d: pd.DataFrame) -> pd.DataFrame:
    c = rule_class(d)
    return pd.DataFrame({"age": d.age, "smoker": (c == "smoker").astype(int), "standard": (c == "standard").astype(int)})


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    order = np.argsort(-p)
    n = len(y)
    top, bottom = order[: n // 10], order[-(n // 2):]
    return {"auc": round(roc_auc_score(y, p), 4), "brier": round(brier_score_loss(y, p), 4),
            "deaths_in_riskiest_10pct": round(float(y[top].mean()), 3),
            "deaths_in_safest_50pct": round(float(y[bottom].mean()), 3),
            "predicted_over_actual": round(float(p.sum() / y.sum()), 3)}


def actual_to_expected(y: np.ndarray, p: np.ndarray, groups: pd.Series) -> dict:
    """Deaths observed / deaths predicted, per group: 1.0 means the model is right on average for that group."""
    out = {}
    for g, idx in groups.groupby(groups).groups.items():
        i = groups.index.get_indexer(idx)
        out[str(g)] = {"people": int(len(i)), "deaths": int(y[i].sum()), "actual_to_expected": round(float(y[i].sum() / p[i].sum()), 3)}
    return out
