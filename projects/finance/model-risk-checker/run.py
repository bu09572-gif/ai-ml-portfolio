"""Validate the two finance models in this repo, then check that the validator catches known faults.

    python run.py      # -> reports/credit_default.md, reports/card_fraud.md, results/fault_injection.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import checks
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
FIN = HERE.parent


def _load(project: str, module: str, alias: str):
    """Import a module from a sibling project under a unique name (both projects have a data.py)."""
    path = FIN / project
    sys.path.insert(0, str(path))
    try:
        for name in ("data", "model", "credit"):
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location(alias, path / f"{module}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(path))


# ---------- the credit default model (explainable-credit-risk) ----------

def credit_model() -> tuple[checks.ModelUnderTest, dict]:
    data = _load("explainable-credit-risk", "data", "credit_data")
    sys.modules["data"] = data
    credit = _load("explainable-credit-risk", "credit", "credit_credit")
    df = data.load()
    x, y = data.features(df), df.default.to_numpy()
    tr, va, te = ((df.split == s).to_numpy() for s in ["train", "valid", "test"])

    def fit(kind, xtr, ytr, xva=None):
        mdl = credit.fit(kind, xtr, pd.Series(ytr), x[va] if xva is None else xva, pd.Series(y[va]))
        return lambda z: credit.predict(mdl, z)

    def retrain(seed):
        i = np.random.default_rng(seed).integers(0, tr.sum(), tr.sum())
        return fit("xgb_monotone", x[tr].iloc[i], y[tr][i])

    signs = dict(zip(x.columns, credit.monotone(list(x.columns))))
    m = checks.ModelUnderTest(
        name="Credit default model (monotone XGBoost)", predict=fit("xgb_monotone", x[tr], y[tr]),
        x_train=x[tr], y_train=y[tr], x_test=x[te], y_test=y[te],
        challenger=fit("logistic", x[tr], y[tr]),
        monotone={c: signs[c] for c in ["delay_sep", "max_delay", "utilization_sep", "limit", "paid_sep"]},
        protected_test=df.loc[te, ["sex", "age_group", "marriage"]].reset_index(drop=True).set_index(x[te].index),
        decision_share=0.80, retrain=retrain)
    return m, {"fit": fit, "x": x, "y": y, "tr": tr, "va": va, "te": te, "df": df}


def fault_injection(base: checks.ModelUnderTest, ctx: dict) -> list[dict]:
    """Break the credit model or its data on purpose, one fault at a time, and see what the checker flags."""
    x, y, tr, te, fit, df = ctx["x"], ctx["y"], ctx["tr"], ctx["te"], ctx["fit"], ctx["df"]
    rng = np.random.default_rng(7)
    cases = []

    # 1. target leakage: a column computed from the outcome sneaks into the features
    xl = x.assign(collections_flag=y + rng.normal(0, 0.05, len(y)))
    leak = fit("xgb_monotone", xl[tr], y[tr], xl[ctx["va"]])
    cases.append(("target leakage", "single-feature leakage",
                  checks.ModelUnderTest(**{**base.__dict__, "predict": leak, "x_train": xl[tr], "x_test": xl[te],
                                           "challenger": None, "monotone": {}, "retrain": None})))
    # 2. test rows copied into the training set
    cases.append(("test rows in training data", "test rows also in train",
                  checks.ModelUnderTest(**{**base.__dict__, "x_train": pd.concat([x[tr], x[te]]),
                                           "y_train": np.r_[y[tr], y[te]], "retrain": None})))
    # 3. population drift: test customers have 60% higher bills and 40% lower limits
    xd = x[te].copy()
    bills = [c for c in x.columns if c.startswith("bill_")]
    xd[bills] *= 1.6
    xd["limit"] *= 0.6
    xd["utilization_sep"] = xd.bill_sep / xd.limit
    cases.append(("population drift", "feature PSI",
                  checks.ModelUnderTest(**{**base.__dict__, "x_test": xd, "retrain": None})))
    # 4. miscalibration: probabilities inflated by 60% (e.g. a wrong prior after resampling)
    cases.append(("miscalibrated scores", "observed / expected events",
                  checks.ModelUnderTest(**{**base.__dict__, "predict": lambda z: np.clip(base.predict(z) * 1.6, 0, 1),
                                           "retrain": None})))
    # 5. monotonicity broken: the unconstrained model, reviewed against the same expectations
    cases.append(("unconstrained model", "monotone in",
                  checks.ModelUnderTest(**{**base.__dict__, "predict": fit("xgb_free", x[tr], y[tr]), "retrain": None})))
    # 6. a model that penalises young applicants directly
    young = (df.age <= 25).to_numpy()

    def biased(z):
        return base.predict(z) + 0.25 * young[z.index.to_numpy()]
    cases.append(("age penalty", "adverse impact ratio by age_group",
                  checks.ModelUnderTest(**{**base.__dict__, "predict": biased, "retrain": None})))

    out = []
    for fault, expected_check, m in cases:
        found = checks.validate(m)
        hit = [f for f in found if f.check.startswith(expected_check)]
        flagged = [f"{f.check}: {f.status}" for f in found if f.status != "PASS"]
        out.append({"fault": fault, "expected_check": expected_check,
                    "expected_check_status": hit[0].status if hit else "missing",
                    "caught": bool(hit) and any(f.status == "FAIL" for f in hit),
                    "all_flags": flagged})
        print(out[-1], flush=True)
    return out


# ---------- the card fraud model (realtime-fraud-scoring) ----------

def fraud_model() -> checks.ModelUnderTest:
    data = _load("realtime-fraud-scoring", "data", "fraud_data")
    sys.modules["data"] = data
    model = _load("realtime-fraud-scoring", "model", "fraud_model")
    df = data.load()
    tr, va, te = (df[df.split == s] for s in ["train", "valid", "test"])
    scorer = model.Scorer.load(FIN / "realtime-fraud-scoring" / "model")
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)).fit(tr[data.FEATURES], tr.Class)

    def retrain(seed):
        b = model.train(tr.sample(frac=1, replace=True, random_state=seed), va)
        s = model.Scorer(b, 0.0)
        return lambda z: s.score(z)

    return checks.ModelUnderTest(
        name="Card fraud model (XGBoost)", predict=lambda z: scorer.score(z),
        x_train=tr[data.FEATURES], y_train=tr.Class.to_numpy(), x_test=te[data.FEATURES], y_test=te.Class.to_numpy(),
        challenger=lambda z: lr.predict_proba(z[data.FEATURES])[:, 1], retrain=retrain, metric="average_precision")


def main():
    (HERE / "reports").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    summary = {}
    credit, ctx = credit_model()
    found = checks.validate(credit)
    (HERE / "reports" / "credit_default.md").write_text(checks.report(credit, found))
    summary["credit_default"] = pd.Series([f.status for f in found]).value_counts().to_dict()
    faults = fault_injection(credit, ctx)
    (HERE / "results" / "fault_injection.json").write_text(json.dumps(faults, indent=2))

    fraud = fraud_model()
    found = checks.validate(fraud)
    (HERE / "reports" / "card_fraud.md").write_text(checks.report(fraud, found))
    summary["card_fraud"] = pd.Series([f.status for f in found]).value_counts().to_dict()
    summary["faults_caught"] = f"{sum(f['caught'] for f in faults)} of {len(faults)}"
    (HERE / "results" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print((HERE / "reports" / "credit_default.md").read_text())
    print((HERE / "reports" / "card_fraud.md").read_text())


if __name__ == "__main__":
    main()
