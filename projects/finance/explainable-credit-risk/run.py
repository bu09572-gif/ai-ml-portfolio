"""Fit the three models, set an approval cutoff on the validation split, then measure accuracy,
fairness by sex / age / marital status, and decline reasons on the test split.

    python run.py      # -> results/metrics.json, results/reasons_examples.json
"""
from __future__ import annotations

import json
from pathlib import Path

import credit
import data
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

OUT = Path(__file__).resolve().parent / "results"
APPROVE_SHARE = 0.80      # approve the 80% of applicants with the lowest predicted risk


def ks(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def calibration(y, p, bins: int = 10) -> float:
    """Expected calibration error over predicted-risk deciles."""
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    d = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((d.n * (d.y - d.p).abs()).sum() / d.n.sum())


def fairness(test: pd.DataFrame, approve: np.ndarray, p: np.ndarray) -> dict:
    out = {}
    for attr in ["sex", "age_group", "marriage"]:
        rows = {}
        for g, idx in test.groupby(attr).groups.items():
            m = test.index.isin(idx)
            y = test.default.to_numpy()[m]
            a = approve[m]
            rows[g] = {"applicants": int(m.sum()), "approval_rate": round(a.mean(), 3),
                       "approval_rate_of_non_defaulters": round(a[y == 0].mean(), 3),   # equal opportunity
                       "default_rate_if_approved": round(y[a].mean(), 3),
                       "predicted_default": round(float(p[m].mean()), 3), "actual_default": round(y.mean(), 3)}
        best = max(r["approval_rate"] for r in rows.values())
        for r in rows.values():
            r["adverse_impact_ratio"] = round(r["approval_rate"] / best, 3)    # four-fifths rule: >= 0.8
        out[attr] = rows
    return out


def main():
    df = data.load()
    x = data.features(df)
    tr, va, te = (df.split == s for s in ["train", "valid", "test"])
    y = df.default
    OUT.mkdir(exist_ok=True)
    res = {"data": {s: {"applicants": int(m.sum()), "default_rate": round(y[m].mean(), 4)}
                    for s, m in [("train", tr), ("valid", va), ("test", te)]}, "models": {}}
    models, preds = {}, {}
    for kind in ["logistic", "xgb_monotone", "xgb_free"]:
        m = credit.fit(kind, x[tr], y[tr], x[va], y[va])
        pv, pt = credit.predict(m, x[va]), credit.predict(m, x[te])
        cutoff = float(np.quantile(pv, APPROVE_SHARE))
        approve = pt < cutoff
        yt = y[te].to_numpy()
        res["models"][kind] = {
            "test_auc": round(roc_auc_score(yt, pt), 4), "test_ks": round(ks(yt, pt), 4),
            "test_brier": round(brier_score_loss(yt, pt), 4), "test_calibration_error": round(calibration(yt, pt), 4),
            "cutoff_pd": round(cutoff, 4), "test_approval_rate": round(approve.mean(), 3),
            "default_rate_approved": round(yt[approve].mean(), 4), "default_rate_declined": round(yt[~approve].mean(), 4),
            "share_of_defaults_declined": round((~approve & (yt == 1)).sum() / yt.sum(), 3)}
        models[kind], preds[kind] = m, (pt, approve)
        print(kind, res["models"][kind], flush=True)

    m = models["xgb_monotone"]
    pt, approve = preds["xgb_monotone"]
    test = df[te].reset_index(drop=True)
    res["fairness_xgb_monotone"] = fairness(test, approve, pt)

    declined = x[te].reset_index(drop=True)[~approve]
    rs = credit.reasons(m, declined)
    res["decline_reasons"] = {"declined": len(rs),
                              "top_reason_counts": pd.Series([credit.reason_family(r[0]) for r in rs if r])
                              .value_counts().to_dict(),
                              "any_reason_counts": pd.Series([credit.reason_family(t) for r in rs for t in r])
                              .value_counts().to_dict(),
                              "declines_with_no_positive_reason": sum(not r for r in rs)}
    sv = credit.shap_values(m, x[te])
    res["global_importance_mean_abs_shap"] = (pd.Series(np.abs(sv).mean(axis=0).astype(float), index=x.columns)
                                              .sort_values(ascending=False).round(4).head(10).to_dict())
    ex_idx = declined.index[:3]
    examples = [{"applicant": int(test.index[i]), "predicted_default": round(float(pt[i]), 3),
                 "actual_default": int(test.default[i]), "reasons": r}
                for i, r in zip(ex_idx, rs[:3])]
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    (OUT / "reasons_examples.json").write_text(json.dumps(examples, indent=2))
    print(json.dumps({k: res[k] for k in ["fairness_xgb_monotone", "decline_reasons",
                                          "global_importance_mean_abs_shap"]}, indent=1, default=float))
    print(json.dumps(examples, indent=1))


if __name__ == "__main__":
    main()
