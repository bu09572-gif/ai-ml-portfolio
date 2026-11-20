"""Subrogation referral model on 2023 NYPD crash reports. Train January-September, test October-December.

    python run.py      # -> results/metrics.json, results/precision_recall.png   (about a minute)
"""
from __future__ import annotations

import json
from pathlib import Path

import data
import numpy as np
import subro as s
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve

OUT = Path(__file__).resolve().parent / "results"
SPLIT = "2023-10-01"


def chart(y, p, rule_pt: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pr, rc, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rc, pr, label="model")
    ax.plot([rule_pt["recall"]], [rule_pt["precision"]], "o", color="#c33", label="referral rule (parked / rear-ended)")
    ax.axhline(y.mean(), color="grey", ls=":", label=f"base rate ({100 * y.mean():.0f}%)")
    ax.set_xlabel("recall: share of recoverable claims referred")
    ax.set_ylabel("precision: share of referrals recoverable")
    ax.set_title("Subrogation referrals, Oct-Dec 2023 test claims")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "precision_recall.png", dpi=120)
    plt.close(fig)


def main():
    c = s.claims(data.crashes(), data.vehicles())
    c = c[c.date.notna()].reset_index(drop=True)
    X = s.x(c)
    lab = c.labelled.to_numpy()
    tr, te = lab & (c.date < SPLIT).to_numpy(), lab & (c.date >= SPLIT).to_numpy()
    y = c.recovery.to_numpy()
    m = s.model().fit(X[tr], y[tr])
    p = m.predict_proba(X)[:, 1]
    r = s.rule(c).to_numpy().astype(bool)
    rule_te = s.referral_scores(y[te], r[te])
    res = {"data": {"two_vehicle_crashes": int(c.collision_id.nunique()), "claims": len(c),
                    "labelled_claims": int(lab.sum()), "train": int(tr.sum()), "test": int(te.sum()),
                    "recovery_rate_labelled": round(float(y[lab].mean()), 3)},
           "rule": rule_te, "model": s.scores(y[te], p[te]),
           "model_at_share": {f"{int(100 * k)}pct": s.ranked(y[te], p[te], k) for k in [rule_te["referred_share"], 0.3, 0.4]}}
    pi = permutation_importance(m, X[te], y[te], scoring="average_precision", n_repeats=5, random_state=0)
    res["top_features"] = dict(sorted(zip(X.columns, pi.importances_mean.round(4)), key=lambda kv: -kv[1])[:8])
    L = c[lab]
    res["recovery_rate_by_our_maneuver"] = (L.groupby("our_pre_crash").recovery.agg(["size", "mean"])
                                            .query("size >= 1000").sort_values("mean", ascending=False).round(3)
                                            .to_dict(orient="index"))
    unl = ~lab
    res["unlabelled_claims"] = {"claims": int(unl.sum()), "model_score_above_0.9": int((p[unl] > 0.9).sum()),
                                "rule_flags": int(r[unl].sum())}
    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    chart(y[te], p[te], rule_te)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    np.random.seed(0)
    main()
