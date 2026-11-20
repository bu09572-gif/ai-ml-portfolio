"""Claim severity on real French motor claims: what the claims are made of, how heavy the tail is,
and how well claim size can be predicted below a large-loss cap.

    python run.py     # -> results/metrics.json, results/mean_excess.png
"""
from __future__ import annotations

import json
from pathlib import Path

import data
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import GammaRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_gamma_deviance, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder, StandardScaler

OUT = Path(__file__).resolve().parent / "results"
CAP = 10_000            # EUR. Claims above this are "large" and handled by the tail model
FLAT_MIN_COUNT = 100    # an exact amount paid this often in training is a fixed-fee settlement
TAIL_Q = 0.98           # GPD threshold: 98th percentile of training claims


def lorenz_gini(actual, pred) -> float:
    def g(order):
        cum = np.cumsum(actual[order]) / actual.sum()
        x = np.arange(1, len(actual) + 1) / len(actual)
        return 1 - 2 * np.trapezoid(cum, x)
    return float(g(np.argsort(pred)) / g(np.argsort(actual)))


def composition(tr: pd.DataFrame, all_claims: pd.DataFrame) -> tuple[dict, set]:
    vc = tr.ClaimAmount.value_counts()
    flat = set(vc[vc >= FLAT_MIN_COUNT].index)
    y = all_claims.ClaimAmount
    is_flat = y.isin(flat)
    top = y.nlargest(5)
    return {
        "claims": len(y), "total_eur": round(float(y.sum())),
        "fixed_fee_amounts": sorted(float(v) for v in flat),
        "fixed_fee_share_of_claims": round(is_flat.mean(), 3), "fixed_fee_share_of_cost": round(y[is_flat].sum() / y.sum(), 3),
        "share_of_claims_above_cap": round((y > CAP).mean(), 4),
        "share_of_cost_above_cap": round(np.maximum(y - CAP, 0).sum() / y.sum(), 3),
        "share_of_cost_from_top_5_claims": round(top.sum() / y.sum(), 3),
        "top_5_claims_eur": [round(float(v)) for v in top],
        "median_claim": round(float(y.median()), 1), "mean_claim": round(float(y.mean()), 1),
    }, flat


def tail(tr: pd.DataFrame) -> dict:
    y = tr.ClaimAmount.to_numpy()
    u = float(np.quantile(y, TAIL_Q))
    exc = y[y > u] - u
    xi, _, sigma = stats.genpareto.fit(exc, floc=0)
    rng = np.random.default_rng(0)
    boot = [stats.genpareto.fit(rng.choice(exc, len(exc)), floc=0)[0] for _ in range(200)]
    p_u = (y > u).mean()

    def level(n):    # claim size exceeded on average once in n claims
        return u + sigma / xi * ((n * p_u) ** xi - 1)
    return {"threshold_eur": round(u), "exceedances": int(len(exc)), "shape_xi": round(float(xi), 3),
            "shape_xi_95ci": [round(float(v), 3) for v in np.percentile(boot, [2.5, 97.5])],
            "scale_sigma": round(float(sigma)),
            "mean_is_finite": bool(xi < 1),
            "one_in_1000_claims_eur": {"gpd": round(level(1000)), "empirical": round(float(np.quantile(y, 1 - 1 / 1000)))},
            "one_in_10000_claims_eur": {"gpd": round(level(10_000)), "empirical": round(float(np.quantile(y, 1 - 1e-4)))}}


def glm(cols):
    return ColumnTransformer([
        ("binned", KBinsDiscretizer(n_bins=8, strategy="quantile", encode="onehot", subsample=None), ["VehAge", "DrivAge"]),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20), data.CATEGORICAL + ["VehPower"]),
        ("bonus", StandardScaler(), ["BonusMalus"]),
        ("density", make_pipeline(FunctionTransformer(np.log), StandardScaler()), ["Density"]),
    ])


def _x(df):
    x = df[data.FEATURES].copy()
    for c in data.CATEGORICAL:
        x[c] = x[c].astype("category")
    return x


def hgb(cls, **kw):
    return cls(learning_rate=0.05, max_iter=500, max_depth=3, min_samples_leaf=100, l2_regularization=1.0,
               early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, categorical_features="from_dtype",
               random_state=0, **kw)


def models(tr: pd.DataFrame, te: pd.DataFrame, flat: set) -> dict:
    yc_tr, yc_te = tr.ClaimAmount.clip(upper=CAP), te.ClaimAmount.clip(upper=CAP)
    preds = {"constant": np.full(len(te), yc_tr.mean())}
    g = make_pipeline(glm(data.FEATURES), GammaRegressor(alpha=1e-2, solver="newton-cholesky")).fit(tr[data.FEATURES], yc_tr)
    preds["gamma_glm"] = g.predict(te[data.FEATURES])
    preds["gbm_gamma"] = hgb(HistGradientBoostingRegressor, loss="gamma").fit(_x(tr), yc_tr).predict(_x(te))
    # composite: probability of a fixed-fee settlement x its average, plus a GBM for the other claims
    is_flat = tr.ClaimAmount.isin(flat)
    clf = hgb(HistGradientBoostingClassifier).fit(_x(tr), is_flat)
    p_flat = clf.predict_proba(_x(te))[:, 1]
    body = hgb(HistGradientBoostingRegressor, loss="gamma").fit(_x(tr[~is_flat]), yc_tr[~is_flat])
    preds["composite"] = p_flat * yc_tr[is_flat].mean() + (1 - p_flat) * body.predict(_x(te))
    out = {}
    for k, p in preds.items():
        out[k] = {"gamma_deviance": round(mean_gamma_deviance(yc_te, p), 4), "mae_eur": round(mean_absolute_error(yc_te, p), 1),
                  "gini": round(lorenz_gini(yc_te.to_numpy(), p), 3), "predicted_over_actual": round(p.sum() / yc_te.sum(), 3)}
    te_flat = te.ClaimAmount.isin(flat)
    out["fixed_fee_classifier_test_auc"] = round(roc_auc_score(te_flat, p_flat), 3)
    big = hgb(HistGradientBoostingClassifier).fit(_x(tr), tr.ClaimAmount > CAP)
    pb = big.predict_proba(_x(te))[:, 1]
    out["large_claim_classifier"] = {"test_auc": round(roc_auc_score(te.ClaimAmount > CAP, pb), 3),
                                     "test_average_precision": round(average_precision_score(te.ClaimAmount > CAP, pb), 3),
                                     "base_rate": round(float((te.ClaimAmount > CAP).mean()), 4)}
    return out


def large_loss_load(tr: pd.DataFrame, te: pd.DataFrame) -> dict:
    """Average cost above the cap per claim: the amount every claim's price must carry for large losses."""
    ex_tr = np.maximum(tr.ClaimAmount.to_numpy() - CAP, 0)
    rng = np.random.default_rng(0)
    boot = [rng.choice(ex_tr, len(ex_tr)).mean() for _ in range(2000)]
    capped_mean = tr.ClaimAmount.clip(upper=CAP).mean()
    return {"excess_per_claim_train_eur": round(ex_tr.mean(), 1),
            "excess_per_claim_95ci_eur": [round(float(v), 1) for v in np.percentile(boot, [2.5, 97.5])],
            "excess_per_claim_test_eur": round(float(np.maximum(te.ClaimAmount - CAP, 0).mean()), 1),
            "load_on_capped_severity": round(ex_tr.mean() / capped_mean, 3),
            "excess_per_claim_train_without_largest_claim_eur": round(float(np.sort(ex_tr)[:-1].mean()), 1)}


def chart(tr: pd.DataFrame, t: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y = np.sort(tr.ClaimAmount.to_numpy())
    us = np.quantile(y, np.linspace(0.80, 0.995, 60))
    me = [(y[y > u] - u).mean() for u in us]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(us, me, "o-", ms=3)
    ax.axvline(t["threshold_eur"], color="#c33", ls="--", label=f"GPD threshold ({t['threshold_eur']:,} EUR)")
    ax.set_xscale("log")
    ax.set_xlabel("threshold u (EUR, log scale)")
    ax.set_ylabel("mean excess over u (EUR)")
    ax.set_title("Mean excess keeps rising: a very heavy tail")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "mean_excess.png", dpi=120)
    plt.close(fig)


def main():
    c = data.claims()
    tr, te = c[~c.test].reset_index(drop=True), c[c.test].reset_index(drop=True)
    OUT.mkdir(exist_ok=True)
    comp, flat = composition(tr, c)
    t = tail(tr)
    res = {"data": {"train_claims": len(tr), "test_claims": len(te), "cap_eur": CAP},
           "composition": comp, "tail": t, "capped_severity_models": models(tr, te, flat),
           "large_loss_load": large_loss_load(tr, te)}
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    chart(tr, t)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
