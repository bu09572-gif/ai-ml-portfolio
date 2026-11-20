"""Fit GLM and GBM pricing models on freMTPL2, compare aware / unaware / discrimination-free
prices on the held-out 20%, and write results/metrics.json plus two charts.

    python run.py              # full data, a few minutes on a laptop CPU
    python run.py --sample 0.1 # quick try
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import data
import numpy as np
import pandas as pd
import pricing
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
AGE_BANDS = [17, 25, 64, 200]
AGE_LABELS = ["18-25", "26-64", "65+"]


def gini(actual: np.ndarray, pred: np.ndarray, exposure: np.ndarray) -> float:
    """Normalised Gini of the ordered Lorenz curve: how well the price ranks policies by actual cost.
    1 is perfect ranking, 0 is random."""
    def g(order):
        cum_exp = np.cumsum(exposure[order]) / exposure.sum()
        cum_loss = np.cumsum(actual[order]) / actual.sum()
        return 1 - 2 * np.trapezoid(cum_loss, cum_exp)
    return float(g(np.argsort(pred)) / g(np.argsort(actual / exposure)))


def model_scores(m: pricing.FreqSev, test: pd.DataFrame) -> dict:
    claims = test[test.ClaimNb > 0]
    freq, w = m.frequency(test), test.Exposure
    pp = m.pure_premium(test)
    null_freq = np.full(len(test), (test.ClaimNb.sum() / w.sum()))
    return {
        "frequency_poisson_deviance": mean_poisson_deviance(test.ClaimNb / w, freq, sample_weight=w),
        "frequency_deviance_explained": 1 - mean_poisson_deviance(test.ClaimNb / w, freq, sample_weight=w)
        / mean_poisson_deviance(test.ClaimNb / w, null_freq, sample_weight=w),
        "severity_gamma_deviance": mean_gamma_deviance(claims.ClaimAmount / claims.ClaimNb, m.severity(claims),
                                                       sample_weight=claims.ClaimNb),
        "pure_premium_gini": gini(test.ClaimAmount.to_numpy(), pp, w.to_numpy()),
        "predicted_over_actual_claims": float((pp * w).sum() / test.ClaimAmount.sum()),
    }


def by_age(test: pd.DataFrame, prices: pd.DataFrame) -> dict:
    band = pd.cut(test.DrivAge, AGE_BANDS, labels=AGE_LABELS)
    w = test.Exposure
    out = {}
    for b in AGE_LABELS:
        m = band == b
        actual = test.ClaimAmount[m].sum() / w[m].sum()
        row = {"policies": int(m.sum()), "actual_cost_per_year": round(actual, 1)}
        for p in prices:
            avg = (prices[p][m] * w[m]).sum() / w[m].sum()
            row[f"{p}_premium"] = round(avg, 1)
            row[f"{p}_loss_ratio"] = round(actual / avg, 3)   # claims paid / premium charged
        out[b] = row
    return out


def proxy_share(test: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """How much of the age gap in premiums each price keeps, relative to the aware price.
    Gap = mean premium of 18-25 year olds / mean premium of 26-64 year olds, minus 1."""
    band = pd.cut(test.DrivAge, AGE_BANDS, labels=AGE_LABELS)
    w = test.Exposure

    def gap(p):
        avg = lambda b: (prices[p][band == b] * w[band == b]).sum() / w[band == b].sum()  # noqa: E731
        return avg("18-25") / avg("26-64") - 1
    g = {p: gap(p) for p in prices}
    return {p: {"young_vs_middle_premium_gap": round(g[p], 3), "share_of_aware_gap": round(g[p] / g["aware"], 3)}
            for p in prices}


def charts(test: pd.DataFrame, prices: dict[str, pd.DataFrame], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    age = pd.cut(test.DrivAge, list(range(17, 86, 4)) + [101])
    w = test.Exposure
    p = prices["gbm"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    mid = [iv.mid for iv in age.cat.categories]
    actual = test.ClaimAmount.groupby(age, observed=False).sum() / w.groupby(age, observed=False).sum()
    ax.plot(mid, actual.values, "o", color="#888", label="actual cost (test set)")
    for name, style in [("aware", "-"), ("unaware", "--"), ("discrimination_free", "-.")]:
        avg = (p[name] * w).groupby(age, observed=False).sum() / w.groupby(age, observed=False).sum()
        ax.plot(mid, avg.values, style, lw=2, label=f"{name.replace('_', '-')} premium")
    ax.set_xlabel("driver age")
    ax.set_ylabel("EUR per policy-year")
    ax.set_title("Average premium by driver age (GBM, held-out policies)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / "premium_by_age.png", dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=float, default=1.0)
    a = ap.parse_args(argv)
    df = data.load()
    if a.sample < 1:
        df = df.sample(frac=a.sample, random_state=0)
    train, test = df[~df.test], df[df.test]
    OUT.mkdir(exist_ok=True)
    res = {"data": {"policies": len(df), "train": len(train), "test": len(test),
                    "test_policy_years": round(float(test.Exposure.sum())), "test_claims": int(test.ClaimNb.sum())},
           "models": {}, "by_age_band": {}, "age_gap": {}}
    prices = {}
    for kind in ["glm", "gbm"]:
        t0 = time.perf_counter()
        models = pricing.fit_prices(train, kind)
        prices[kind] = pricing.price(models, test)
        res["models"][kind] = {
            "aware": model_scores(models["aware"], test),
            "unaware": model_scores(models["unaware"], test),
            "discrimination_free": {
                "pure_premium_gini": gini(test.ClaimAmount.to_numpy(), prices[kind].discrimination_free.to_numpy(),
                                          test.Exposure.to_numpy()),
                "predicted_over_actual_claims": float((prices[kind].discrimination_free * test.Exposure).sum()
                                                      / test.ClaimAmount.sum())},
            "fit_seconds": round(time.perf_counter() - t0, 1)}
        res["by_age_band"][kind] = by_age(test, prices[kind])
        res["age_gap"][kind] = proxy_share(test, prices[kind])
        print(kind, json.dumps(res["models"][kind], indent=1, default=float), flush=True)
    res = json.loads(json.dumps(res, default=float))
    (OUT / ("metrics.json" if a.sample == 1 else "metrics_sample.json")).write_text(json.dumps(res, indent=2))
    if a.sample == 1:
        charts(test, prices, OUT)
    print(json.dumps({k: res[k] for k in ["by_age_band", "age_gap"]}, indent=1))


if __name__ == "__main__":
    main()
