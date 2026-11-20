"""Model validation checks, in the spirit of the Fed's SR 11-7 guidance: an independent review
of data, performance against a challenger, calibration, stability, robustness and fairness.

Every check returns a Finding with status PASS, WARN or FAIL, the measured value, and the rule.
The checker only sees a ModelUnderTest: predictions, data and an optional retrain function,
so it works for any binary classifier.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class ModelUnderTest:
    name: str
    predict: Callable[[pd.DataFrame], np.ndarray]           # features -> probability of the positive class
    x_train: pd.DataFrame
    y_train: np.ndarray
    x_test: pd.DataFrame
    y_test: np.ndarray
    challenger: Callable[[pd.DataFrame], np.ndarray] | None = None
    monotone: dict[str, int] = field(default_factory=dict)  # feature -> +1 / -1 expected direction
    protected_test: pd.DataFrame | None = None              # protected attributes for the test rows
    decision_share: float | None = None                     # share approved (lowest scores) for fairness
    retrain: Callable[[int], Callable[[pd.DataFrame], np.ndarray]] | None = None  # seed -> new predict
    metric: str = "auc"                                     # "auc" or "average_precision" (rare events)


@dataclass
class Finding:
    area: str
    check: str
    status: str
    value: str
    rule: str


def _metric(m: ModelUnderTest, y, p) -> float:
    return roc_auc_score(y, p) if m.metric == "auc" else average_precision_score(y, p)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index over quantile bins of the expected sample."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.histogram(np.clip(expected, edges[0], edges[-1]), edges)[0] / len(expected)
    a = np.histogram(np.clip(actual, edges[0], edges[-1]), edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _status(bad: bool, warn: bool = False) -> str:
    return "FAIL" if bad else "WARN" if warn else "PASS"


def data_checks(m: ModelUnderTest) -> list[Finding]:
    out = []
    miss = float(pd.concat([m.x_train, m.x_test]).isna().mean().max())
    out.append(Finding("data", "missing values", _status(miss > 0.05, miss > 0), f"max {miss:.1%} in one column",
                       "FAIL above 5% in any column"))
    train_keys = set(map(tuple, m.x_train.round(6).to_numpy()))
    dup = float(np.mean([tuple(r) in train_keys for r in m.x_test.round(6).to_numpy()]))
    out.append(Finding("data", "test rows also in train", _status(dup > 0.10, dup > 0.01), f"{dup:.2%}",
                       "WARN above 1%, FAIL above 10%. Some exact matches are real (e.g. unused accounts)"))
    # a single feature that separates the classes almost perfectly is usually leakage
    single = {c: abs(roc_auc_score(m.y_train, m.x_train[c].fillna(0)) - 0.5) + 0.5 for c in m.x_train.columns
              if m.x_train[c].nunique() > 1}
    best, auc = max(single.items(), key=lambda kv: kv[1])
    out.append(Finding("data", "single-feature leakage", _status(auc > 0.95), f"{best}: AUC {auc:.3f}",
                       "FAIL if one feature alone has AUC above 0.95"))
    return out


def performance_checks(m: ModelUnderTest, n_boot: int = 300) -> list[Finding]:
    p = m.predict(m.x_test)
    rng = np.random.default_rng(0)
    vals, diffs = [], []
    pc = m.challenger(m.x_test) if m.challenger else None
    for _ in range(n_boot):
        i = rng.integers(0, len(p), len(p))
        if m.y_test[i].min() == m.y_test[i].max():
            continue
        vals.append(_metric(m, m.y_test[i], p[i]))
        if pc is not None:
            diffs.append(vals[-1] - _metric(m, m.y_test[i], pc[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    name = "AUC" if m.metric == "auc" else "average precision"
    out = [Finding("performance", f"test {name}", "PASS", f"{_metric(m, m.y_test, p):.3f} (95% CI {lo:.3f}-{hi:.3f})",
                   "reported with a bootstrap interval")]
    if pc is not None:
        dlo, dhi = np.percentile(diffs, [2.5, 97.5])
        out.append(Finding("performance", "beats the challenger model", _status(dhi < 0, dlo <= 0),
                           f"difference {np.mean(diffs):+.3f} (95% CI {dlo:+.3f} to {dhi:+.3f})",
                           "FAIL if the challenger is clearly better, WARN if the interval includes 0"))
    tr = _metric(m, m.y_train, m.predict(m.x_train))
    gap = tr - _metric(m, m.y_test, p)
    out.append(Finding("performance", "overfitting (train minus test)", _status(gap > 0.10, gap > 0.03),
                       f"train {tr:.3f}, gap {gap:+.3f}", "WARN above 0.03, FAIL above 0.10"))
    return out


def calibration_checks(m: ModelUnderTest, bins: int = 10) -> list[Finding]:
    p = m.predict(m.x_test)
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    d = pd.DataFrame({"y": m.y_test, "p": p, "q": q}).groupby("q").agg(o=("y", "sum"), e=("p", "sum"), n=("y", "size"))
    ece = float((d.o - d.e).abs().sum() / d.n.sum())
    ratio = float(d.o.sum() / d.e.sum())
    # Hosmer-Lemeshow statistic
    hl = float((((d.o - d.e) ** 2) / (d.e * (1 - d.e / d.n)).clip(lower=1e-9)).sum())
    pval = float(1 - stats.chi2.cdf(hl, len(d) - 2))
    return [
        Finding("calibration", "observed / expected events", _status(abs(ratio - 1) > 0.2, abs(ratio - 1) > 0.1),
                f"{ratio:.3f}", "WARN outside 0.9-1.1, FAIL outside 0.8-1.2"),
        Finding("calibration", "Hosmer-Lemeshow test", _status(pval < 0.001, pval < 0.05),
                f"chi2 {hl:.1f}, p = {pval:.3g}", "WARN if p < 0.05, FAIL if p < 0.001"),
        Finding("calibration", "calibration error by decile", _status(ece > 0.05, ece > 0.02), f"{ece:.4f}",
                "WARN above 0.02, FAIL above 0.05"),
    ]


def stability_checks(m: ModelUnderTest) -> list[Finding]:
    s = psi(m.predict(m.x_train), m.predict(m.x_test))
    out = [Finding("stability", "score PSI (train vs test)", _status(s > 0.25, s > 0.10), f"{s:.3f}",
                   "WARN above 0.10, FAIL above 0.25")]
    feats = sorted(((psi(m.x_train[c].to_numpy(float), m.x_test[c].to_numpy(float)), c) for c in m.x_train.columns),
                   reverse=True)
    worst, col = feats[0]
    n_warn = sum(v > 0.10 for v, _ in feats)
    out.append(Finding("stability", "feature PSI (worst feature)", _status(worst > 0.25, worst > 0.10),
                       f"{col}: {worst:.3f}; {n_warn} features above 0.10", "WARN above 0.10, FAIL above 0.25"))
    return out


def robustness_checks(m: ModelUnderTest, n: int = 1000) -> list[Finding]:
    x = m.x_test.sample(min(n, len(m.x_test)), random_state=0)
    p = m.predict(x)
    rng = np.random.default_rng(0)
    noisy = x + rng.normal(0, 0.01, x.shape) * x.std().to_numpy()      # 1% of a standard deviation
    shift = float(np.median(np.abs(m.predict(noisy) - p)))
    out = [Finding("robustness", "score change under 1% noise", _status(shift > 0.05, shift > 0.01),
                   f"median {shift:.4f}", "WARN above 0.01, FAIL above 0.05")]
    for col, sign in m.monotone.items():
        grid = np.quantile(m.x_train[col], [0.05, 0.25, 0.5, 0.75, 0.95])
        preds = np.array([m.predict(x.assign(**{col: g})) for g in grid])
        viol = float(np.mean((sign * np.diff(preds, axis=0) < -1e-6).any(axis=0)))
        out.append(Finding("robustness", f"monotone in {col} ({'+' if sign > 0 else '-'})",
                           _status(viol > 0.05, viol > 0), f"{viol:.1%} of rows break it",
                           "FAIL above 5% of rows"))
    return out


def fairness_checks(m: ModelUnderTest) -> list[Finding]:
    if m.protected_test is None or m.decision_share is None:
        return []
    p = m.predict(m.x_test)
    approve = p < np.quantile(p, m.decision_share)
    out = []
    for attr in m.protected_test.columns:
        g = m.protected_test[attr].to_numpy()
        groups = [v for v in pd.unique(g) if (g == v).sum() >= 100]          # skip groups too small to read
        rates = {v: approve[g == v].mean() for v in groups}
        best = max(rates.values())
        worst_g = min(rates, key=rates.get)
        air = rates[worst_g] / best
        out.append(Finding("fairness", f"adverse impact ratio by {attr}", _status(air < 0.8, air < 0.9),
                           f"{air:.3f} (lowest: {worst_g})", "four-fifths rule: FAIL below 0.8, WARN below 0.9"))
    return out


def reproducibility_checks(m: ModelUnderTest, seeds=(1, 2, 3)) -> list[Finding]:
    if m.retrain is None:
        return []
    vals = [_metric(m, m.y_test, m.retrain(s)(m.x_test)) for s in seeds]
    spread = max(vals) - min(vals)
    return [Finding("reproducibility", f"retrain with {len(seeds)} seeds", _status(spread > 0.03, spread > 0.01),
                    f"{min(vals):.3f} to {max(vals):.3f}", "WARN if the spread is above 0.01, FAIL above 0.03")]


def validate(m: ModelUnderTest) -> list[Finding]:
    return (data_checks(m) + performance_checks(m) + calibration_checks(m) + stability_checks(m)
            + robustness_checks(m) + fairness_checks(m) + reproducibility_checks(m))


def report(m: ModelUnderTest, findings: list[Finding]) -> str:
    counts = pd.Series([f.status for f in findings]).value_counts()
    head = ", ".join(f"{counts.get(s, 0)} {s}" for s in ["FAIL", "WARN", "PASS"])
    lines = [f"# Validation report: {m.name}", "",
             f"{len(m.x_train):,} training rows, {len(m.x_test):,} test rows, {int(m.y_test.sum()):,} test events. "
             f"Result: {head}.", "",
             "| Area | Check | Status | Value | Rule |", "|---|---|---|---|---|"]
    lines += [f"| {f.area} | {f.check} | **{f.status}** | {f.value} | {f.rule} |" for f in findings]
    return "\n".join(lines) + "\n"
