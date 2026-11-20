"""Life underwriting on real health exams and death records (NHANES 2005-2010, deaths to 2019).

Train on the 2005-06 and 2007-08 cycles, test on 2009-10 (people the model never saw, examined later).

    python run.py      # -> results/metrics.json, results/fast_track.png   (under a minute)
"""
from __future__ import annotations

import json
from pathlib import Path

import data
import numpy as np
import pandas as pd
import underwriting as uw
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent / "results"
RACE = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White", 4: "Non-Hispanic Black", 5: "Other or multiracial"}


def paired_auc_gain(y, p_small, p_big, n=1000, seed=0) -> list[float]:
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        gains.append(roc_auc_score(y[i], p_big[i]) - roc_auc_score(y[i], p_small[i]))
    return [round(float(v), 4) for v in np.percentile(gains, [2.5, 97.5])]


def fast_track(y, p_app, p_labs, shares=(0.2, 0.3, 0.4, 0.5, 0.6)) -> dict:
    """Skip the blood test for the safest share by the application model: how many more deaths does that
    group hold than the same share picked with full labs?"""
    out = {}
    for s in shares:
        k = int(len(y) * s)
        a, b = np.argsort(p_app)[:k], np.argsort(p_labs)[:k]
        out[f"{int(s * 100)}pct"] = {"death_rate_picked_by_application": round(float(y[a].mean()), 4),
                                     "death_rate_picked_with_labs": round(float(y[b].mean()), 4),
                                     "overlap": round(len(set(a) & set(b)) / k, 3)}
    return out


def chart(ft: dict, overall: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [int(k[:-3]) for k in ft]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(xs, [100 * v["death_rate_picked_by_application"] for v in ft.values()], "o-", label="picked from the application only")
    ax.plot(xs, [100 * v["death_rate_picked_with_labs"] for v in ft.values()], "s-", label="picked with blood tests")
    ax.axhline(100 * overall, color="grey", ls=":", label=f"all applicants ({100 * overall:.1f}%)")
    ax.set_xlabel("share of applicants fast-tracked (%)")
    ax.set_ylabel("died within 8 years (%)")
    ax.set_title("Who gets fast-tracked: chosen with or without a blood test")
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fast_track.png", dpi=120)
    plt.close(fig)


def main():
    d = data.cohort()
    tr, te = d[d.cycle < 2009].reset_index(drop=True), d[d.cycle == 2009].reset_index(drop=True)
    y, yt = tr.died_8y.to_numpy(), te.died_8y.to_numpy()
    res = {"data": {"people": len(d), "deaths_8y": int(d.died_8y.sum()), "train": len(tr), "test": len(te),
                    "test_death_rate": round(float(yt.mean()), 4), "ages": "40-79 at exam"}, "models": {}}
    preds = {}
    m = uw.logistic().fit(tr[["age"]], y)
    preds["age only"] = m.predict_proba(te[["age"]])[:, 1]
    m = uw.logistic().fit(uw.rules_matrix(tr), y)
    preds["rules class + age"] = m.predict_proba(uw.rules_matrix(te))[:, 1]
    fitted = {}
    for tier, cols in uw.TIERS.items():
        for name, make in [("logistic", uw.logistic), ("gbm", uw.gbm)]:
            fitted[(tier, name)] = make().fit(tr[cols], y)
            preds[f"{tier}, {name}"] = fitted[(tier, name)].predict_proba(te[cols])[:, 1]
    res["models"] = {k: uw.scores(yt, p) for k, p in preds.items()}
    cls = uw.rule_class(te)
    res["rules_classes_test"] = {c: {"people": int((cls == c).sum()), "death_rate": round(float(yt[cls == c].mean()), 4)}
                                 for c in ["preferred", "standard", "smoker"]}
    app, labs = preds["application, logistic"], preds["full_labs, logistic"]
    res["auc_gain_labs_over_application_95ci"] = paired_auc_gain(yt, app, labs)
    res["auc_gain_paramedical_over_application_95ci"] = paired_auc_gain(yt, app, preds["paramedical, logistic"])
    res["fast_track"] = fast_track(yt, app, labs)
    pi = permutation_importance(fitted[("full_labs", "gbm")], te[uw.FULL_LABS], yt, scoring="roc_auc", n_repeats=10, random_state=0)
    res["top_features_full_labs_gbm"] = dict(sorted(zip(uw.FULL_LABS, pi.importances_mean.round(4)), key=lambda kv: -kv[1])[:10])
    race = te.race_eth.map(RACE)
    income = pd.cut(te.income_poverty_ratio, [-1, 1, 2, 4, 10], labels=["below poverty line", "1-2x", "2-4x", "4x and above"])
    res["actual_to_expected_full_labs_logistic"] = {"by_race_ethnicity": uw.actual_to_expected(yt, labs, race),
                                                    "by_income_to_poverty": uw.actual_to_expected(yt, labs, income.astype(str)),
                                                    "by_sex": uw.actual_to_expected(yt, labs, te.female.map({0: "male", 1: "female"}))}
    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    chart(res["fast_track"], float(yt.mean()))
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
