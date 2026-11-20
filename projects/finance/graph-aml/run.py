"""Illicit transaction detection on the Elliptic Bitcoin graph.

    python run.py      # -> results/metrics.json, results/f1_by_time_step.png   (about 3 minutes on CPU)
"""
from __future__ import annotations

import json
from pathlib import Path

import aml
import data
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results"


def chart(steps: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = sorted(steps)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(s, [steps[k]["illicit"] for k in s], color="#c9d6e8", label="illicit transactions (right)")
    ax2 = ax.twinx()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax2.yaxis.tick_left()
    ax2.plot(s, [steps[k]["f1"] for k in s], "o-", color="#3b6ea8", label="illicit F1 (left)")
    ax2.set_ylim(0, 1)
    ax.axvline(43, color="#c33", ls="--")
    ax.text(43.2, ax.get_ylim()[1] * 0.9, "dark market\nshut down", color="#c33", fontsize=8)
    ax.set_xlabel("time step (test period)")
    ax.set_title("Best model by time step: it misses what comes after the shutdown")
    fig.legend(loc="center right", bbox_to_anchor=(0.9, 0.55), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "f1_by_time_step.png", dpi=120)
    plt.close(fig)


def main():
    nodes, edges = data.load()
    a = aml.adjacency(nodes, edges)
    tr, te = aml.split(nodes)
    y = nodes.label.to_numpy()
    g = aml.graph_features(nodes, a)
    sets = {"local": nodes[data.LOCAL], "all": nodes[data.LOCAL + data.AGG],
            "all+graph": pd.concat([nodes[data.LOCAL + data.AGG], g], axis=1)}
    res = {"data": {"transactions": len(nodes), "flows": len(edges), "illicit": int((y == 1).sum()), "licit": int((y == 0).sum()),
                    "train_labelled": int(tr.sum()), "test_labelled": int(te.sum()),
                    "illicit_share_train": round(float(y[tr].mean()), 4), "illicit_share_test": round(float(y[te].mean()), 4)},
           "models": {}}
    res["models"]["logistic, local"] = aml.scores(y[te], aml.logistic().fit(sets["local"][tr], y[tr]).predict_proba(sets["local"][te])[:, 1])
    preds = {}
    for k, x in sets.items():
        m = aml.xgb().fit(x[tr], y[tr])
        preds[k] = m.predict_proba(x[te])[:, 1]
        res["models"][f"xgboost, {k}"] = aml.scores(y[te], preds[k])
    gain = pd.Series(m.get_booster().get_score(importance_type="total_gain"))
    res["graph_feature_share_of_gain"] = round(float(gain[gain.index.isin(g.columns)].sum() / gain.sum()), 3)
    res["top_features_by_gain"] = gain.sort_values(ascending=False).head(10).round(1).to_dict()
    res["models"]["gcn, local"] = aml.scores(y[te], aml.gcn_scores(nodes, a, tr)[te])
    steps = aml.per_step_f1(nodes, te, preds["all+graph"])
    t = nodes.time_step.to_numpy()[te]
    for name, mask in [("steps_35_to_42", t <= 42), ("steps_43_to_49", t >= 43)]:
        res[f"xgboost_all+graph_{name}"] = aml.scores(y[te][mask], preds["all+graph"][mask])
    res["f1_by_time_step"] = steps
    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    chart(steps)
    print(json.dumps({k: v for k, v in res.items() if k != "f1_by_time_step"}, indent=2, default=float))


if __name__ == "__main__":
    np.random.seed(0)
    main()
