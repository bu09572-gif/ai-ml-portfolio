"""MovieLens 1M: hide each user's latest movie and try to find it among all ~3,700 movies.

    python run.py      # -> results/metrics.json   (about 2 minutes on CPU)
"""
from __future__ import annotations

import json
from pathlib import Path

import data
import numpy as np
import pandas as pd
import recs

OUT = Path(__file__).resolve().parent / "results"


def similar(b: np.ndarray, ix: recs.Index, movies: pd.DataFrame, title: str, n: int = 5) -> list[str]:
    t = movies.set_index("item").title
    j = ix.i[int(movies[movies.title == title].item.iloc[0])]
    return [t[ix.items[k]] for k in np.argsort(-b[j])[:n]]


def main():
    r, movies, users = data.load()
    tr, va, te = recs.split_last(r)
    ix = recs.Index(r)
    target = lambda d: d.set_index("user").item.map(ix.i).reindex(ix.users).to_numpy()  # noqa: E731
    tv, tt = target(va), target(te)
    trva = pd.concat([tr, va])
    x = ix.matrix(trva)
    res = {"data": {"ratings": len(r), "users": len(ix.users), "movies": len(ix.items),
                    "protocol": "leave-last-out by time; rank the held-out movie among all movies the user has not rated"},
           "test": {}}
    for name, f in [("popularity", recs.popularity), ("item_knn", recs.item_knn), ("ials", recs.ials), ("ease", recs.ease)]:
        res["test"][name] = recs.hit_ndcg(recs.mask_seen(f(x), x), tt)
    # EASE scored from only the user's last n movies; n chosen on validation
    xt = ix.matrix(tr)
    bt = recs.ease_weights(xt)
    val = {n: recs.hit_ndcg(recs.mask_seen(np.asarray(recs.recent_matrix(tr, ix, n) @ bt), xt), tv)["ndcg_at_10"]
           for n in [1, 3, 5, 10, 20, 50]}
    best = max(val, key=val.get)
    res["recent_history_validation_ndcg_at_10"] = val
    res["test"][f"ease_last_{best}_movies"] = recs.hit_ndcg(
        recs.mask_seen(np.asarray(recs.recent_matrix(trva, ix, best) @ recs.ease_weights(x)), x), tt)
    # two-stage: learn the ranker on validation targets (features from train), apply to test (features from train + val)
    cv = recs.candidate_features(tr, ix, movies, users)
    yv = (cv.i.to_numpy() == tv[cv.u.to_numpy()]).astype(int)
    ct = recs.candidate_features(trva, ix, movies, users)
    for name, feats in [("two_stage_ranker", recs.FEATURES), ("two_stage_ranker_without_recent_history",
                                                                [f for f in recs.FEATURES if f != "ease_recent"])]:
        m = recs.ranker().fit(cv[feats], yv)
        res["test"][name] = recs.candidate_metrics(ct, m.predict_proba(ct[feats])[:, 1], tt)
    res["candidate_recall_at_100"] = res["test"]["two_stage_ranker"].pop("target_in_candidates")
    res["test"]["two_stage_ranker_without_recent_history"].pop("target_in_candidates")
    b = recs.ease_weights(x)
    res["more_like_this"] = {t: similar(b, ix, movies, t) for t in
                             ["Toy Story (1995)", "Silence of the Lambs, The (1991)", "Godfather, The (1972)"]}
    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
