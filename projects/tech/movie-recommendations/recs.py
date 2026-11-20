"""Recommenders on MovieLens 1M, evaluated the strict way: each user's most recent movie is hidden and
must be found among all ~3,700 movies (no sampled negatives, which inflate scores; Krichene & Rendle 2020).

  popularity    most-watched movies the user hasn't seen
  item_knn      movies similar to the ones the user watched (cosine on who watched what)
  ials          implicit matrix factorisation (Hu, Koren & Volinsky 2008), alternating least squares
  ease          EASE (Steck 2019): a closed-form linear item-to-item model
  ranker        two-stage: EASE retrieves 100 candidates, a gradient-boosted ranker re-orders them using
                the model scores plus recency, popularity trend and the user's genre taste
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def split_last(r: pd.DataFrame):
    """Per user: last interaction -> test, second-to-last -> validation, rest -> train."""
    r = r.sort_values(["user", "ts", "item"]).reset_index(drop=True)
    pos = r.groupby("user").cumcount(ascending=False)
    return r[pos >= 2], r[pos == 1], r[pos == 0]


class Index:
    def __init__(self, r: pd.DataFrame):
        self.users = np.sort(r.user.unique())
        self.items = np.sort(r.item.unique())
        self.u = {u: i for i, u in enumerate(self.users)}
        self.i = {m: j for j, m in enumerate(self.items)}

    def matrix(self, r: pd.DataFrame) -> sp.csr_matrix:
        rows, cols = r.user.map(self.u).to_numpy(), r.item.map(self.i).to_numpy()
        return sp.csr_matrix((np.ones(len(r), np.float32), (rows, cols)), shape=(len(self.users), len(self.items)))


def popularity(x: sp.csr_matrix) -> np.ndarray:
    return np.tile(np.asarray(x.sum(0)).ravel().astype(np.float32), (x.shape[0], 1))


def item_knn(x: sp.csr_matrix, k: int = 100) -> np.ndarray:
    n = np.sqrt(np.asarray(x.multiply(x).sum(0)).ravel()) + 1e-9
    s = (x.T @ x).toarray() / np.outer(n, n)
    np.fill_diagonal(s, 0)
    keep = np.argsort(-s, axis=0)[:k]                      # keep each item's k nearest neighbours
    mask = np.zeros_like(s, bool)
    np.put_along_axis(mask, keep, True, axis=0)
    return np.asarray(x @ np.where(mask, s, 0).astype(np.float32))


def ease(x: sp.csr_matrix, lam: float = 400.0) -> np.ndarray:
    g = (x.T @ x).toarray().astype(np.float64)
    g[np.diag_indices_from(g)] += lam
    p = np.linalg.inv(g)
    b = -p / np.diag(p)
    np.fill_diagonal(b, 0)
    return np.asarray(x @ b.astype(np.float32))


def ials(x: sp.csr_matrix, factors: int = 64, iters: int = 12, reg: float = 0.1, alpha: float = 20.0, seed: int = 0):
    """Implicit ALS: confidence 1 + alpha for watched movies, preference 1 watched / 0 not."""
    rng = np.random.default_rng(seed)
    nu, ni = x.shape
    U = rng.normal(0, 0.01, (nu, factors)).astype(np.float64)
    V = rng.normal(0, 0.01, (ni, factors)).astype(np.float64)
    xc, xt = x.tocsr(), x.T.tocsr()

    def solve(A, B, m):
        BtB = B.T @ B + reg * np.eye(factors)
        for row in range(m.shape[0]):
            idx = m.indices[m.indptr[row]:m.indptr[row + 1]]
            if len(idx) == 0:
                continue
            Bi = B[idx]
            A[row] = np.linalg.solve(BtB + alpha * Bi.T @ Bi, (1 + alpha) * Bi.sum(0))
    for _ in range(iters):
        solve(U, V, xc)
        solve(V, U, xt)
    return (U @ V.T).astype(np.float32)


def mask_seen(scores: np.ndarray, x: sp.csr_matrix) -> np.ndarray:
    s = scores.copy()
    s[x.nonzero()] = -np.inf
    return s


def hit_ndcg(scores: np.ndarray, target: np.ndarray, ks=(10, 20)) -> dict:
    """target[u] = index of the held-out movie. Rank among every movie the user hasn't seen."""
    t = scores[np.arange(len(target)), target]
    rank = (scores > t[:, None]).sum(1)                    # 0 = first
    out = {}
    for k in ks:
        hit = rank < k
        out[f"hit_at_{k}"] = round(float(hit.mean()), 4)
        out[f"ndcg_at_{k}"] = round(float((hit / np.log2(rank + 2)).mean()), 4)
    out["mrr"] = round(float((1 / (rank + 1)).mean()), 4)
    return out


# ---------- two-stage: retrieve with EASE, re-rank with gradient boosting ----------

DAY = 86400
FEATURES = ["ease", "ease_rank", "ease_recent", "knn", "ials", "pop", "pop_30d", "item_age_days", "genre_match",
            "user_count", "user_age"]


def ease_weights(x: sp.csr_matrix, lam: float = 400.0) -> np.ndarray:
    g = (x.T @ x).toarray().astype(np.float64)
    g[np.diag_indices_from(g)] += lam
    p = np.linalg.inv(g)
    b = -p / np.diag(p)
    np.fill_diagonal(b, 0)
    return b.astype(np.float32)


def recent_matrix(hist: pd.DataFrame, ix: Index, n: int = 10) -> sp.csr_matrix:
    last = hist.sort_values(["user", "ts"]).groupby("user").tail(n)
    return ix.matrix(last)


def candidate_features(hist: pd.DataFrame, ix: Index, movies: pd.DataFrame, users: pd.DataFrame, k: int = 100,
                       ials_factors: int = 64) -> pd.DataFrame:
    """Top-k EASE candidates per user with ranking features, all computed from `hist` only."""
    x = ix.matrix(hist)
    B = ease_weights(x)
    e = mask_seen(np.asarray(x @ B), x)
    top = np.argsort(-e, axis=1)[:, :k]
    uu = np.repeat(np.arange(x.shape[0]), k)
    ii = top.ravel()
    er = np.asarray(recent_matrix(hist, ix) @ B)
    kn = item_knn(x)
    al = ials(x, factors=ials_factors)
    pop = np.asarray(x.sum(0)).ravel()
    # time features: each user's last activity, each movie's first rating and its ratings in the 30 days before
    last_ts = hist.groupby("user").ts.max().reindex(ix.users).to_numpy()
    first_ts = hist.groupby("item").ts.min().reindex(ix.items).to_numpy(dtype=float)
    item_ts = hist.assign(j=hist.item.map(ix.i)).sort_values("ts").groupby("j").ts.apply(np.asarray)
    t_end = last_ts[uu]
    pop30 = np.zeros(len(ii))
    order = np.argsort(ii, kind="stable")
    for j, idx in pd.Series(order).groupby(ii[order]):
        ts = item_ts.get(j)
        if ts is None:
            continue
        e_ = t_end[idx.to_numpy()]
        pop30[idx.to_numpy()] = np.searchsorted(ts, e_, "right") - np.searchsorted(ts, e_ - 30 * DAY, "left")
    # genre taste: share of the user's movies in each genre, dotted with the candidate's genres
    g = movies.set_index("item").genres.str.get_dummies("|").reindex(ix.items).fillna(0).to_numpy(np.float32)
    taste = np.asarray(x @ g) / np.maximum(np.asarray(x.sum(1)), 1)
    gm = (taste[uu] * g[ii]).sum(1) / np.maximum(g[ii].sum(1), 1)
    ua = users.set_index("user").age.reindex(ix.users).to_numpy()
    return pd.DataFrame({"u": uu, "i": ii, "ease": e[uu, ii], "ease_rank": np.tile(np.arange(k), x.shape[0]),
                         "ease_recent": er[uu, ii], "knn": kn[uu, ii], "ials": al[uu, ii], "pop": pop[ii], "pop_30d": pop30,
                         "item_age_days": (t_end - first_ts[ii]) / DAY, "genre_match": gm,
                         "user_count": np.asarray(x.sum(1)).ravel()[uu], "user_age": ua[uu]})


def ranker():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=6, min_samples_leaf=100,
                                          l2_regularization=1.0, random_state=0)


def candidate_metrics(c: pd.DataFrame, score: np.ndarray, target: np.ndarray, ks=(10, 20)) -> dict:
    """Hit and NDCG when the list is the re-ordered candidates; a target outside the candidates is a miss."""
    c = c.assign(s=score, hit=c.i.to_numpy() == target[c.u.to_numpy()])
    c["rank"] = c.groupby("u").s.rank(ascending=False, method="first") - 1
    r = c[c.hit].set_index("u")["rank"].reindex(range(len(target)))
    out = {}
    for k in ks:
        h = r < k
        out[f"hit_at_{k}"] = round(float(h.mean()), 4)
        out[f"ndcg_at_{k}"] = round(float((h / np.log2(r.fillna(1e9) + 2)).mean()), 4)
    out["target_in_candidates"] = round(float(r.notna().mean()), 4)
    return out
