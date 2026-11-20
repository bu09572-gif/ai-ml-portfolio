"""Find illicit Bitcoin transactions with and without the transaction graph.

Train on time steps 1-34 and test on 35-49, the split the dataset's authors use: the model is always
scored on later transactions than it learned from, as a monitoring system would be.

  local       the transaction's own 93 features
  all         + the authors' 72 one-hop neighbour aggregates
  all+graph   + my graph features: mean local features of payers (in) and payees (out), degrees,
              and the mean of each over two hops
  GCN         2-layer graph convolutional network on the local features (Kipf & Welling), plain PyTorch
"""
from __future__ import annotations

import data
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

TRAIN_END = 34


def adjacency(nodes: pd.DataFrame, edges: pd.DataFrame) -> sp.csr_matrix:
    """A[i, j] = 1 when transaction i pays into transaction j."""
    idx = pd.Series(np.arange(len(nodes)), index=nodes.txId)
    i, j = idx[edges.src].to_numpy(), idx[edges.dst].to_numpy()
    return sp.csr_matrix((np.ones(len(i), np.float32), (i, j)), shape=(len(nodes), len(nodes)))


def _mean_agg(a: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
    deg = np.asarray(a.sum(axis=1)).ravel()
    return sp.diags(1 / np.maximum(deg, 1)) @ a @ x


def graph_features(nodes: pd.DataFrame, a: sp.csr_matrix) -> pd.DataFrame:
    x = nodes[data.LOCAL].to_numpy(np.float32)
    out_mean, in_mean = _mean_agg(a, x), _mean_agg(a.T.tocsr(), x)
    und = ((a + a.T) > 0).astype(np.float32).tocsr()
    two = _mean_agg(und, _mean_agg(und, x))
    deg = np.c_[np.asarray(a.sum(1)).ravel(), np.asarray(a.sum(0)).ravel()]
    cols = {"out_deg": deg[:, 0], "in_deg": deg[:, 1]}
    frames = [pd.DataFrame(cols)]
    for name, m in [("payee_mean", out_mean), ("payer_mean", in_mean), ("hop2_mean", two)]:
        frames.append(pd.DataFrame(m, columns=[f"{name}_{c}" for c in data.LOCAL]))
    return pd.concat(frames, axis=1).astype(np.float32)


def split(nodes: pd.DataFrame):
    lab = nodes.label >= 0
    return (lab & (nodes.time_step <= TRAIN_END)).to_numpy(), (lab & (nodes.time_step > TRAIN_END)).to_numpy()


def scores(y, p, thr: float = 0.5) -> dict:
    yhat = p >= thr
    order = np.argsort(-p)
    return {"precision": round(precision_score(y, yhat, zero_division=0), 3), "recall": round(recall_score(y, yhat), 3),
            "f1": round(f1_score(y, yhat), 3), "average_precision": round(average_precision_score(y, p), 3),
            "precision_at_100": round(float(y[order[:100]].mean()), 3)}


def xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                         tree_method="hist", n_jobs=-1, random_state=0)


def logistic():
    return LogisticRegression(max_iter=3000, C=10)


def gcn_scores(nodes: pd.DataFrame, a: sp.csr_matrix, tr: np.ndarray, epochs: int = 300, seed: int = 0) -> np.ndarray:
    import torch
    torch.manual_seed(seed)
    x = StandardScaler().fit(nodes.loc[tr, data.LOCAL]).transform(nodes[data.LOCAL]).astype(np.float32)
    und = ((a + a.T) > 0).astype(np.float32) + sp.eye(len(nodes), dtype=np.float32)
    d = 1 / np.sqrt(np.asarray(und.sum(1)).ravel())
    norm = (sp.diags(d) @ und @ sp.diags(d)).tocoo()
    ahat = torch.sparse_coo_tensor(check_invariants=False, indices=np.vstack([norm.row, norm.col]), values=norm.data, size=norm.shape).coalesce()
    xt, y = torch.tensor(x), torch.tensor(nodes.label.to_numpy(), dtype=torch.float32)
    trt = torch.tensor(tr)
    w1, w2 = torch.nn.Linear(x.shape[1], 64), torch.nn.Linear(64, 1)
    opt = torch.optim.Adam([*w1.parameters(), *w2.parameters()], lr=0.01, weight_decay=5e-4)
    pos = float((y[trt] == 1).sum())
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y[trt]) - pos) / pos / 3))

    def fwd(train: bool):
        h = torch.relu(torch.sparse.mm(ahat, w1(xt)))
        h = torch.nn.functional.dropout(h, 0.3, training=train)
        return torch.sparse.mm(ahat, w2(h)).squeeze(1)

    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(fwd(True)[trt], y[trt])
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(fwd(False)).numpy()


def per_step_f1(nodes: pd.DataFrame, te: np.ndarray, p: np.ndarray) -> dict:
    t = nodes.time_step.to_numpy()[te]
    y = nodes.label.to_numpy()[te]
    return {int(s): {"illicit": int(y[t == s].sum()), "f1": round(f1_score(y[t == s], p[t == s] >= 0.5, zero_division=0), 3)}
            for s in np.unique(t)}
