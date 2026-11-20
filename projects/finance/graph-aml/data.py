"""Elliptic Bitcoin transaction graph (Weber et al., KDD 2019 workshop "Anti-Money Laundering in Bitcoin").

203,769 real Bitcoin transactions over 49 time steps (about two weeks each), 234,355 directed payment
flows between them, and 166 features per transaction (94 about the transaction itself, 72 aggregated
from its neighbours by the dataset's authors). 4,545 are labelled illicit (scams, ransomware, dark
markets...), 42,019 licit, the rest unknown. Mirrored from the PyTorch Geometric copy into this repo's
data-v2 release; SHA-256 pinned below.
"""
from __future__ import annotations

import hashlib
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
RAW = Path(os.environ.get("PUBLIC_V2_DIR", ROOT / "data" / "raw" / "public_v2"))
BASE = "https://github.com/bu09572-gif/ai-ml-portfolio/releases/download/data-v2/"
SHA = {"elliptic_txs_features.csv.zip": "d33d62159e64b5e889f1a7ea880227c612775b58d409598855e0c4400fa52b3e",
       "elliptic_txs_classes.csv.zip": "4ca957f0ceffd5dd164e255c7d5ad9ee69a6fa64ae1dd94d6f113e5ebf3b07ba",
       "elliptic_txs_edgelist.csv.zip": "a2f9f6b67a39da2d8cf87fe77b9db89571ba6d880e5dd5b5991dc45c80fa34ec"}
LOCAL = [f"local_{i}" for i in range(93)]          # the transaction's own features (after time step)
AGG = [f"agg_{i}" for i in range(72)]              # the authors' one-hop neighbour aggregates


def _csv(name: str, **kw) -> pd.DataFrame:
    p = RAW / name
    if not p.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        p.write_bytes(requests.get(BASE + name, timeout=900).content)
    if hashlib.sha256(p.read_bytes()).hexdigest() != SHA[name]:
        raise RuntimeError(f"{name} does not match the mirrored copy")
    z = zipfile.ZipFile(p)
    return pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), **kw)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(nodes, edges). nodes: txId, time_step, label (1 illicit, 0 licit, -1 unknown), features."""
    cache = RAW / "elliptic_nodes.parquet"
    if cache.exists():
        nodes = pd.read_parquet(cache)
    else:
        cols = ["txId", "time_step"] + LOCAL + AGG
        f = _csv("elliptic_txs_features.csv.zip", header=None, names=cols,
                 dtype={c: (np.int64 if c == "txId" else np.int16 if c == "time_step" else np.float32) for c in cols})
        c = _csv("elliptic_txs_classes.csv.zip")
        lab = c.set_index("txId")["class"].astype(str).map({"1": 1, "2": 0}).fillna(-1).astype(np.int8)
        f.insert(2, "label", f.txId.map(lab).astype(np.int8))
        f = f.copy()
        f.to_parquet(cache, index=False)
        nodes = f
    edges = _csv("elliptic_txs_edgelist.csv.zip")
    edges.columns = ["src", "dst"]
    return nodes, edges
