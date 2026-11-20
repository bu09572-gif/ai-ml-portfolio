"""freMTPL2: French motor third-party liability policies from one insurer (CASdatasets, Dutang & Charpentier).

  freMTPL2freq  677,991 policies: exposure, number of claims, driver and vehicle features
  freMTPL2sev   26,444 individual claim amounts, keyed by policy id

Cleaning follows the usual treatment of this dataset (scikit-learn's Tweedie example,
Noll, Salzmann & Wüthrich 2020): claim counts capped at 4, exposure at 1 year, claim amounts
at 200,000, and policies with a claim count but no recorded amount get a count of 0.

    python data.py     # -> data/raw/fremtpl2/policies.parquet
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
RAW = Path(os.environ.get("FREMTPL_DIR", ROOT / "data" / "raw" / "fremtpl2"))
BASE = "https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/"
FILES = {
    "freMTPL2freq.rda": "82c8598513d9fa78226b6c7d271a9940eca4b8083e9e32320a75d377bdfe15a3",
    "freMTPL2sev.rda": "78e6e5016ae046603b37e88ff8ad8ca327f5d59f827c1f75d140388de09b14b3",
}
FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region"]
CATEGORICAL = ["VehBrand", "VehGas", "Area", "Region"]


def _fetch(name: str) -> Path:
    p = RAW / name
    if not p.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        r = requests.get(BASE + name, timeout=300)
        r.raise_for_status()
        p.write_bytes(r.content)
    if hashlib.sha256(p.read_bytes()).hexdigest() != FILES[name]:
        raise RuntimeError(f"{name} changed upstream; check the new version before using it")
    return p


def _read_rda(p: Path) -> pd.DataFrame:
    import pyreadr
    return next(iter(pyreadr.read_r(p).values())).reset_index(drop=True)


def load() -> pd.DataFrame:
    out = RAW / "policies.parquet"
    if out.exists():
        return pd.read_parquet(out)
    freq = _read_rda(_fetch("freMTPL2freq.rda"))
    sev = _read_rda(_fetch("freMTPL2sev.rda"))
    freq["IDpol"] = freq.IDpol.astype(str).astype(float).astype(int)   # R stored some ids as "1e+05"
    sev["IDpol"] = sev.IDpol.astype(str).astype(float).astype(int)
    amount = sev.groupby("IDpol").ClaimAmount.sum()
    df = freq.set_index("IDpol").join(amount).reset_index()
    df["ClaimAmount"] = df.ClaimAmount.fillna(0).clip(upper=200_000)
    df["ClaimNb"] = df.ClaimNb.clip(upper=4).astype(int)
    df.loc[(df.ClaimAmount == 0) & (df.ClaimNb >= 1), "ClaimNb"] = 0
    df["Exposure"] = df.Exposure.clip(upper=1)
    for c in CATEGORICAL:
        df[c] = df[c].astype(str)
    # fixed 80/20 split by policy, so every model is trained and tested on the same rows
    df["test"] = np.random.default_rng(0).random(len(df)) < 0.2
    df.to_parquet(out, index=False)
    return df


if __name__ == "__main__":
    d = load()
    print(f"{len(d):,} policies, {d.Exposure.sum():,.0f} policy-years, {d.ClaimNb.sum():,} claims, "
          f"{d.ClaimAmount.sum() / 1e6:,.1f}M EUR claims; test rows {d.test.sum():,}")
