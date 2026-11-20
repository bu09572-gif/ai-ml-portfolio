"""UCI "default of credit card clients" (Yeh & Lien, 2009; CC BY 4.0): 30,000 credit card holders
of a Taiwanese bank, April to September 2005, and whether each defaulted the next month (22.1%).

The copy is OpenML dataset 42477, fetched by tools/fetch_public_data.py in CI and attached to this
repository's data-v1 release, with its SHA-256 pinned here.

    python data.py     # -> data/raw/public/credit_default_taiwan.parquet
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
RAW = Path(os.environ.get("PUBLIC_DATA_DIR", ROOT / "data" / "raw" / "public"))
URL = "https://github.com/bu09572-gif/ai-ml-portfolio/releases/download/data-v1/credit_default_taiwan.parquet"
SHA256 = "bff8f2c210a37e5659c3b0863534773e8f97ebc01e90806561936db062e511e7"
MONTHS = ["sep", "aug", "jul", "jun", "may", "apr"]     # most recent first, as in the source
NAMES = (["limit", "sex", "education", "marriage", "age"] + [f"delay_{m}" for m in MONTHS]
         + [f"bill_{m}" for m in MONTHS] + [f"paid_{m}" for m in MONTHS] + ["default"])
PROTECTED = ["sex", "marriage", "age"]     # kept out of the model, used only to test fairness


def load() -> pd.DataFrame:
    p = RAW / "credit_default_taiwan.parquet"
    if not p.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        r = requests.get(URL, timeout=300)
        r.raise_for_status()
        p.write_bytes(r.content)
    if hashlib.sha256(p.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("credit_default_taiwan.parquet does not match the pinned copy")
    df = pd.read_parquet(p)
    df.columns = NAMES
    df = df.astype({c: float for c in NAMES[:-1]})
    df["default"] = df["default"].astype(int)
    df["sex"] = df.sex.map({1.0: "male", 2.0: "female"})
    df["marriage"] = df.marriage.map({1.0: "married", 2.0: "single"}).fillna("other")
    df["age_group"] = pd.cut(df.age, [0, 25, 40, 60, 120], labels=["<=25", "26-40", "41-60", "60+"]).astype(str)
    # fixed stratified split: 70% train, 15% valid (cutoff and calibration), 15% test
    rng = np.random.default_rng(0)
    u = rng.random(len(df))
    df["split"] = np.where(u < 0.70, "train", np.where(u < 0.85, "valid", "test"))
    return df


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs: credit limit, repayment history, bills, payments, and a few ratios.
    Sex, marital status and age are left out, and so is education, which can act as a proxy for them."""
    x = df[["limit"] + [c for c in df.columns if c.startswith(("delay_", "bill_", "paid_"))]].copy()
    x["utilization_sep"] = df.bill_sep / df.limit
    x["paid_ratio_sep"] = df.paid_sep / df.bill_aug.clip(lower=1)
    x["months_delayed"] = (df[[f"delay_{m}" for m in MONTHS]] > 0).sum(axis=1)
    x["max_delay"] = df[[f"delay_{m}" for m in MONTHS]].max(axis=1)
    return x


if __name__ == "__main__":
    d = load()
    print(d.groupby("split").default.agg(["size", "mean"]))
