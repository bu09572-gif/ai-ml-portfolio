"""JailbreakBench behaviours (Chao et al., NeurIPS 2024 Datasets and Benchmarks; MIT licence).

100 harmful behaviours in 10 categories (from AdvBench, HarmBench, TDC and original ones) and 100 benign
behaviours matched to them on topic. Mirrored from Hugging Face (JailbreakBench/JBB-Behaviors) into
this repo's data-v2 release; SHA-256 pinned below.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
RAW = Path(os.environ.get("PUBLIC_V2_DIR", ROOT / "data" / "raw" / "public_v2"))
BASE = "https://github.com/bu09572-gif/ai-ml-portfolio/releases/download/data-v2/"
SHA = {"harmful": "7007d6b836b1636d0b4cc5e798f05bd22e202354c61801e67751f2592f0942d5",
       "benign": "0d0b3188160b217608a95d1bc8f3721c3df940586081d7d0261bc4a0e5b97149"}


def _parquet(split: str) -> pd.DataFrame:
    name = f"jbb_behaviors__{split}__0.parquet"
    p = RAW / name
    if not p.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        p.write_bytes(requests.get(BASE + name, timeout=120).content)
    if hashlib.sha256(p.read_bytes()).hexdigest() != SHA[split]:
        raise RuntimeError(f"{name} does not match the mirrored copy")
    return pd.read_parquet(p)


def behaviors() -> dict[str, pd.DataFrame]:
    return {"harmful": _parquet("harmful"), "benign": _parquet("benign")}
