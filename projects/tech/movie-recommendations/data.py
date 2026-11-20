"""MovieLens 1M (GroupLens; F. M. Harper and J. A. Konstan, ACM TiiS 2015): 1,000,209 ratings of 3,706 movies
by 6,040 users who joined MovieLens in 2000, with timestamps, movie genres and user age, sex and occupation.
Mirrored from files.grouplens.org into this repo's data-v2 release; SHA-256 pinned below.
"""
from __future__ import annotations

import hashlib
import io
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
RAW = Path(os.environ.get("PUBLIC_V2_DIR", ROOT / "data" / "raw" / "public_v2"))
URL = "https://github.com/bu09572-gif/ai-ml-portfolio/releases/download/data-v2/movielens_ml-1m.zip"
SHA = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"


def _zip() -> zipfile.ZipFile:
    p = RAW / "movielens_ml-1m.zip"
    if not p.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        p.write_bytes(requests.get(URL, timeout=300).content)
    b = p.read_bytes()
    if hashlib.sha256(b).hexdigest() != SHA:
        raise RuntimeError("movielens_ml-1m.zip does not match the mirrored copy")
    return zipfile.ZipFile(io.BytesIO(b))


def _dat(name: str, cols: list[str]) -> pd.DataFrame:
    raw = _zip().read(f"ml-1m/{name}").decode("latin-1")
    return pd.DataFrame([line.split("::") for line in raw.splitlines() if line], columns=cols)


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r = _dat("ratings.dat", ["user", "item", "rating", "ts"]).astype(int)
    m = _dat("movies.dat", ["item", "title", "genres"]).astype({"item": int})
    u = _dat("users.dat", ["user", "sex", "age", "occupation", "zip"]).astype({"user": int, "age": int, "occupation": int})
    return r, m, u
