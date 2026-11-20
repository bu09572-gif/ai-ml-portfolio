"""NYC Motor Vehicle Collisions, 2023 (NYPD crash reports, NYC Open Data): one table of crashes and one
of the vehicles in them, with each vehicle's maneuver, point of impact, damage and the police officer's
contributing factors. Downloaded through the Socrata API (datasets h9gi-nx95 and bm4k-52h4) and mirrored
into this repo's data-v2 release; SHA-256 pinned below.
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
SHA = {"nyc_collisions_crashes_2023.csv": "6cadb229da9ce0b1678e852c7cb4298584547e614af31a503c83d893cbb5f8ea",
       "nyc_collisions_vehicles_2023.csv": "0ba884688673cb558b31e8cc91151585bd9b405cf5ebf11abc21e3f1b0d0ecc9"}


def _csv(name: str) -> pd.DataFrame:
    p = RAW / name
    if not p.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        p.write_bytes(requests.get(BASE + name, timeout=600).content)
    if hashlib.sha256(p.read_bytes()).hexdigest() != SHA[name]:
        raise RuntimeError(f"{name} does not match the mirrored copy")
    return pd.read_csv(p, low_memory=False)


def crashes() -> pd.DataFrame:
    return _csv("nyc_collisions_crashes_2023.csv")


def vehicles() -> pd.DataFrame:
    return _csv("nyc_collisions_vehicles_2023.csv")
