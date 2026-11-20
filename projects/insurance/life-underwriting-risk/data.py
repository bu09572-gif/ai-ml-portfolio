"""NHANES 2005-2010 (CDC National Health and Nutrition Examination Survey) linked to death records.

Each NHANES participant answers health questions, has a physical exam (blood pressure, weight) and gives
blood. NCHS links adult participants to the National Death Index; the public-use linked mortality file
gives vital status and months of follow-up to December 31, 2019. I use three survey cycles (2005-06,
2007-08, 2009-10), so every participant has at least 9 years of follow-up.

All 45 files are mirrored from wwwn.cdc.gov and ftp.cdc.gov into this repo's data-v2 release. The
manifest (one SHA-256 per file) is pinned below, and every file is checked against it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
RAW = Path(os.environ.get("PUBLIC_V2_DIR", ROOT / "data" / "raw" / "public_v2"))
BASE = "https://github.com/bu09572-gif/ai-ml-portfolio/releases/download/data-v2/"
MANIFEST_SHA = "d73f0310e19b61a3c3ae817e8c0489ac1c63ff9e756ee49fec2d34fd05198f05"
CYCLES = {"D": 2005, "E": 2007, "F": 2009}
HORIZON_MONTHS = 96      # 8-year mortality; every cycle has more follow-up than this

COLS = {
    "DEMO": {"RIAGENDR": "sex", "RIDAGEYR": "age", "RIDRETH1": "race_eth", "INDFMPIR": "income_poverty_ratio",
             "WTMEC2YR": "exam_weight"},
    "BMX": {"BMXBMI": "bmi", "BMXWAIST": "waist_cm"},
    "BPX": {"BPXSY1": "sbp1", "BPXSY2": "sbp2", "BPXSY3": "sbp3", "BPXDI1": "dbp1", "BPXDI2": "dbp2", "BPXDI3": "dbp3",
            "BPXPLS": "pulse"},
    "BPQ": {"BPQ020": "told_high_bp"},
    "SMQ": {"SMQ020": "smoked_100", "SMQ040": "smokes_now"},
    "DIQ": {"DIQ010": "told_diabetes"},
    "MCQ": {"MCQ160B": "heart_failure", "MCQ160C": "coronary_disease", "MCQ160E": "heart_attack", "MCQ160F": "stroke",
            "MCQ220": "cancer", "MCQ160G": "emphysema"},
    "TCHOL": {"LBXTC": "total_chol"},
    "HDL": {"LBDHDD": "hdl"},
    "GHB": {"LBXGH": "hba1c"},
    "BIOPRO": {"LBXSAL": "albumin", "LBXSCR": "creatinine", "LBXSGTSI": "ggt", "LBXSATSI": "alt", "LBXSGL": "glucose",
               "LBXSAPSI": "alk_phos", "LBXSBU": "bun", "LBXSUA": "uric_acid"},
    "CBC": {"LBXWBCSI": "wbc", "LBXHGB": "hemoglobin", "LBXRDW": "rdw", "LBXMCVSI": "mcv", "LBXPLTSI": "platelets"},
}


def _manifest() -> dict:
    return json.loads(_file("manifest_nhanes.json", MANIFEST_SHA))


def _file(name: str, sha: str | None = None) -> bytes:
    p = RAW / name
    if not p.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        p.write_bytes(requests.get(BASE + name, timeout=300).content)
    b = p.read_bytes()
    want = sha or _manifest()[name]["sha256"]
    if hashlib.sha256(b).hexdigest() != want:
        raise RuntimeError(f"{name} does not match the mirrored copy")
    return b


def _xpt(table: str, cycle: str) -> pd.DataFrame:
    _file(f"nhanes_{table}_{cycle}.xpt")
    d = pd.read_sas(RAW / f"nhanes_{table}_{cycle}.xpt", format="xport", encoding="latin-1")
    keep = {k: v for k, v in COLS[table].items() if k in d.columns}
    return d[["SEQN", *keep]].rename(columns=keep)


def _mortality(cycle: str) -> pd.DataFrame:
    """NCHS public-use linked mortality file (fixed width, layout from the NCHS reading programs)."""
    _file(f"nhanes_mort_{cycle}.dat")
    spec = [(0, 14), (14, 15), (15, 16), (16, 19), (42, 45), (45, 48)]
    names = ["SEQN", "eligstat", "mortstat", "ucod_leading", "permth_int", "permth_exm"]
    d = pd.read_fwf(RAW / f"nhanes_mort_{cycle}.dat", colspecs=spec, names=names, na_values=["."], dtype=str)
    return d.apply(pd.to_numeric, errors="coerce")


def _yes(s: pd.Series) -> pd.Series:
    return s.map({1.0: 1.0, 2.0: 0.0})        # NHANES codes 1 yes, 2 no, 7/9 refused/don't know -> missing


def cohort(min_age: int = 40, max_age: int = 79) -> pd.DataFrame:
    """One row per adult examined at age 40-79, eligible for linkage, with 8-year death outcome."""
    frames = []
    for c, year in CYCLES.items():
        d = _xpt("DEMO", c)
        for t in COLS:
            if t != "DEMO":
                d = d.merge(_xpt(t, c), on="SEQN", how="left")
        d = d.merge(_mortality(c), on="SEQN", how="left").assign(cycle=year)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d = d[(d.age >= min_age) & (d.age <= max_age) & (d.eligstat == 1) & d.permth_exm.notna()].copy()
    d["died_8y"] = ((d.mortstat == 1) & (d.permth_exm <= HORIZON_MONTHS)).astype(int)
    d["female"] = (d.sex == 2).astype(int)
    for c in ["told_high_bp", "told_diabetes", "heart_failure", "coronary_disease", "heart_attack", "stroke", "cancer",
              "emphysema", "smoked_100"]:
        d[c] = _yes(d[c])
    d["current_smoker"] = np.where(d.smoked_100 == 0, 0, d.smokes_now.map({1.0: 1, 2.0: 1, 3.0: 0}))
    d["former_smoker"] = np.where(d.smoked_100 == 0, 0, d.smokes_now.map({1.0: 0, 2.0: 0, 3.0: 1}))
    d["sbp"] = d[["sbp1", "sbp2", "sbp3"]].mean(axis=1)
    d["dbp"] = d[["dbp1", "dbp2", "dbp3"]].replace(0, np.nan).mean(axis=1)
    d["chol_hdl_ratio"] = d.total_chol / d.hdl
    return d.reset_index(drop=True)
