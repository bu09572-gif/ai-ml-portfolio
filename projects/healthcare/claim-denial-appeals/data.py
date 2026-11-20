"""CMS Transparency in Coverage public use file, plan year 2024 (HealthCare.gov individual market).

Every issuer on HealthCare.gov reports, per plan, the claims it received and denied, the denials by
reason, and per issuer the appeals filed and overturned. Source:
https://download.cms.gov/marketplace-puf/2024/transparency-in-coverage-puf.zip, mirrored into this
repo's data-v2 release (CMS's host is not reachable from my build machine). SHA-256 pinned below.
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
URL = "https://github.com/bu09572-gif/ai-ml-portfolio/releases/download/data-v2/cms_tic_puf_2024.zip"
SHA = "64adb2cb99150f56dc869d6461c4a81fa8261b08a2714c0d268ad58b1100d3c0"
SHEET = "Transparency 2024 - Ind QHP"
P = "Plan_Number_Claims_Denied_"
REASONS = {
    "Referral_Required": "referral required", "Due_To_Out_Of_Network": "out of network",
    "Services_Excluded": "service excluded", "Not_Medically_Necessary_Excluding_Behavioral_Health": "not medically necessary",
    "Not_Medically_Necessary_Behavioral_Health_Only": "not medically necessary (behavioral health)",
    "Due_To_Enrolle_Benefit_Limit_Reached": "benefit limit reached", "Due_To_Member_Not_Covered": "member not covered",
    "Due_To_Investigational_Experimental_Cosmetic_Proceduce": "investigational, experimental or cosmetic",
    "Due_To_Administrative_Reason": "administrative", "Other": "other"}


def _zip() -> bytes:
    p = RAW / "cms_tic_puf_2024.zip"
    if not p.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        p.write_bytes(requests.get(URL, timeout=300).content)
    b = p.read_bytes()
    if hashlib.sha256(b).hexdigest() != SHA:
        raise RuntimeError("cms_tic_puf_2024.zip does not match the mirrored copy")
    return b


def plans() -> pd.DataFrame:
    """One row per medical plan. Counts are numbers; '**' (suppressed, small cell) and '*' become NaN."""
    z = zipfile.ZipFile(io.BytesIO(_zip()))
    d = pd.read_excel(io.BytesIO(z.read("Transparency_In_Coverage_PUF.xlsx")), SHEET, header=2, dtype=str)
    d.columns = [c.strip('"') for c in d.columns]
    keep = ["State", "Issuer_Name", "Issuer_ID", "Plan_ID", "New_or_Returning_Plan", "Plan_Type", "Metal_Level"]
    out = d[keep].copy()
    for c in d.columns:
        if c.startswith(("Issuer_Claims", "Issuer_Internal", "Issuer_External", "Issuer_Number", "Issuer_Percent", "Plan_Number")):
            out[c] = pd.to_numeric(d[c], errors="coerce")
    return out.rename(columns={P + k: "denied_" + k.lower() for k in REASONS})
