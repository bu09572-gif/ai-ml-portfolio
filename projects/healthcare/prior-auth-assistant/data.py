"""Data for the prior authorization assistant, all public and real:

  CMS Medicare Coverage Database export (current LCDs): 952 Local Coverage Determinations with their
      coverage indications, documentation requirements, and the HCPCS codes some of them list
  MTSamples procedure and consult notes (see ../clinical-note-summarizer/data.py)

Both are mirrored into this repo's data-v2 release; SHA-256 pinned below. The LCD export is a
weekly CMS file, so the pinned copy is the one from 2026-11-16.
"""
from __future__ import annotations

import hashlib
import html
import io
import os
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
RAW = Path(os.environ.get("PUBLIC_V2_DIR", ROOT / "data" / "raw" / "public_v2"))
BASE = "https://github.com/bu09572-gif/ai-ml-portfolio/releases/download/data-v2/"
SHA = {"cms_current_lcd.zip": "6911b56a5c035369011bd621df03ea9d15c8abcecedd8fb288b10c13bbd70565",
       "mtsamples__train__0.parquet": "89d8e5eabe6769ea530b0c07da50f27c7a1605b836570e878cef5b956ca9ed80"}

# CMS's hospital outpatient prior authorization services, the LCD that sets each one's criteria,
# and the words that find matching MTSamples notes (title or description)
SERVICES = {
    "blepharoplasty": ("Blepharoplasty, Eyelid Surgery, and Brow Lift", r"blepharoplast|ptosis|brow ?lift"),
    "botulinum toxin": ("Botulinum Toxin Injections", r"botox|botulinum"),
    "varicose vein treatment": ("Varicose Veins of the Lower Extremity, Treatment of", r"varicose|vein strip|saphenous|phlebectom"),
    "cervical fusion": ("Cervical Fusion", r"cervical (?:discectomy|fusion)|anterior cervical|ACDF"),
    "facet joint injection": ("Facet Joint Interventions for Pain Management", r"facet (?:joint )?(?:injection|block)|medial branch"),
    "sleep testing": ("Polysomnography and Sleep Testing", r"polysomnogra|sleep study"),
}


def _file(name: str) -> Path:
    p = RAW / name
    if not p.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        p.write_bytes(requests.get(BASE + name, timeout=600).content)
    if hashlib.sha256(p.read_bytes()).hexdigest() != SHA[name]:
        raise RuntimeError(f"{name} does not match the mirrored copy")
    return p


def clean(x) -> str:
    t = re.sub(r"<(br|/p|/li|/h\d)[^>]*>", "\n", str(x or ""), flags=re.I)
    t = re.sub(r"<li[^>]*>", "\n- ", t, flags=re.I)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n", t)).strip()


def _lcd_tables() -> zipfile.ZipFile:
    outer = zipfile.ZipFile(_file("cms_current_lcd.zip"))
    return zipfile.ZipFile(io.BytesIO(outer.read("current_lcd_csv.zip")))


def lcds() -> pd.DataFrame:
    """One row per LCD id (latest version), with cleaned text fields."""
    out = RAW / "lcds.parquet"
    if out.exists():
        return pd.read_parquet(out)
    z = _lcd_tables()
    d = pd.read_csv(io.BytesIO(z.read("lcd.csv")), encoding="latin-1", low_memory=False)
    d = d.sort_values("lcd_version").drop_duplicates("lcd_id", keep="last")
    d = d.assign(indication=d.indication.map(clean), doc_reqs=d.doc_reqs.map(clean), title=d.title.str.strip())
    d = d[["lcd_id", "display_id", "title", "status", "indication", "doc_reqs"]].reset_index(drop=True)
    d.to_parquet(out, index=False)
    return d


def hcpcs_links() -> pd.DataFrame:
    """HCPCS codes that LCDs list directly (mostly equipment and supplies), as a retrieval test set."""
    z = _lcd_tables()
    h = pd.read_csv(io.BytesIO(z.read("lcd_x_hcpc_code.csv")), encoding="latin-1")
    return h.groupby(["hcpc_code_id", "long_description"]).lcd_id.apply(lambda s: sorted(set(s))).reset_index()


def notes() -> pd.DataFrame:
    d = pd.read_parquet(_file("mtsamples__train__0.parquet")).rename(columns={"Unnamed: 0": "note_id"})
    for c in ["description", "medical_specialty", "sample_name", "transcription", "keywords"]:
        d[c] = d[c].fillna("").str.strip()
    d["text"] = d.transcription.str.replace(r",\s*(?=[A-Z][A-Z /&-]{2,}:)", "\n", regex=True)
    return d


def service_notes(per_service: int = 8) -> pd.DataFrame:
    d = notes()
    rows = []
    for svc, (_, pattern) in SERVICES.items():
        m = d[(d.sample_name + " " + d.description).str.contains(pattern, case=False, regex=True)]
        m = m[m.text.str.len() > 400].drop_duplicates("text").head(per_service)
        rows.append(m.assign(service=svc))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    l = lcds()
    print(len(l), "LCDs;", l.title.nunique(), "titles;", len(hcpcs_links()), "HCPCS codes linked")
    s = service_notes()
    print(s.service.value_counts().to_dict())
    for svc, (title, _) in SERVICES.items():
        print(svc, "->", l[l.title.str.strip() == title].lcd_id.tolist())
