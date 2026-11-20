"""CodiEsp (Miranda-Escalada et al., CLEF eHealth 2020; CC BY 4.0): 1,000 published clinical case
reports, each coded by expert coders with ICD-10 diagnosis codes (CIE-10, the Spanish edition of
ICD-10-CM). The corpus ships with English translations of every case, which this project uses,
and the code list with English descriptions.

Both zips are mirrored from Zenodo into this repo's data-v2 release (Zenodo isn't reachable from
my build environment), with SHA-256 pinned here.

    python data.py
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
BASE = "https://github.com/bu09572-gif/ai-ml-portfolio/releases/download/data-v2/"
SHA = {"codiesp.zip": "52b290233906a2eb589ac7b1d9429adeac88f6a9cae8b0d3c180504afbe61688",
       "codiesp_codes.zip": "ed28f25e50fe958d660a07387414e98dbf9f86a172a1c418f11cc1cc449c9b8e"}


def _zip(name: str) -> zipfile.ZipFile:
    p = RAW / name
    if not p.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        p.write_bytes(requests.get(BASE + name, timeout=300).content)
    if hashlib.sha256(p.read_bytes()).hexdigest() != SHA[name]:
        raise RuntimeError(f"{name} does not match the mirrored copy")
    return zipfile.ZipFile(p)


def cases() -> pd.DataFrame:
    """One row per case: split, id, English text, list of diagnosis codes."""
    out = RAW / "codiesp_cases.parquet"
    if out.exists():
        return pd.read_parquet(out)
    z = _zip("codiesp.zip")
    rows = []
    for split in ["train", "dev", "test"]:
        root = f"final_dataset_v4_to_publish/{split}/"
        labels = pd.read_csv(io.BytesIO(z.read(root + f"{split}D.tsv")), sep="\t", header=None, names=["doc", "code"])
        codes = labels.groupby("doc").code.apply(lambda c: sorted(set(c.str.upper()))).to_dict()
        for name in z.namelist():
            if name.startswith(root + "text_files_en/") and name.endswith(".txt"):
                doc = Path(name).stem
                rows.append({"split": split, "doc": doc, "text": z.read(name).decode("utf-8"),
                             "codes": codes.get(doc, [])})
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    return df


def code_descriptions() -> pd.Series:
    z = _zip("codiesp_codes.zip")
    d = pd.read_csv(io.BytesIO(z.read("codiesp_codes/codiesp-D_codes.tsv")), sep="\t", header=None,
                    names=["code", "es", "en"], dtype=str, quoting=3)
    return d.dropna(subset=["en"]).assign(code=lambda x: x.code.str.upper().str.strip()).set_index("code").en


if __name__ == "__main__":
    c = cases()
    print(c.groupby("split").agg(cases=("doc", "size"), codes_per_case=("codes", lambda s: s.map(len).mean())))
    print(len(code_descriptions()), "ICD-10 diagnosis codes with English descriptions")
