"""Two public medical QA benchmarks with expert answers.

  MedQA (Jin et al., 2021): 1,273 USMLE-style test questions, 4 options (GBaker/MedQA-USMLE-4-options
          on Hugging Face, mirrored to this repo's data-v2 release with SHA-256 pinned)
  PubMedQA (Jin et al., 2019): 500 expert-labelled test questions, answered yes / no / maybe
          from a PubMed abstract with its conclusion removed (github.com/pubmedqa/pubmedqa)
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
RAW = Path(os.environ.get("PUBLIC_V2_DIR", ROOT / "data" / "raw" / "public_v2"))
MEDQA = ("https://github.com/bu09572-gif/ai-ml-portfolio/releases/download/data-v2/medqa_usmle_4opt__test__0.parquet",
         "a504b21e1412ee17c231c99f4b1aaab0518eb0fc681fcf0c0f87f11afe5a86c2")
PUBMEDQA = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/"


def _get(url: str, name: str) -> bytes:
    p = RAW / name
    if not p.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        p.write_bytes(r.content)
    return p.read_bytes()


def medqa() -> pd.DataFrame:
    raw = _get(MEDQA[0], "medqa_usmle_4opt__test__0.parquet")
    if hashlib.sha256(raw).hexdigest() != MEDQA[1]:
        raise RuntimeError("MedQA file does not match the mirrored copy")
    d = pd.read_parquet(RAW / "medqa_usmle_4opt__test__0.parquet")
    d["id"] = [f"medqa-{i}" for i in range(len(d))]
    d["options"] = d.options.map(lambda o: {k: o[k] for k in "ABCD"})
    return d[["id", "question", "options", "answer_idx", "meta_info"]]


def pubmedqa() -> pd.DataFrame:
    gold = json.loads(_get(PUBMEDQA + "test_ground_truth.json", "pubmedqa_test_ground_truth.json"))
    allq = json.loads(_get(PUBMEDQA + "ori_pqal.json", "pubmedqa_ori_pqal.json"))
    rows = [{"id": f"pubmedqa-{k}", "question": allq[k]["QUESTION"], "context": " ".join(allq[k]["CONTEXTS"]),
             "answer": v} for k, v in gold.items()]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(len(medqa()), "MedQA test;", pubmedqa().answer.value_counts().to_dict())
