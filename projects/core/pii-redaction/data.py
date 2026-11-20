"""Text Anonymization Benchmark (TAB): 1,268 real European Court of Human Rights cases,
annotated by hand with every personal identifier and whether it must be masked.
Pilán et al. 2022, https://github.com/NorskRegnesentral/text-anonymization-benchmark (MIT).

    python data.py      # -> data/raw/tab/{echr_train,echr_dev,echr_test}.json + evaluation.py
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[3]
RAW = Path(os.environ.get("TAB_DIR", ROOT / "data" / "raw" / "tab"))
BASE = "https://raw.githubusercontent.com/NorskRegnesentral/text-anonymization-benchmark/master"
FILES = {  # sha256 of the v1.0 files, so a changed upstream file is noticed
    "echr_train.json": "4aba41f8ac305ff9e93dd6f0bbc16756e57e9ace396c827931fab70e18d8c6a6",
    "echr_dev.json": "8c3c7306f46b8d54debeb38ae11d8b0b8bcf4bdccbc3b6f13c12ad7be16893ec",
    "echr_test.json": "cd0f0f15f84a8739654c7cf30c6be8ce27b051ef73974d39d792a0cb8c846379",
    "evaluation.py": None,   # the benchmark's official scorer
    "LICENSE.txt": None,
}
MASK = {"DIRECT", "QUASI"}


def download(dest: Path = RAW) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for name, sha in FILES.items():
        p = dest / name
        if not p.exists():
            r = requests.get(f"{BASE}/{name}", timeout=120)
            r.raise_for_status()
            p.write_bytes(r.content)
        if sha and hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            raise RuntimeError(f"{name}: checksum changed upstream")
    return dest


def load(split: str) -> list[dict]:
    return json.loads((download() / f"echr_{split}.json").read_text())


def gold_spans(doc: dict, annotator: str | None = None) -> list[tuple[int, int, str]]:
    """(start, end, entity_type) of mentions that must be masked, from one annotator."""
    ann = doc["annotations"]
    a = ann[annotator] if annotator else ann[sorted(ann)[0]]
    return sorted({(m["start_offset"], m["end_offset"], m["entity_type"])
                   for m in a["entity_mentions"] if m["identifier_type"] in MASK})


if __name__ == "__main__":
    download()
    for s in ["train", "dev", "test"]:
        d = load(s)
        n = sum(len(gold_spans(x)) for x in d)
        print(f"{s}: {len(d)} cases, {n} mentions to mask, {sum(len(x['text']) for x in d) / 1e6:.1f}M characters")
