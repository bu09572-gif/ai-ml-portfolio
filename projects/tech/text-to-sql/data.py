"""BIRD dev set (Li et al., 2023; https://bird-bench.github.io, CC BY-SA 4.0): 1,534 questions
written by people over 11 real SQLite databases (schools, banking, Formula 1, card games,
medical tests, ...), each with a reference SQL query and a difficulty label.

    python data.py     # -> data/raw/bird/dev.json + data/raw/bird/dev_databases/<db>/<db>.sqlite
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[3]
RAW = Path(os.environ.get("BIRD_DIR", ROOT / "data" / "raw" / "bird"))
URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
SHA256 = "cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630"


def download(dest: Path = RAW) -> Path:
    if (dest / "dev.json").exists() and (dest / "dev_databases").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    z = dest / "dev.zip"
    if not z.exists():
        with requests.get(URL, stream=True, timeout=600) as r:
            r.raise_for_status()
            with z.open("wb") as fh:
                for block in r.iter_content(1 << 20):
                    fh.write(block)
    h = hashlib.sha256()
    with z.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != SHA256:
        raise RuntimeError("dev.zip checksum changed upstream")
    with zipfile.ZipFile(z) as outer:
        for name in outer.namelist():
            base = Path(name).name
            if base in {"dev.json", "dev_tables.json"}:
                (dest / base).write_bytes(outer.read(name))
            elif base == "dev_databases.zip":
                inner = dest / base
                inner.write_bytes(outer.read(name))
                with zipfile.ZipFile(inner) as dbs:
                    for m in dbs.namelist():
                        if m.endswith(".sqlite") and not Path(m).name.startswith("."):
                            dbs.extract(m, dest)
                inner.unlink()
    return dest


def questions() -> list[dict]:
    return json.loads((download() / "dev.json").read_text())


def db_path(db_id: str) -> Path:
    return download() / "dev_databases" / db_id / f"{db_id}.sqlite"


if __name__ == "__main__":
    q = questions()
    from collections import Counter
    print(len(q), "questions;", Counter(x["difficulty"] for x in q), Counter(x["db_id"] for x in q))
