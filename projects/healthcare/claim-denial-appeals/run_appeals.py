"""Draft and score appeal letters for the prior authorization requests whose checklist found evidence.

Needs ../prior-auth-assistant/results/{criteria,checks}.jsonl from that project's eval.

    python run_appeals.py --shard 0/5
    python run_appeals.py --merge       # -> results/appeals.jsonl, results/appeals_summary.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import appeal
import pandas as pd

HERE = Path(__file__).resolve().parent
PA = HERE.parent / "prior-auth-assistant"
OUT = HERE / "results"
MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b")
NOTE_WORDS = 2500


def _pa_data():
    spec = importlib.util.spec_from_file_location("pa_data", PA / "data.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["pa_data"] = m
    spec.loader.exec_module(m)
    return m


def requests_with_evidence() -> tuple[list[dict], int]:
    pa = _pa_data()
    crit = json.loads((PA / "results" / "criteria.json").read_text())
    lcds = pa.lcds().set_index("lcd_id")
    notes = pa.service_notes().set_index("note_id")
    checks = pd.read_json(PA / "results" / "checks.jsonl", lines=True)
    checks = checks[checks.pairing == "matched"]
    out = []
    for _, r in checks.iterrows():
        ev = [c for c in r.checks if c.get("status") == "met"]
        if not ev:
            continue
        c = crit[r.service]
        out.append({"note_id": int(r.note_id), "service": r.service, "lcd": c["display_id"], "title": c["title"],
                    "policy": lcds.loc[c["lcd_id"]].indication, "note": " ".join(notes.loc[r.note_id].text.split()[:NOTE_WORDS]),
                    "criteria": [x["criterion"] for x in r.checks], "evidence": ev})
    return out, len(checks)


def run(shard: str) -> None:
    reqs, _ = requests_with_evidence()
    i, n = map(int, shard.split("/"))
    with (OUT / f"appeals_shard{i}.jsonl").open("w") as fh:
        for k, q in enumerate(reqs[i::n]):
            row = {"note_id": q["note_id"], "service": q["service"], "lcd": q["lcd"], "evidence_items": len(q["evidence"])}
            prompts = {"grounded": appeal.grounded_prompt(q["service"], q["lcd"], q["title"], q["evidence"]),
                       "free": appeal.free_prompt(q["service"], q["lcd"], q["title"], q["criteria"], q["note"])}
            for mode, prompt in prompts.items():
                try:
                    c = appeal.draft(prompt, MODEL)
                    row[mode] = {"letter": c.text, "latency_s": round(c.latency_s, 1)} | appeal.score(c.text, q["note"], q["policy"], q["evidence"])
                except Exception as e:  # noqa: BLE001 - one failed call skips one letter
                    row[mode] = {"error": type(e).__name__}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(k, flush=True)


def merge() -> dict:
    files = sorted(OUT.glob("appeals_shard*.jsonl"))
    df = pd.concat([pd.read_json(f, lines=True) for f in files if f.stat().st_size], ignore_index=True)
    df.to_json(OUT / "appeals.jsonl", orient="records", lines=True)
    for f in files:
        f.unlink()
    _, matched = requests_with_evidence()
    s = {"model": MODEL, "matched_requests": matched, "requests_with_evidence": len(df), "modes": {}}
    for mode in ["grounded", "free"]:
        r = pd.json_normalize(df[mode].tolist())
        r = r[r.get("error", pd.Series(index=r.index, dtype=object)).isna()]
        s["modes"][mode] = {
            "letters": len(r), "quotes": int(r.quotes.sum()),
            "quote_verified_rate": round(float(r.quotes_verified.sum() / max(r.quotes.sum(), 1)), 3),
            "letters_with_an_unverified_quote": int((r.quotes > r.quotes_verified).sum()),
            "letters_with_unsupported_numbers": int((r.unsupported_numbers.map(len) > 0).sum()),
            "evidence_used_mean": round(float(r.evidence_used.mean()), 3),
            "median_words": float(r.words.median())}
    (OUT / "appeals_summary.json").write_text(json.dumps(s, indent=2, default=float))
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(merge(), indent=2, default=float)) if a.merge else run(a.shard)


if __name__ == "__main__":
    main()
