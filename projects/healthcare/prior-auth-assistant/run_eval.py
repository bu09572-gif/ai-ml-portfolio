"""Prior authorization checklist eval, run in CI with a free local model.

    python run_eval.py --criteria         # LCD -> quoted criteria, results/criteria.json
    python run_eval.py --shard 0/10       # check each note against its own service's criteria and,
                                          # as a control, against another service's criteria
    python run_eval.py --merge            # results/checks.jsonl, results/summary.json

The control pairing has no right answer per criterion, but a working checker should almost never
say "met" there: a cervical fusion note does not document blepharoplasty criteria.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import criteria as cr
import data
import pandas as pd

OUT = Path(__file__).resolve().parent / "results"
MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b")
NOTE_WORDS = 2500


def policy_lcds() -> dict:
    """One active LCD per service (the lowest id among contractors publishing that policy)."""
    lc = data.lcds()
    out = {}
    for svc, (title, _) in data.SERVICES.items():
        x = lc[(lc.title == title) & (lc.status == "A")].sort_values("lcd_id").iloc[0]
        out[svc] = {"lcd_id": int(x.lcd_id), "display_id": f"L{int(x.lcd_id)}", "title": title, "text": x.indication}
    return out


def run_criteria() -> None:
    res = {}
    for svc, p in policy_lcds().items():
        c = cr.extract(p["text"], MODEL)
        res[svc] = {k: p[k] for k in ("lcd_id", "display_id", "title")} | {
            "passages": len(cr.passages(p["text"])), "criteria": c}
        print(svc, len(c), sum(x["verified"] for x in c), flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "criteria.json").write_text(json.dumps(res, indent=2))


def pairs() -> list[dict]:
    notes = data.service_notes()
    svcs = list(data.SERVICES)
    out = []
    for _, n in notes.iterrows():
        other = svcs[(svcs.index(n.service) + 3) % len(svcs)]
        text = " ".join(n.text.split()[:NOTE_WORDS])
        out += [{"note_id": int(n.note_id), "service": n.service, "policy": n.service, "pairing": "matched", "text": text},
                {"note_id": int(n.note_id), "service": n.service, "policy": other, "pairing": "control", "text": text}]
    return out


def run_checks(shard: str) -> None:
    crit = json.loads((OUT / "criteria.json").read_text())
    i, n = map(int, shard.split("/"))
    todo = pairs()[i::n]
    with (OUT / f"checks_shard{i}.jsonl").open("w") as fh:
        for k, p in enumerate(todo):
            pool = [c["criterion"] for c in crit[p["policy"]]["criteria"] if c["verified"]]
            checks = []
            for c in cr.relevant(pool, p["text"]):
                try:
                    checks.append(cr.check(p["text"], c, MODEL))
                except Exception as e:  # noqa: BLE001 - one failed call marks one criterion
                    checks.append({"criterion": c, "status": None, "error": type(e).__name__})
            row = {k2: p[k2] for k2 in ("note_id", "service", "policy", "pairing")}
            fh.write(json.dumps(row | {"checks": checks, "decision": cr.decision(checks)}) + "\n")
            fh.flush()
            print(k, len(todo), flush=True)


def summarize(df: pd.DataFrame, crit: dict) -> dict:
    out = {"model": MODEL, "notes": int(df.note_id.nunique()), "policies": {}}
    for svc, c in crit.items():
        out["policies"][svc] = {"lcd": c["display_id"], "passages": c["passages"], "criteria_extracted": len(c["criteria"]),
                                "criteria_quote_verified": sum(x["verified"] for x in c["criteria"])}
    allc = [x for x in crit.values() for x in x["criteria"]]
    out["criteria_quote_verified_rate"] = round(sum(x["verified"] for x in allc) / max(len(allc), 1), 3)
    for pairing, g in df.groupby("pairing"):
        ch = pd.DataFrame([c for cs in g.checks for c in cs])
        claimed = ch[ch.model_status.isin(["met", "not_met"])]
        out[pairing] = {
            "requests": len(g), "criteria_checked": len(ch),
            "model_status_share": ch.model_status.value_counts(normalize=True).round(3).to_dict(),
            "final_status_share": ch.status.value_counts(normalize=True).round(3).to_dict(),
            "met_or_not_met_with_verified_quote": round(float(claimed.quote_verified.mean()), 3) if len(claimed) else None,
            "decisions": g.decision.value_counts().to_dict(),
            "median_latency_s": float(ch.latency_s.median()) if "latency_s" in ch else None}
    return json.loads(json.dumps(out, default=float))


def merge() -> dict:
    files = sorted(OUT.glob("checks_shard*.jsonl"))
    df = pd.concat([pd.read_json(f, lines=True) for f in files if f.stat().st_size], ignore_index=True)
    df.to_json(OUT / "checks.jsonl", orient="records", lines=True)
    for f in files:
        f.unlink()
    s = summarize(df, json.loads((OUT / "criteria.json").read_text()))
    (OUT / "summary.json").write_text(json.dumps(s, indent=2))
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--criteria", action="store_true")
    ap.add_argument("--shard")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(argv)
    if a.criteria:
        run_criteria()
    elif a.merge:
        print(json.dumps(merge(), indent=2))
    else:
        run_checks(a.shard)


if __name__ == "__main__":
    main()
