"""Run the red-team grid in CI with free local models (Ollama), sharded.

    python run_eval.py --shard 0/20
    python run_eval.py --merge          # -> results/rows.jsonl, results/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import data
import pandas as pd
import redteam as rt

from docqa import llm

OUT = Path(__file__).resolve().parent / "results"
MODELS = os.environ.get("EVAL_MODELS", "qwen2.5:3b,llama3.2:3b").split(",")
JUDGE = os.environ.get("EVAL_JUDGE_MODEL", "llama-guard3:1b")


def cases() -> list[dict]:
    out = []
    for kind, df in data.behaviors().items():
        for _, b in df.iterrows():
            attacks = list(rt.ATTACKS) if kind == "harmful" else ["direct"]
            for a in attacks:
                out.append({"kind": kind, "behavior": b.Behavior, "category": b.Category, "attack": a,
                            "prompt": rt.prompt(a, b.Goal, b.Target), "goal": b.Goal})
    return out


def run(shard: str) -> None:
    i, n = map(int, shard.split("/"))
    todo = cases()[i::n]
    OUT.mkdir(exist_ok=True)
    with (OUT / f"rows_shard{i}.jsonl").open("w") as fh:
        for k, c in enumerate(todo):
            for model in MODELS:
                row = {k2: c[k2] for k2 in ("kind", "behavior", "category", "attack")} | {"model": model}
                try:
                    row["self_check_blocked"] = rt.self_check(c["prompt"], model)
                    for defence, system in [("none", rt.PLAIN_SYSTEM), ("safety_prompt", rt.SAFETY_SYSTEM)]:
                        r = llm.chat(system, c["prompt"], model=model, max_tokens=200)
                        row[defence] = {"refusal": rt.is_refusal(r.text), "unsafe": rt.guard_unsafe(c["goal"], r.text, JUDGE),
                                        "reply_words": len(r.text.split())}
                        if c["kind"] == "benign":    # replies to harmful requests are never saved or published
                            row[defence]["reply"] = r.text[:600]
                except Exception as e:  # noqa: BLE001 - one failed call skips one case
                    row["error"] = type(e).__name__
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            print(k, len(todo), flush=True)


def summarize(df: pd.DataFrame) -> dict:
    df = df[df.get("error", pd.Series(index=df.index, dtype=object)).isna()].copy()
    rows = []
    for _, r in df.iterrows():
        for defence in ["none", "safety_prompt", "self_check"]:
            base = r["none"] if defence == "self_check" else r[defence]
            blocked = defence == "self_check" and bool(r.self_check_blocked)
            rows.append({"model": r.model, "kind": r.kind, "attack": r.attack, "category": r.category, "defence": defence,
                         "unsafe": (not blocked) and base["unsafe"] is True,
                         "refused": blocked or base["refusal"], "judged": base["unsafe"] is not None or blocked})
    x = pd.DataFrame(rows)
    out = {"judge": JUDGE, "cases": int(len(df) / max(df.model.nunique(), 1)), "models": {}}
    for model, g in x.groupby("model"):
        h, b = g[g.kind == "harmful"], g[g.kind == "benign"]
        out["models"][model] = {
            "attack_success_rate": {d: {a: round(float(s.unsafe.mean()), 3) for a, s in gd.groupby("attack")} |
                                    {"all_attacks": round(float(gd.unsafe.mean()), 3)} for d, gd in h.groupby("defence")},
            "benign_refusal_rate": {d: round(float(gd.refused.mean()), 3) for d, gd in b.groupby("defence")},
            "benign_flagged_unsafe": {d: round(float(gd.unsafe.mean()), 3) for d, gd in b.groupby("defence")},
            "worst_categories_no_defence": h[h.defence == "none"].groupby("category").unsafe.mean().sort_values(ascending=False)
                                           .head(5).round(3).to_dict(),
            "judge_unreadable": int((~g.judged).sum())}
    return json.loads(json.dumps(out, default=float))


def merge() -> dict:
    files = sorted(OUT.glob("rows_shard*.jsonl"))
    df = pd.concat([pd.read_json(f, lines=True) for f in files if f.stat().st_size], ignore_index=True)
    df.to_json(OUT / "rows.jsonl", orient="records", lines=True)
    for f in files:
        f.unlink()
    s = summarize(df)
    (OUT / "summary.json").write_text(json.dumps(s, indent=2))
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(merge(), indent=2)) if a.merge else run(a.shard)


if __name__ == "__main__":
    main()
