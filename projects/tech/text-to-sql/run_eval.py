"""Score the text-to-SQL assistant on the BIRD dev set by execution accuracy.

For every question it runs two configs, with and without BIRD's evidence hint, and records
whether the first query and the query after repairs return the same rows as the reference SQL.

    python run_eval.py --shard 0/20        # one CI shard
    python run_eval.py --merge             # combine shards, write the summary
    python run_eval.py --limit 5           # quick local try
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import data
import pandas as pd
import sqlagent

from docqa import llm

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CONFIGS = {"with_hint": True, "no_hint": False}
GOLD_TIME_LIMIT_S = 120


def tag() -> str:
    return llm.model_name().replace(":", "-").replace("/", "-")


def items() -> list[dict]:
    qs = data.questions()
    for i, q in enumerate(qs):
        q.setdefault("question_id", i)
    # grouped by database so a shard reuses one schema prompt (Ollama keeps the shared prefix cached)
    return sorted(qs, key=lambda q: (q["db_id"], q["question_id"]))


def score_one(q: dict) -> dict:
    db = data.db_path(q["db_id"])
    row = {"question_id": q["question_id"], "db_id": q["db_id"], "difficulty": q.get("difficulty"),
           "question": q["question"], "hint": q.get("evidence", ""), "gold_sql": q["SQL"], "gold_error": None}
    try:
        gold = sqlagent.execute(db, q["SQL"], time_limit=GOLD_TIME_LIMIT_S)
        row["gold_rows"] = len(gold)
    except sqlite3.Error as e:      # a reference query that fails here is left out of scoring
        gold, row["gold_error"] = None, str(e)[:200]
    for name, use_hint in CONFIGS.items():
        try:
            r = sqlagent.answer(q["question"], q.get("evidence", ""), db, use_evidence=use_hint)
        except Exception as e:  # noqa: BLE001 - a failed model call ends this config for this question only
            row[name] = {"model_error": type(e).__name__}
            continue
        first, last = r.attempts[0], r.final
        row[name] = {
            "first_sql": first.sql, "final_sql": last.sql, "tries": len(r.attempts),
            "first_error": first.error, "final_error": last.error,
            "first_correct": gold is not None and sqlagent.same_result(first.rows, gold),
            "final_correct": gold is not None and sqlagent.same_result(last.rows, gold),
            "input_tokens": r.input_tokens, "output_tokens": r.output_tokens, "latency_s": round(r.latency_s, 2)}
    return row


def _mark(r: dict) -> str:
    if "model_error" in r:
        return "ERR"
    return ("ok" if r["final_correct"] else "--") + ("*" if r["tries"] > 1 else " ")


def run(qs: list[dict], out_path: Path) -> None:
    with out_path.open("w") as fh:
        for n, q in enumerate(qs, 1):
            row = score_one(q)
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            print(f"[{n}/{len(qs)}] {q['db_id']:22s} {str(q.get('difficulty')):12s} "
                  + " ".join(f"{c}={_mark(row[c])}" for c in CONFIGS), flush=True)


def summarize(df: pd.DataFrame) -> dict:
    scored = df[df.gold_error.isna()]
    s = {"model": llm.model_name(), "questions": len(df), "scored": len(scored),
         "reference_sql_failed": int(df.gold_error.notna().sum()), "configs": {}}
    for c in CONFIGS:
        r = pd.json_normalize(scored[c].tolist()).set_index(scored.index)
        r["difficulty"] = scored.difficulty
        for col in ["first_correct", "final_correct"]:
            r[col] = r.get(col, False).fillna(False).astype(bool)
        tries = r.get("tries", pd.Series(1, index=r.index)).fillna(1)
        s["configs"][c] = {
            "execution_accuracy": round(r.final_correct.mean(), 4),
            "execution_accuracy_first_try": round(r.first_correct.mean(), 4),
            "by_difficulty": r.groupby("difficulty").final_correct.mean().round(4).to_dict(),
            "by_difficulty_first_try": r.groupby("difficulty").first_correct.mean().round(4).to_dict(),
            "by_database": r.groupby(scored.db_id).final_correct.mean().round(3).to_dict(),
            "first_query_errored": round(r.get("first_error", pd.Series(dtype=object)).notna().mean(), 4),
            "final_query_errored": round(r.get("final_error", pd.Series(dtype=object)).notna().mean(), 4),
            "repaired_questions": int((tries > 1).sum()),
            "fixed_by_repair": int((r.final_correct & ~r.first_correct).sum()),
            "model_errors": int(r["model_error"].notna().sum()) if "model_error" in r else 0,
            "median_latency_s": round(float(r.latency_s.median()), 1) if "latency_s" in r else None,
            "mean_input_tokens": round(float(r.input_tokens.mean())) if "input_tokens" in r else None,
        }
    s["counts_by_difficulty"] = scored.difficulty.value_counts().to_dict()
    return json.loads(json.dumps(s, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def merge() -> dict:
    files = sorted(OUT.glob(f"bird_dev_{tag()}_shard*.jsonl")) or [OUT / f"bird_dev_{tag()}.jsonl"]
    df = pd.concat([pd.read_json(f, lines=True, dtype=False) for f in files], ignore_index=True)
    df = df.drop_duplicates("question_id").sort_values("question_id")
    s = summarize(df)
    (OUT / f"summary_{tag()}.json").write_text(json.dumps(s, indent=2))
    df.to_json(OUT / f"bird_dev_{tag()}.jsonl", orient="records", lines=True)
    for f in OUT.glob(f"bird_dev_{tag()}_shard*.jsonl"):
        f.unlink()
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    if a.merge:
        print(json.dumps(merge(), indent=2))
        return
    qs, name = items(), f"bird_dev_{tag()}.jsonl"
    if a.shard:   # contiguous slices, so each shard sees only one or two databases
        i, n = map(int, a.shard.split("/"))
        size = -(-len(qs) // n)
        qs, name = qs[i * size:(i + 1) * size], f"bird_dev_{tag()}_shard{i}.jsonl"
    if a.limit:
        qs = qs[: a.limit]
    run(qs, OUT / name)
    if not a.shard:
        print(json.dumps(merge(), indent=2))


if __name__ == "__main__":
    main()
