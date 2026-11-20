"""Answer and grade FinanceBench questions with the analyst, then compare with plain Q&A.

    python run_eval.py --shard 0/10            # one CI shard of the 50 metric questions
    python run_eval.py --merge                 # combine shards, write the summary
    python run_eval.py --types all --limit 5   # quick local try on any question type
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from analyst import Analyst

from docpipe import storage
from docqa import llm
from docqa.retrieve import Retriever
from evalkit import metrics as M
from evalkit.dataset import from_financebench
from evalkit.run import summarize

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
ROOT = HERE.parents[2]
BASELINE = ROOT / "evals" / "results" / "financebench" / "answers_ollama_qwen2.5-3b.jsonl"


def tag() -> str:
    return f"{llm.PROVIDER}_{llm.model_name().replace(':', '-').replace('/', '-')}"


def run(items, out_path: Path) -> None:
    chunks = pd.read_parquet(storage.path("gold", "financebench", "chunks.parquet"))
    pages = pd.read_parquet(storage.path("silver", "financebench", "pages.parquet"))
    an = Analyst(Retriever(chunks), pages)
    with out_path.open("w") as fh:
        for i, it in enumerate(items, 1):
            res = an.ask(it.question)
            a = res.answer
            gold = {M.page_key(e["doc_id"], e["page_num"]) for e in it.gold_evidence}
            g = M.judge(it.question, it.gold_answer, a.text)
            cited = [M.page_key(c.doc_id, c.page_num) for c in a.citations]
            numeric_text = a.text.splitlines()[0] if res.route == "calculator" and a.text else None
            row = {"id": it.id, "question": it.question, "gold_answer": it.gold_answer, "answer": a.text,
                   "question_type": it.tags.get("question_type"), "grade": g["grade"], "judge_reason": g["reason"],
                   # for calculator answers, check only the result line, not the inputs listed under it
                   "numeric_text": numeric_text,
                   "numeric_match": M.numeric_match(numeric_text or a.text, it.gold_answer),
                   "retrieval_hit": bool(set(cited) & gold),
                   "citations": cited, "citation_page_precision": M.citation_page_precision(cited, gold),
                   "refused": a.refused, "input_tokens": a.input_tokens, "output_tokens": a.output_tokens,
                   "cost_usd": a.cost_usd, "latency_s": a.latency_s, "provider": a.provider, "model": a.model,
                   "judge_model": g["judge_model"], "config": "analyst", "k": 3,
                   "route": res.route, "llm_calls": res.llm_calls, "plan": res.plan,
                   "inputs": [vars(x) for x in res.inputs], "value": res.value}
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            print(f"[{i}/{len(items)}] {res.route:10s} {g['grade']:9s} num={row['numeric_match']} "
                  f"{a.latency_s:5.1f}s {it.id}", flush=True)


def merge() -> dict:
    files = sorted(OUT.glob(f"analyst_{tag()}_shard*.jsonl")) or [OUT / f"analyst_{tag()}.jsonl"]
    df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    s = summarize(df)
    calc = df[df.route == "calculator"]
    s["analyst"] = {
        "answered_by_calculator": int(len(calc)), "fell_back_to_plain_qa": int((df.route == "fallback").sum()),
        "calculator_numeric_match": round(float(calc.numeric_match.mean()), 3) if len(calc) else None,
        "llm_calls_mean": round(float(df.llm_calls.mean()), 2),
    }
    if BASELINE.exists():   # same questions, plain Q&A, same answer and judge models
        b = pd.read_json(BASELINE, lines=True)
        b = b[b.id.isin(df.id)]
        s["plain_qa_same_questions"] = {k: summarize(b)[k] for k in
                                         ["n", "accuracy", "refusal_rate", "incorrect_rate", "citation_page_precision"]}
    s = json.loads(json.dumps(s, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    (OUT / f"summary_{tag()}.json").write_text(json.dumps(s, indent=2))
    df.to_json(OUT / f"analyst_{tag()}.jsonl", orient="records", lines=True)
    for f in OUT.glob(f"analyst_{tag()}_shard*.jsonl"):
        f.unlink()
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="metrics-generated")
    ap.add_argument("--shard")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    if a.merge:
        print(json.dumps(merge(), indent=2))
        return
    items = [it for it in from_financebench() if a.types == "all" or it.tags["question_type"] == a.types]
    name = f"analyst_{tag()}.jsonl"
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        items, name = items[i::n], f"analyst_{tag()}_shard{i}.jsonl"
    if a.limit:
        items = items[: a.limit]
    run(items, OUT / name)
    if not a.shard:
        print(json.dumps(merge(), indent=2))


if __name__ == "__main__":
    main()
