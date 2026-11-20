"""Answer and grade all 150 FinanceBench questions with the agent, then compare with plain Q&A.

    python run_eval.py --shard 0/10        # one CI shard
    python run_eval.py --merge             # combine shards, write the summary
    python run_eval.py --limit 5           # quick local try
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from copilot import Copilot

from docpipe import storage
from docqa import llm
from docqa.answer import Answer
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
    agent = Copilot(Retriever(chunks), pages)
    with out_path.open("w") as fh:
        for i, it in enumerate(items, 1):
            res = agent.ask(it.question)
            a = res.answer
            gold = {M.page_key(e["doc_id"], e["page_num"]) for e in it.gold_evidence}
            g = M.judge(it.question, it.gold_answer, a.text)
            cited = [M.page_key(c.doc_id, c.page_num) for c in a.citations]
            row = {"id": it.id, "question": it.question, "gold_answer": it.gold_answer, "answer": a.text,
                   "question_type": it.tags.get("question_type"), "grade": g["grade"], "judge_reason": g["reason"],
                   "numeric_match": M.numeric_match(a.text, it.gold_answer),
                   "retrieval_hit": bool({M.page_key(d, p) for d, p in res.pages_read} & gold),  # read it
                   "citations": cited, "citation_page_precision": M.citation_page_precision(cited, gold),
                   "refused": a.refused, "input_tokens": a.input_tokens, "output_tokens": a.output_tokens,
                   "cost_usd": a.cost_usd, "latency_s": a.latency_s, "provider": a.provider, "model": a.model,
                   "judge_model": g["judge_model"], "config": "agent", "k": 5,
                   "n_steps": len(res.steps), "actions": [s.action for s in res.steps],
                   "pages_read": [M.page_key(d, p) for d, p in res.pages_read],
                   "revised": res.revised, "guard": res.guard, "auto_cited": res.auto_cited,
                   "steps": [vars(s) for s in res.steps]}
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            print(f"[{i}/{len(items)}] steps={len(res.steps)} {g['grade']:9s} num={row['numeric_match']} "
                  f"{a.latency_s:5.1f}s {it.id}", flush=True)


def by_type(df: pd.DataFrame) -> dict:
    """Final grade (same rules as evalkit.run.summarize) averaged per question type."""
    refused = df.answer.map(lambda t: Answer(str(t)).refused)
    num = [M.numeric_match(str(a), str(g)) if not r else None for a, g, r in zip(df.answer, df.gold_answer, refused)]
    final = [M.final_grade(j, r, n) for j, r, n in zip(df.grade, refused, num)]
    return pd.Series([f == "correct" for f in final], index=df.index).groupby(df.question_type).mean().round(3).to_dict()


def merge() -> dict:
    files = sorted(OUT.glob(f"agent_{tag()}_shard*.jsonl")) or [OUT / f"agent_{tag()}.jsonl"]
    df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    s = summarize(df)
    acts = df.actions.explode()
    s["agent"] = {
        "steps_mean": round(float(df.n_steps.mean()), 2),
        "actions_total": acts.value_counts().to_dict(),
        "read_an_evidence_page": round(float(df.retrieval_hit.mean()), 3),
        "answers_sent_back_by_guardrail": int(df.revised.sum()),
        "citations_added_by_code": int(df.auto_cited.sum()) if "auto_cited" in df else 0,
        "refused_by_guardrail": df.guard[df.guard != ""].map(lambda g: g.split(":")[0]).value_counts().to_dict(),
        "accuracy_by_question_type": by_type(df),
    }
    if BASELINE.exists():   # same questions, plain Q&A, same answer and judge models
        b = pd.read_json(BASELINE, lines=True)
        b = b[b.id.isin(df.id)]
        s["plain_qa_same_questions"] = {k: summarize(b)[k] for k in
                                         ["n", "accuracy", "refusal_rate", "incorrect_rate", "citation_page_precision"]}
        s["plain_qa_same_questions"]["accuracy_by_question_type"] = by_type(b)
    s = json.loads(json.dumps(s, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    (OUT / f"summary_{tag()}.json").write_text(json.dumps(s, indent=2))
    df.to_json(OUT / f"agent_{tag()}.jsonl", orient="records", lines=True)
    for f in OUT.glob(f"agent_{tag()}_shard*.jsonl"):
        f.unlink()
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="all")
    ap.add_argument("--shard")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    if a.merge:
        print(json.dumps(merge(), indent=2))
        return
    items = [it for it in from_financebench() if a.types == "all" or it.tags["question_type"] == a.types]
    name = f"agent_{tag()}.jsonl"
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        items, name = items[i::n], f"agent_{tag()}_shard{i}.jsonl"
    if a.limit:
        items = items[: a.limit]
    run(items, OUT / name)
    if not a.shard:
        print(json.dumps(merge(), indent=2))


if __name__ == "__main__":
    main()
