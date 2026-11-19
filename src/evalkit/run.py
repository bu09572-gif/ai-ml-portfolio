"""Run an evaluation and write results + a summary.

    python -m evalkit.run --dataset financebench                 # retrieval ablations (free, no API key)
    python -m evalkit.run --dataset financebench --answers       # + generate and grade answers (needs ANTHROPIC_API_KEY)
    python -m evalkit.run --dataset financebench --answers --limit 20

Outputs (under evals/results/<dataset>/):
    retrieval_summary.json     hit@1/5/10 and MRR for every retrieval config, overall and by question type
    answers_<config>.jsonl     one line per question: answer, citations, grade, tokens, cost, latency
    answers_summary.json       accuracy, refusal rate, numeric match, citation precision, cost, p50/p95 latency
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from docpipe import storage
from docqa.retrieve import Retriever

from . import metrics as M
from .dataset import DATASETS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evals" / "results"

# Each config changes one thing at a time, so the table reads as an ablation study.
RETRIEVAL_CONFIGS = {
    "bm25_shared_store":       dict(use_filters=False, expand=False, dedupe_pages=False),
    "+page_dedupe":            dict(use_filters=False, expand=False, dedupe_pages=True),
    "+metadata_filters":       dict(use_filters=True, expand=False, dedupe_pages=True),
    "+query_expansion":        dict(use_filters=True, expand=True, dedupe_pages=True),
    "single_document_oracle":  dict(oracle_doc=True, expand=True, dedupe_pages=True),
}
DEFAULT_CONFIG = "+query_expansion"
# Added only when sentence-transformers is installed (it downloads an embedding model):
DENSE_CONFIGS = {
    "dense_only+filters":      dict(use_filters=True, expand=False, dedupe_pages=True, mode="dense"),
    "hybrid_bm25_dense+filters": dict(use_filters=True, expand=True, dedupe_pages=True, mode="hybrid"),
}


def load_encoder():
    """Optional embedding model for dense/hybrid retrieval. Returns None if not installed."""
    name = os.environ.get("DOCQA_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    class Enc:
        def __init__(self):
            self.m = SentenceTransformer(name)

        def encode(self, texts):
            return self.m.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    return Enc()


def _search(r: Retriever, item, k: int, cfg: dict):
    cfg = dict(cfg)
    doc = item.doc_id if cfg.pop("oracle_doc", False) else None
    hits, _ = r.search(item.question, k=k, doc_id=doc, **cfg)
    return hits


def evidence_alignment(items, pages: pd.DataFrame) -> dict:
    """Data check: does each expert evidence snippet really appear on the page number we extracted?
    If page numbering were off by one, every retrieval score would be silently wrong."""
    import re
    norm = lambda t: re.sub(r"\W+", "", t.lower())  # noqa: E731
    text = pages.set_index(["doc_id", "page_num"]).text
    found = total = 0
    for it in items:
        for e in it.raw_evidence:
            total += 1
            key = (e["doc_id"], e["page_num"])
            found += key in text.index and norm(e["text"])[:60] in norm(text[key])
    return {"evidence_snippets": total, "found_on_stated_page": found, "rate": round(found / max(total, 1), 4)}


def eval_retrieval(items, r: Retriever, ks=(1, 5, 10)) -> dict:
    out = {}
    for name, cfg in RETRIEVAL_CONFIGS.items():
        rows = []
        t0 = time.perf_counter()
        for it in items:
            hits = _search(r, it, max(ks), cfg)
            got = [M.page_key(h.doc_id, h.page_num) for h in hits]
            gold = {M.page_key(e["doc_id"], e["page_num"]) for e in it.gold_evidence}
            rows.append({"question_type": it.tags.get("question_type"), "mrr": M.mrr(got, gold),
                         "doc_hit@5": float(any(h.doc_id == it.doc_id for h in hits[:5])),
                         **{f"hit@{k}": M.hit_at_k(got, gold, k) for k in ks}})
        df = pd.DataFrame(rows)
        cols = [f"hit@{k}" for k in ks] + ["mrr", "doc_hit@5"]
        out[name] = {"overall": df[cols].mean().round(3).to_dict(),
                     "by_question_type": df.groupby("question_type")[cols].mean().round(3).to_dict("index"),
                     "ms_per_query": round((time.perf_counter() - t0) / len(items) * 1000, 2), "n": len(items)}
    return out


def eval_answers(items, r: Retriever, config: str, k: int, out_path: Path) -> dict:
    import anthropic

    from docqa.answer import MODEL, answer
    client = anthropic.Anthropic()
    judge_model = os.environ.get("EVAL_JUDGE_MODEL", MODEL)
    rows = []
    with out_path.open("w") as fh:
        for i, it in enumerate(items, 1):
            hits = _search(r, it, k, RETRIEVAL_CONFIGS[config])
            gold = {M.page_key(e["doc_id"], e["page_num"]) for e in it.gold_evidence}
            a = answer(it.question, hits, client=client)
            g = M.judge(it.question, it.gold_answer, a.text, client, judge_model)
            row = {"id": it.id, "question": it.question, "gold_answer": it.gold_answer, "answer": a.text,
                   "question_type": it.tags.get("question_type"), "grade": g["grade"], "judge_reason": g["reason"],
                   "numeric_match": M.numeric_match(a.text, it.gold_answer),
                   "retrieval_hit": M.hit_at_k([M.page_key(h.doc_id, h.page_num) for h in hits], gold, k),
                   "citations": [M.page_key(c.doc_id, c.page_num) for c in a.citations],
                   "citation_page_precision": M.citation_page_precision(
                       [M.page_key(c.doc_id, c.page_num) for c in a.citations], gold),
                   "input_tokens": a.input_tokens, "output_tokens": a.output_tokens,
                   "cost_usd": a.cost_usd, "latency_s": a.latency_s, "model": a.model}
            fh.write(json.dumps(row) + "\n")
            rows.append(row)
            print(f"[{i}/{len(items)}] {g['grade']:9s} {it.id}")
    df = pd.DataFrame(rows)
    acc_by_hit = df.groupby("retrieval_hit").grade.apply(lambda s: (s == "correct").mean()).round(3).to_dict()
    return {
        "config": config, "k": k, "n": len(df), "model": df.model.iloc[0], "judge_model": judge_model,
        "accuracy": round((df.grade == "correct").mean(), 3),
        "incorrect_rate": round((df.grade == "incorrect").mean(), 3),
        "refusal_rate": round((df.grade == "refused").mean(), 3),
        "numeric_match": round(df.numeric_match.dropna().mean(), 3) if df.numeric_match.notna().any() else None,
        "citation_page_precision": round(df.citation_page_precision.dropna().mean(), 3)
        if df.citation_page_precision.notna().any() else None,
        "accuracy_by_question_type": df.groupby("question_type").grade.apply(lambda s: (s == "correct").mean()).round(3).to_dict(),
        "accuracy_when_retrieval_hit_vs_miss": {("hit" if k else "miss"): v for k, v in acc_by_hit.items()},
        "cost_usd_total": round(df.cost_usd.sum(), 4), "cost_usd_per_question": round(df.cost_usd.mean(), 5),
        "latency_s_p50": round(float(np.percentile(df.latency_s, 50)), 2),
        "latency_s_p95": round(float(np.percentile(df.latency_s, 95)), 2),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="financebench", choices=list(DATASETS))
    ap.add_argument("--answers", action="store_true")
    ap.add_argument("--config", default=DEFAULT_CONFIG, choices=list(RETRIEVAL_CONFIGS))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dense", action="store_true", help="also evaluate dense and hybrid retrieval")
    a = ap.parse_args(argv)

    items = DATASETS[a.dataset]()
    if a.limit:
        items = items[: a.limit]
    chunks = pd.read_parquet(storage.path("gold", a.dataset, "chunks.parquet"))
    enc = load_encoder() if a.dense else None
    if enc is not None:
        RETRIEVAL_CONFIGS.update(DENSE_CONFIGS)
    r = Retriever(chunks, encoder=enc)
    out = OUT / a.dataset
    out.mkdir(parents=True, exist_ok=True)

    ret = eval_retrieval(items, r)
    pages_path = storage.path("silver", a.dataset, "pages.parquet")
    if all(it.raw_evidence for it in items) and storage.exists(pages_path):
        ret["_data_check_evidence_alignment"] = evidence_alignment(items, pd.read_parquet(pages_path))
    (out / "retrieval_summary.json").write_text(json.dumps(ret, indent=2))
    print(pd.DataFrame({k: v["overall"] for k, v in ret.items() if not k.startswith("_")}).T.to_string())
    print("evidence alignment:", ret.get("_data_check_evidence_alignment"))

    if a.answers:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("--answers needs ANTHROPIC_API_KEY")
        s = eval_answers(items, r, a.config, a.k, out / f"answers_{a.config.strip('+')}.jsonl")
        (out / "answers_summary.json").write_text(json.dumps(s, indent=2))
        print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
