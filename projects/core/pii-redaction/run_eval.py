"""Score each redaction method on the 127 TAB test cases with the benchmark's own scorer.

    python run_eval.py                    # all methods (the trained model must exist)
    python run_eval.py --methods rules ner

Metrics (from TAB's evaluation.py):
  recall_direct_entities  share of directly identifying entities (names, case numbers) with
                          every mention masked; one missed mention means that person can be found
  recall_quasi_entities   same for quasi-identifiers (dates, places, jobs, ...)
  token_recall            share of tokens that should be masked and are
  token_precision         share of masked tokens that needed masking (over-masking hurts this)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import redact as R
from data import RAW, download, load

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
KEEP = ["recall_direct_entities", "recall_quasi_entities", "recall_all_entities", "token_recall",
        "token_precision", "token_f1", "mention_recall", "token_recall_by_type"]


def spans_for(method: str, docs: list[dict]) -> tuple[dict, float]:
    t0 = time.perf_counter()
    if method == "rules":
        fns = [R.rules]
    elif method == "ner":
        fns = [R.Ner()]
    elif method == "trained":
        fns = [R.trained()]
    elif method == "trained+rules":
        fns = [R.trained(), R.rules]
    elif method == "trained+ner":      # highest recall: mask anything either model flags
        fns = [R.trained(), R.Ner()]
    else:
        raise ValueError(method)
    out = {d["doc_id"]: [[s, e] for s, e, _ in R.merge([x for f in fns for x in f(d["text"])])] for d in docs}
    return out, time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["rules", "ner", "trained", "trained+rules", "trained+ner"])
    a = ap.parse_args(argv)
    download()
    sys.path.insert(0, str(RAW))
    import evaluation as tab  # the benchmark's official scorer

    docs = load("test")
    with contextlib.redirect_stdout(io.StringIO()):
        gold = tab.GoldCorpus(str(RAW / "echr_test.json"))
    OUT.mkdir(exist_ok=True)
    results = {}
    for m in a.methods:
        spans, secs = spans_for(m, docs)
        (OUT / f"masked_{m.replace('+', '_')}.json").write_text(json.dumps(spans))
        masked = [tab.MaskedDocument(k, [tuple(x) for x in v]) for k, v in spans.items()]
        with contextlib.redirect_stdout(io.StringIO()):
            scores = tab.evaluate(gold, masked)
        results[m] = {k: scores[k] for k in KEEP} | {
            "seconds": round(secs, 1), "ms_per_case": round(secs / len(docs) * 1000),
            "masked_spans": sum(len(v) for v in spans.values())}
        print(m, {k: results[m][k] for k in KEEP[:6]}, flush=True)
    path = OUT / "test_scores.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(old | results, indent=2))


if __name__ == "__main__":
    main()
