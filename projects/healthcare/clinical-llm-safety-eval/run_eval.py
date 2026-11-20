"""Run the clinical checks for each model on MedQA and PubMedQA test sets.

    python run_eval.py --shard 0/20     # one CI shard (both models, both datasets)
    python run_eval.py --merge          # combine shards, write the summary
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import clinical_eval as ce
import data
import pandas as pd

OUT = Path(__file__).resolve().parent / "results"
MODELS = os.environ.get("EVAL_MODELS", "qwen2.5:3b,llama3.2:3b").split(",")


def run(shard: str | None, limit: int | None) -> None:
    mq, pq = data.medqa(), data.pubmedqa()
    i, n = map(int, shard.split("/")) if shard else (0, 1)
    mq, pq = mq.iloc[i::n], pq.iloc[i::n]
    if limit:
        mq, pq = mq.head(limit), pq.head(limit)
    OUT.mkdir(exist_ok=True)
    with (OUT / f"rows_shard{i}.jsonl").open("w") as fh:
        for model in MODELS:
            for k, row in enumerate(mq.itertuples()):
                try:
                    r = ce.medqa_row(row, model, int(row.id.split("-")[1])) | {"dataset": "medqa"}
                except Exception as e:  # noqa: BLE001 - one failed call skips one question
                    r = {"id": row.id, "model": model, "dataset": "medqa", "error": type(e).__name__}
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                if k % 20 == 0:
                    print(model, "medqa", k, len(mq), flush=True)
            for row in pq.itertuples():
                try:
                    r = ce.pubmedqa_row(row, model) | {"dataset": "pubmedqa"}
                except Exception as e:  # noqa: BLE001
                    r = {"id": row.id, "model": model, "dataset": "pubmedqa", "error": type(e).__name__}
                fh.write(json.dumps(r) + "\n")
                fh.flush()


def summarize(df: pd.DataFrame) -> dict:
    out = {}
    for model, d in df.groupby("model"):
        m = d[d.dataset == "medqa"]
        p = d[d.dataset == "pubmedqa"]
        agree = m[m.order_consistent == True]  # noqa: E712
        sx = m[m.sex_neutral == True]  # noqa: E712
        out[model] = {
            "medqa": {"questions": len(m), "accuracy": round(m.correct.mean(), 4),
                      "accuracy_shuffled_options": round(m.correct_shuffled.mean(), 4),
                      "unparseable_answers": int(m.answer.isna().sum()),
                      "same_answer_after_shuffling_options": round(m.order_consistent.mean(), 4),
                      "abstain_when_shuffle_disagrees": {"coverage": round(len(agree) / len(m), 4),
                                                         "accuracy_on_answered": round(agree.correct.mean(), 4)}}
            | ({"by_step": m.groupby("meta_info").correct.mean().round(4).to_dict()} if "meta_info" in m else {}),
            "patient_sex_swap": {"questions": len(sx),
                                 "same_answer_after_sex_swap": round(sx.sex_consistent.mean(), 4),
                                 "same_answer_after_shuffle_same_questions": round(sx.order_consistent.mean(), 4),
                                 "accuracy_original": round(sx.correct.mean(), 4),
                                 "accuracy_sex_swapped": round(sx.correct_sex_swapped.mean(), 4)},
            "pubmedqa": {"questions": len(p), "accuracy": round(p.correct.mean(), 4),
                         "unparseable_answers": int(p.answer.isna().sum()),
                         "answer_counts": p.answer.fillna("none").value_counts().to_dict(),
                         "majority_class_baseline": round((p.gold == p.gold.mode()[0]).mean(), 4)},
            "errors": int(d.get("error", pd.Series(dtype=object)).notna().sum())}
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
    ap.add_argument("--limit", type=int)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(argv)
    if a.merge:
        print(json.dumps(merge(), indent=2))
    else:
        run(a.shard, a.limit)


if __name__ == "__main__":
    main()
