"""Cost, latency and quality grid for the filing Q&A system, run in CI with free local models, sharded.

    python run_eval.py --shard 0/20      # every config for one shard of questions
    python run_eval.py --merge           # results/answers.jsonl, results/summary.json, results/*.png

Quality is numeric accuracy on the 117 of 150 FinanceBench questions whose answer is a number
(the free 3B judge was shown to be unreliable in other projects here, so no LLM grading).
Cost is measured as tokens and CPU seconds per question on a 4 vCPU GitHub runner.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import serving as sv

from docpipe import storage
from docqa.answer import answer, numbered_sources
from docqa.retrieve import Retriever
from evalkit import metrics as M
from evalkit.dataset import from_financebench
from evalkit.run import DEFAULT_CONFIG, RETRIEVAL_CONFIGS

OUT = Path(__file__).resolve().parent / "results"
MODELS = os.environ.get("EVAL_MODELS", "qwen2.5:1.5b,qwen2.5:3b,llama3.2:3b").split(",")
KS = [3, 5, 10]
SMALL, LARGE = "qwen2.5:1.5b", "qwen2.5:3b"


def run(shard: str) -> None:
    i, n = map(int, shard.split("/"))
    items = from_financebench()[i::n]
    chunks = pd.read_parquet(storage.path("gold", "financebench", "chunks.parquet"))
    r = Retriever(chunks)
    cfg = dict(RETRIEVAL_CONFIGS[DEFAULT_CONFIG])
    OUT.mkdir(exist_ok=True)
    for model in MODELS:                  # load every model once, so no timed call pays the load time
        os.environ["LLM_MODEL"] = model
        answer("warm-up", hits_for_warmup(r, cfg), provider="ollama")
    with (OUT / f"answers_shard{i}.jsonl").open("w") as fh:
        for j, it in enumerate(items):
            hits_all, _ = r.search(it.question, k=max(KS), **cfg)
            gold = {M.page_key(e["doc_id"], e["page_num"]) for e in it.gold_evidence}
            for k in KS:
                for comp in (False, True):
                    hits = sv.compress(it.question, hits_all[:k]) if comp else hits_all[:k]
                    ctx = numbered_sources(hits)
                    for model in MODELS:
                        os.environ["LLM_MODEL"] = model
                        row = {"id": it.id, "question_type": it.tags.get("question_type"), "model": model, "k": k,
                               "compress": comp, "gold": it.gold_answer,
                               "retrieval_hit": M.hit_at_k([M.page_key(h.doc_id, h.page_num) for h in hits], gold, k),
                               "context_words": len(ctx.split())}
                        try:
                            a = answer(it.question, hits, provider="ollama")
                            row |= {"answer": a.text[:1500], "refused": a.refused,
                                    "citations": [M.page_key(c.doc_id, c.page_num) for c in a.citations],
                                    "input_tokens": a.input_tokens, "output_tokens": a.output_tokens,
                                    "latency_s": round(a.latency_s, 2),
                                    "numeric_match": M.numeric_match(a.text, it.gold_answer),
                                    "unsupported_numbers": sv.unsupported_numbers(a.text, ctx)}
                        except Exception as e:  # noqa: BLE001 - one failed call skips one answer
                            row["error"] = type(e).__name__
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
            print(j, len(items), flush=True)


def hits_for_warmup(r: Retriever, cfg: dict):
    return r.search("total revenue", k=1, **cfg)[0]


def _stats(g: pd.DataFrame) -> dict:
    num = g.numeric_match.dropna()
    return {"answers": len(g), "numeric_accuracy": round(float(num.mean()), 3), "numeric_questions": len(num),
            "refusal_rate": round(float(g.refused.mean()), 3),
            "answers_with_unsupported_number": round(float((g.unsupported_numbers > 0).mean()), 3),
            "retrieval_hit": round(float(g.retrieval_hit.mean()), 3),
            "input_tokens_mean": round(float(g.input_tokens.mean()), 1),
            "output_tokens_mean": round(float(g.output_tokens.mean()), 1),
            "latency_p50_s": round(float(g.latency_s.median()), 2),
            "latency_p95_s": round(float(g.latency_s.quantile(0.95)), 2),
            "cpu_seconds_mean": round(float(g.latency_s.mean()), 2)}


def router(df: pd.DataFrame) -> pd.DataFrame:
    """Small model first; escalate to the large one on refusal, no citation or an unsupported number.
    Cost is the small model's call plus, when escalated, the large one's."""
    rows = []
    for (k, comp), g in df[df.model.isin([SMALL, LARGE])].groupby(["k", "compress"]):
        big = g[g.model == LARGE].set_index("id")
        for _, s in g[g.model == SMALL].iterrows():
            if s.id not in big.index:
                continue
            esc = sv.needs_escalation(s)
            b = big.loc[s.id]
            pick = b if esc else s
            rows.append({"id": s.id, "model": f"router {SMALL} -> {LARGE}", "k": k, "compress": comp,
                         "escalated": esc, "numeric_match": pick.numeric_match, "refused": pick.refused,
                         "unsupported_numbers": pick.unsupported_numbers, "retrieval_hit": s.retrieval_hit,
                         "input_tokens": s.input_tokens + (b.input_tokens if esc else 0),
                         "output_tokens": s.output_tokens + (b.output_tokens if esc else 0),
                         "latency_s": s.latency_s + (b.latency_s if esc else 0)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    df = df[df.get("error", pd.Series(index=df.index, dtype=object)).isna()].copy()
    rt = router(df)
    both = pd.concat([df, rt], ignore_index=True)
    configs = []
    for (model, k, comp), g in both.groupby(["model", "k", "compress"]):
        c = {"model": model, "k": int(k), "compress": bool(comp)} | _stats(g)
        if "escalated" in g and g.escalated.notna().any():
            c["escalation_rate"] = round(float(g.escalated.mean()), 3)
        configs.append(c)
    t = pd.DataFrame(configs)
    best = t.sort_values(["numeric_accuracy", "cpu_seconds_mean"], ascending=[False, True]).iloc[0]
    out = {"questions": int(df.id.nunique()), "configs": configs,
           "most_accurate": best[["model", "k", "compress", "numeric_accuracy", "cpu_seconds_mean"]].to_dict(),
           "pareto": pareto(t)}
    return json.loads(json.dumps(out, default=float))


def pareto(t: pd.DataFrame) -> list[dict]:
    """Configs no other config beats on both accuracy and CPU seconds."""
    keep = []
    for _, r in t.iterrows():
        dominated = ((t.numeric_accuracy >= r.numeric_accuracy) & (t.cpu_seconds_mean <= r.cpu_seconds_mean) &
                     ((t.numeric_accuracy > r.numeric_accuracy) | (t.cpu_seconds_mean < r.cpu_seconds_mean))).any()
        if not dominated:
            keep.append({k: r[k] for k in ("model", "k", "compress", "numeric_accuracy", "cpu_seconds_mean")})
    return sorted(keep, key=lambda x: x["cpu_seconds_mean"])


def plot(summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = pd.DataFrame(summary["configs"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, g in t.groupby("model"):
        ax.scatter(g.cpu_seconds_mean, g.numeric_accuracy * 100, label=model, s=40)
        for _, r in g.iterrows():
            ax.annotate(f"k{r.k}{'c' if r.compress else ''}", (r.cpu_seconds_mean, r.numeric_accuracy * 100),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    p = pd.DataFrame(summary["pareto"])
    ax.plot(p.cpu_seconds_mean, p.numeric_accuracy * 100, color="grey", lw=1, ls="--", label="best trade-off")
    ax.set_xlabel("CPU seconds per question (4 vCPU runner)")
    ax.set_ylabel("Numeric accuracy (%)")
    ax.set_title("FinanceBench Q&A: quality vs cost (k = chunks, c = compressed)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "cost_vs_accuracy.png", dpi=150)


def merge() -> dict:
    files = sorted(OUT.glob("answers_shard*.jsonl"))
    df = pd.concat([pd.read_json(f, lines=True) for f in files if f.stat().st_size], ignore_index=True)
    df.to_json(OUT / "answers.jsonl", orient="records", lines=True)
    for f in files:
        f.unlink()
    s = summarize(df)
    (OUT / "summary.json").write_text(json.dumps(s, indent=2))
    plot(s)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(argv)
    if a.merge:
        print(json.dumps(merge(), indent=2)[:3000])
    else:
        run(a.shard)


if __name__ == "__main__":
    main()
