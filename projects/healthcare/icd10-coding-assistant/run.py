"""Fit on train, tune ensemble weights on dev, refit on train + dev, report on the 250 test cases.

    python run.py      # -> results/metrics.json, results/examples.json
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import coder
import data

OUT = Path(__file__).resolve().parent / "results"
METHODS = ["frequency", "classifier", "neighbours", "dictionary"]


def main():
    df = data.cases()
    desc = data.code_descriptions()
    tr, dv, te = (df[df.split == s] for s in ["train", "dev", "test"])
    OUT.mkdir(exist_ok=True)

    m = coder.Coder(desc).fit(tr.text.tolist(), tr.codes.tolist())
    s_dev = m.scores(dv.text.tolist())
    grid = [0, 0.25, 0.5, 1]
    best = max((dict(zip(METHODS, w)) for w in itertools.product([0, 0.1], grid, grid, grid) if any(w)),
               key=lambda w: coder.evaluate(m.rank(m.ensemble(s_dev, w)), dv.codes.tolist())["MAP"])
    print("weights", best, flush=True)

    trdv = df[df.split != "test"]
    m = coder.Coder(desc).fit(trdv.text.tolist(), trdv.codes.tolist())
    s = m.scores(te.text.tolist())
    res = {"data": {"train_and_dev_cases": len(trdv), "test_cases": len(te),
                    "test_codes_per_case": round(te.codes.map(len).mean(), 2),
                    "test_codes_seen_in_training": round(float(te.codes.explode().isin(m.freq).mean()), 3),
                    "candidate_codes": len(m.vocab)},
           "ensemble_weights": best, "test": {}}
    for k in METHODS:
        res["test"][k] = coder.evaluate(m.rank(s[k]), te.codes.tolist())
    ranked = m.rank(m.ensemble(s, best))
    res["test"]["ensemble"] = coder.evaluate(ranked, te.codes.tolist())
    print(json.dumps(res, indent=2))
    examples = []
    for i in range(3):
        gold = set(te.codes.iloc[i])
        examples.append({"doc": te.doc.iloc[i], "gold": sorted(gold), "suggested": [
            {"code": c, "description": m._describe(c), "correct": c in gold,
             "evidence": m.evidence[i].get(c, "")[:200]} for c in ranked[i][:8]]})
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2))
    (OUT / "examples.json").write_text(json.dumps(examples, indent=2))


if __name__ == "__main__":
    main()
