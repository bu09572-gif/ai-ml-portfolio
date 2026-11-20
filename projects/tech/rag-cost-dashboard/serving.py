"""Serving choices for the filing Q&A system that change cost, speed and quality.

  k           how many retrieved chunks go into the prompt (3, 5, 10)
  compress    keep only the sentences of each chunk that share the most words with the question,
              about 40% of the words, in their original order. Fewer prompt tokens; may drop the fact
  model       Qwen2.5 1.5B, Qwen2.5 3B, Llama 3.2 3B, all free and local (Ollama, CPU)
  router      ask the small model first; send the question to the larger model only if the small
              model refuses, cites nothing, or uses a number that is not in its sources.
              Scored after the fact from the two models' real answers (both run at temperature 0)
"""
from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pandas as pd

from docqa.retrieve import BM25Index, Hit
from evalkit.metrics import numbers

SENT = re.compile(r"(?<=[.;:!?])\s+")


def compress(question: str, hits: list[Hit], keep: float = 0.4) -> list[Hit]:
    out = []
    for h in hits:
        sents = [s for s in SENT.split(h.text) if s.strip()]
        if len(sents) <= 2:
            out.append(h)
            continue
        ix = BM25Index(pd.DataFrame({"header": "", "text": sents}))
        order = np.argsort(-ix.scores(question), kind="stable")
        budget, used, chosen = keep * len(h.text.split()), 0, set()
        for j in order:
            if used >= budget:
                break
            chosen.add(int(j))
            used += len(sents[j].split())
        out.append(replace(h, text=" ".join(sents[j] for j in sorted(chosen))))
    return out


def unsupported_numbers(answer: str, context: str, rel_tol: float = 0.005) -> int:
    """Numbers in the answer (above 10, so not list markers) that match no number in the sources,
    allowing a change of scale (millions vs billions, ratio vs percent)."""
    ctx = [abs(x) for x in numbers(context)]
    bad = 0
    for a in numbers(answer):
        a = abs(a)
        if a < 10:
            continue
        if not any(abs(a - c * s) <= rel_tol * a for c in ctx for s in (1, 1e3, 1e-3, 1e6, 1e-6, 100, 0.01)):
            bad += 1
    return bad


def needs_escalation(row: dict) -> bool:
    return bool(row["refused"] or not row["citations"] or row["unsupported_numbers"] > 0)
