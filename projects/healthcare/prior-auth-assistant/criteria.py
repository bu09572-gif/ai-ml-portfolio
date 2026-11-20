"""Turn an LCD (a Medicare coverage policy) into a checklist, then check a clinical note against it.

  passages      split the LCD's coverage text into ~350-word pieces a small model can read
  extract       ask the model for each coverage criterion as an exact quote from the passage;
                a quote that is not in the LCD text is dropped, so every criterion is traceable
  check         for one note and one criterion: met, not met, or not documented, with a quote
                from the note; a quote not found in the note turns the answer into "not documented"
  decision      ready to submit only if every criterion checked is met with a verified quote
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from docqa import llm
from docqa.retrieve import BM25Index

SYSTEM_EXTRACT = (
    "You read Medicare coverage policies. From the policy text, list the conditions a patient's record must "
    "show for the service to be covered. Copy each condition word for word from the text, one sentence or "
    'list item each. Reply as JSON: {"criteria": ["...", "..."]}. If the text has no such conditions, '
    'reply {"criteria": []}.')
SYSTEM_CHECK = (
    "You review a clinical note for a prior authorization request. Decide whether the note documents the "
    "coverage criterion. Reply as JSON: "
    '{"status": "met" | "not_met" | "not_documented", "quote": "<exact words from the note, or empty>"}. '
    "Use met only if the note clearly shows the criterion; not_met if the note shows it is not true; "
    "not_documented if the note does not say. The quote must be copied exactly from the note.")
SYSTEM_CHECK_NUMBERED = (
    "You review a clinical note for a prior authorization request. Below are numbered sentences from the "
    "note. Decide whether they document the coverage criterion. Reply as JSON: "
    '{"status": "met" | "not_met" | "not_documented", "evidence": [sentence numbers]}. '
    "Use met only if the sentences clearly show the criterion; not_met if they show it is not true; "
    "not_documented if they do not say. For met or not_met, list the numbers of the sentences that show it.")
STATUSES = ("met", "not_met", "not_documented")


def norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(s).lower()))


def verified(quote: str, source: str, min_words: int = 3) -> bool:
    q = norm(quote)
    return len(q.split()) >= min_words and q in norm(source)


def passages(text: str, words: int = 350) -> list[str]:
    out, cur, n = [], [], 0
    for line in str(text).split("\n"):
        k = len(line.split())
        if cur and n + k > words:
            out.append("\n".join(cur))
            cur, n = [], 0
        cur.append(line)
        n += k
    if cur and n:
        out.append("\n".join(cur))
    return out


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def extract(lcd_text: str, model: str) -> list[dict]:
    """Criteria quoted from the LCD, each marked verified or not."""
    out, seen = [], set()
    for i, p in enumerate(passages(lcd_text)):
        c = llm.chat(SYSTEM_EXTRACT, p, model=model, max_tokens=800, json_mode=True)
        items = _json(c.text).get("criteria", [])
        for q in items if isinstance(items, list) else []:
            q = str(q).strip()
            if norm(q) in seen or not q:
                continue
            seen.add(norm(q))
            out.append({"passage": i, "criterion": q, "verified": verified(q, lcd_text, min_words=4)})
    return out


def relevant(criteria: list[str], note: str, k: int = 8) -> list[str]:
    """The k criteria that share the most words with the note (an LCD covers many conditions)."""
    if len(criteria) <= k:
        return criteria
    ix = BM25Index(pd.DataFrame({"header": "", "text": criteria}))
    return [criteria[j] for j in np.argsort(-ix.scores(note[:6000]), kind="stable")[:k]]


def check(note: str, criterion: str, model: str) -> dict:
    c = llm.chat(SYSTEM_CHECK, f"Criterion: {criterion}\n\nClinical note:\n{note}", model=model,
                 max_tokens=200, json_mode=True)
    d = _json(c.text)
    status = d.get("status") if d.get("status") in STATUSES else None
    quote = str(d.get("quote") or "").strip()
    ok = verified(quote, note)
    final = status if status in ("met", "not_met") and ok else ("not_documented" if status else None)
    return {"criterion": criterion, "model_status": status, "quote": quote, "quote_verified": ok,
            "status": final, "latency_s": round(c.latency_s, 1)}


def sentences(note: str) -> list[str]:
    parts = re.split(r"(?<=[.!?:])\s+(?=[A-Z0-9])|\n+", str(note))
    return [p.strip() for p in parts if len(p.split()) >= 3]


def check_numbered(note: str, criterion: str, model: str, k: int = 12) -> dict:
    """The model sees the k note sentences closest to the criterion, numbered, and cites numbers.
    The evidence is the cited sentences themselves, so it is word for word from the note; a met or
    not_met with no valid number turns into not_documented."""
    sents = sentences(note)
    if sents:
        ix = BM25Index(pd.DataFrame({"header": "", "text": sents}))
        top = sorted(np.argsort(-ix.scores(criterion), kind="stable")[:k].tolist())
    else:
        top = []
    shown = [sents[j] for j in top]
    body = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(shown))
    c = llm.chat(SYSTEM_CHECK_NUMBERED, f"Criterion: {criterion}\n\nNote sentences:\n{body}", model=model,
                 max_tokens=120, json_mode=True)
    d = _json(c.text)
    status = d.get("status") if d.get("status") in STATUSES else None
    ev = d.get("evidence") if isinstance(d.get("evidence"), list) else []
    nums = sorted({int(x) for x in ev if str(x).strip().isdigit() and 1 <= int(x) <= len(shown)})
    quote = " ".join(shown[i - 1] for i in nums)
    ok = bool(nums)
    final = status if status in ("met", "not_met") and ok else ("not_documented" if status else None)
    return {"criterion": criterion, "model_status": status, "evidence": nums, "quote": quote,
            "quote_verified": ok, "status": final, "latency_s": round(c.latency_s, 1)}


def decision(checks: list[dict]) -> str:
    st = [c["status"] for c in checks]
    if not st or None in st:
        return "error"
    if all(s == "met" for s in st):
        return "ready to submit"
    return "criterion not met" if "not_met" in st else "missing documentation"
