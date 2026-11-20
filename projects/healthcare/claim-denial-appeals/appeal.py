"""Draft an appeal letter for a prior authorization denied as "not medically necessary".

The letter is built from the prior authorization checklist (../prior-auth-assistant): the Medicare
coverage policy (LCD), its criteria, and the note quotes that show each criterion is met.

  grounded   the model sees only the denial, the policy and the verified (criterion, quote) pairs
  free       the model sees the denial, the policy criteria and the whole note
Each letter is scored without a reference letter:
  quotes          text in double quotes must appear word for word in the note or the policy
  numbers         numbers in the letter that appear in neither the note nor the policy
  evidence used   share of the verified note quotes the letter actually cites
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prior-auth-assistant"))
from criteria import norm, verified  # noqa: E402

from docqa import llm  # noqa: E402

SYSTEM = (
    "You write appeal letters for a physician's office to a health plan. Be factual and brief (under 300 words). "
    "Use only the facts given. Put every piece of evidence from the medical record in double quotes, copied "
    "exactly. Do not invent dates, test results, names or numbers. Use [Patient name], [Member ID] and "
    "[Physician name] as placeholders.")


def grounded_prompt(service: str, lcd: str, title: str, evidence: list[dict]) -> str:
    ev = "\n".join(f"- Criterion: {e['criterion']}\n  Record: \"{e['quote']}\"" for e in evidence)
    return (f"The health plan denied prior authorization for {service} as not medically necessary.\n"
            f"Coverage policy: Medicare LCD {lcd}, {title}.\n\nCriteria met, with evidence from the record:\n{ev}\n\n"
            "Write the appeal letter.")


def free_prompt(service: str, lcd: str, title: str, criteria: list[str], note: str) -> str:
    cr = "\n".join(f"- {c}" for c in criteria)
    return (f"The health plan denied prior authorization for {service} as not medically necessary.\n"
            f"Coverage policy: Medicare LCD {lcd}, {title}. Its criteria include:\n{cr}\n\n"
            f"Medical record:\n{note}\n\nWrite the appeal letter.")


QUOTE = re.compile(r"[\"“]([^\"”]{12,}?)[\"”]")
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")


def score(letter: str, note: str, policy: str, evidence: list[dict]) -> dict:
    quotes = QUOTE.findall(letter)
    ok = [verified(q, note) or verified(q, policy) for q in quotes]
    src = set(NUM.findall(note + " " + policy))
    nums = [n for n in NUM.findall(re.sub(r"\bL\d{5}\b", "", letter)) if n not in src]
    nl = norm(letter)
    used = [norm(e["quote"]) in nl for e in evidence]
    return {"quotes": len(quotes), "quotes_verified": int(sum(ok)),
            "unverified_quotes": [q for q, k in zip(quotes, ok) if not k][:5],
            "unsupported_numbers": nums[:10], "evidence_used": round(sum(used) / len(used), 3) if used else None,
            "words": len(letter.split())}


def draft(prompt: str, model: str):
    return llm.chat(SYSTEM, prompt, model=model, max_tokens=600)
