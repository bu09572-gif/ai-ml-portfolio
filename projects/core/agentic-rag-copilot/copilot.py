"""An agent that answers questions about SEC filings by deciding its own next step.

Each turn the model picks one action and gets the result back:

  search(query)         top 5 pages for a query (BM25, company/year filters), as short previews
  read(source)          the full text of one page from the last search
  calculate(expression) + - * / arithmetic, done in Python
  answer(text)          the final answer, citing pages as [P#]

Guardrails, applied in code rather than left to the model:
  - at most MAX_STEPS actions, then it must answer or say it can't
  - every [P#] cited must be a page the agent actually read
  - every number in the answer must appear on a page it read or come out of calculate();
    if not, the answer goes back once with the list of unsupported numbers, then is refused
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import pandas as pd

from docqa import llm
from docqa.answer import REFUSAL_TEXT, Answer, Citation
from docqa.retrieve import Retriever

MAX_STEPS = 8
PREVIEW_CHARS = 300
PAGE_CHARS = 3500
PROMPT_CHARS = 18000      # stay well inside an 8k-token context window
KEEP_FULL = 2             # the last 2 page reads stay in full; older ones are shortened

SYSTEM = f"""You answer questions about company SEC filings. You work in steps. Each step, reply with
one JSON object and nothing else:
  {{"action": "search", "query": "<what to look for>"}}
  {{"action": "read", "source": "P3"}}
  {{"action": "calculate", "expression": "(1577 / 32765) * 100"}}
  {{"action": "answer", "text": "<final answer, citing pages like [P3]>"}}
How to work:
- Search first. Searches return page previews labeled P1, P2, ... Read the pages that look right.
- Only use numbers you saw on a page you read, or results of calculate.
- Use calculate for any arithmetic. Do not do math in your head.
- Cite every page you used, like [P3].
- If the filings don't have the answer, answer with: "{REFUSAL_TEXT}"
You have at most {MAX_STEPS} steps."""

NUM = re.compile(r"(?<![\w.])\(?-?\$?\d[\d,]*\.?\d*\)?%?")
CITE = re.compile(r"\[P(\d+)\]")
YEARS = re.compile(r"^(19|20)\d\d$")
OPS = re.compile(r"^[\d\s+\-*/().,]+$")


def numbers_in(text: str) -> set[float]:
    out = set()
    for m in NUM.findall(text):
        s = m.strip("()$%").replace(",", "").replace("$", "").rstrip(".")
        try:
            out.add(round(abs(float(s)), 4))
        except ValueError:
            pass
    return out


def unsupported_numbers(answer: str, evidence: set[float]) -> list[str]:
    """Numbers in the answer that don't match evidence, allowing rounding and scale changes."""
    bad = []
    for raw in NUM.findall(CITE.sub("", answer)):
        s = raw.strip("()$%").replace(",", "").rstrip(".")
        if not s or (YEARS.match(s) and raw.rstrip(".") == s):   # a bare 2019 is a year; $2,019 is not
            continue
        try:
            v = abs(float(s))
        except ValueError:
            continue
        if v < 10 and float(v).is_integer():    # small counts and list markers ("3 segments", "1.")
            continue
        # allow for rounding the printed value, and for a change of scale (millions -> billions, ratio -> %)
        tol = 0.5 * 10 ** -_decimals(s) + 0.005 * v
        ok = any(abs(v - e * f) <= tol for e in evidence for f in (1, 1e-3, 1e-6, 1e-9, 1e3, 100))
        if not ok:
            bad.append(raw)
    return bad


def _decimals(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def safe_calc(expr: str) -> float:
    expr = expr.replace(",", "")
    if not OPS.match(expr) or "**" in expr:
        raise ValueError("only numbers and + - * / ( ) are allowed")
    return float(eval(compile(expr, "<calc>", "eval"), {"__builtins__": {}}, {}))  # noqa: S307 (checked above)


def _fit(transcript: list[str]) -> str:
    """Keep the prompt short: shorten all but the last KEEP_FULL page reads, then drop the
    oldest middle turns if it is still too long. The question (first entry) always stays."""
    reads = [i for i, t in enumerate(transcript) if "\nResult: [P" in t and " page " in t[:400]]
    old = set(reads[:-KEEP_FULL]) if len(reads) > KEEP_FULL else set()
    parts = [t[:700] + " ...(shortened; read it again if you need it)" if i in old else t
             for i, t in enumerate(transcript)]
    while len("\n\n".join(parts)) > PROMPT_CHARS and len(parts) > 2:
        parts.pop(1)
    return "\n\n".join(parts)


@dataclass
class Step:
    action: str
    arg: str
    observation: str


@dataclass
class Run:
    answer: Answer
    steps: list[Step] = field(default_factory=list)
    pages_read: list[tuple[str, int]] = field(default_factory=list)
    revised: bool = False
    auto_cited: bool = False      # the agent forgot citations and code added the pages it read
    guard: str = ""               # why the answer was refused by a guardrail, if it was


class Copilot:
    def __init__(self, retriever: Retriever, pages: pd.DataFrame):
        self.r = retriever
        self.pages = pages.set_index(["doc_id", "page_num"]).text

    def _llm(self, messages: str, run: Run) -> dict:
        try:
            c = llm.chat(SYSTEM, messages, json_mode=True, max_tokens=500)
        except Exception as e:  # noqa: BLE001 - a failed model call ends this question, not the whole run
            run.guard = f"model error: {type(e).__name__}"
            return {"action": "_error"}
        a = run.answer
        a.input_tokens += c.input_tokens
        a.output_tokens += c.output_tokens
        a.model, a.provider = c.model, c.provider
        m = re.search(r"\{.*\}", c.text, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}

    def ask(self, question: str) -> Run:
        t0 = time.perf_counter()
        run = Run(Answer(""))
        labels: dict[str, tuple[str, int]] = {}     # P# -> page, stable across searches
        page_numbers: dict[str, set[float]] = {}    # P# -> numbers on that page, for pages read
        evidence: set[float] = set()
        transcript = [f"Question: {question}"]
        final = None
        for step in range(MAX_STEPS):
            left = MAX_STEPS - step
            if left == 1:
                nudge = 'This is your last step. Reply with {"action": "answer", ...} now.'
            elif len(run.pages_read) >= 3:
                nudge = f"{left} steps left. You have read {len(run.pages_read)} pages; answer as soon as you can."
            else:
                nudge = f"{left} steps left. Reply with JSON."
            prompt = _fit(transcript) + f"\n\n({nudge})"
            d = self._llm(prompt, run)
            act = str(d.get("action", "")).lower()
            if act == "_error":
                break
            if act == "search":
                q = str(d.get("query") or question)
                # keep the question's company and year in the query so the filters still apply
                hits, _ = self.r.search(f"{question} {q}", k=5)
                lines = []
                for h in hits:
                    key = (h.doc_id, h.page_num)
                    label = next((k for k, v in labels.items() if v == key), f"P{len(labels) + 1}")
                    labels[label] = key
                    lines.append(f"[{label}] {h.header}: {h.text[:PREVIEW_CHARS]}...")
                obs = "\n".join(lines) or "no results"
                run.steps.append(Step("search", q, obs))
            elif act == "read":
                label = str(d.get("source", "")).strip("[]").upper()
                if label in labels:
                    key = labels[label]
                    text = str(self.pages.get(key, ""))[:PAGE_CHARS]
                    if key not in run.pages_read:
                        run.pages_read.append(key)
                    evidence |= numbers_in(text)
                    page_numbers[label] = numbers_in(text)
                    obs = f"[{label}] page {key[1] + 1} of {key[0]}:\n{text}"
                else:
                    obs = f"There is no source {label!r}. Search first, then read a P# from the results."
                run.steps.append(Step("read", label, obs[:120]))
            elif act == "calculate":
                expr = str(d.get("expression", ""))
                try:
                    val = safe_calc(expr)
                    evidence.add(round(abs(val), 4))
                    obs = f"{expr} = {val:.6g}"
                except Exception as e:  # noqa: BLE001 - any bad expression is reported back to the model
                    obs = f"error: {e}"
                run.steps.append(Step("calculate", expr, obs))
            elif act == "answer":
                final = str(d.get("text", "")).strip()
                if final and not CITE.search(final) and REFUSAL_TEXT.lower().rstrip(".") not in final.lower():
                    # small models often forget the [P#] tags: cite the pages read that hold the answer's numbers
                    nums = numbers_in(final)
                    used = [lb for lb, ns in page_numbers.items() if nums & ns]
                    if used:
                        final, run.auto_cited = final + " " + "".join(f"[{lb}]" for lb in used), True
                problem = self._check(final, labels, run, evidence)
                if problem and not run.revised and left > 1:
                    run.revised = True
                    obs = f"Your answer was not accepted: {problem}. Fix it (read or calculate if needed), then answer again."
                    run.steps.append(Step("answer", final[:200], obs))
                    final = None
                else:
                    if problem:
                        run.guard, final = problem, REFUSAL_TEXT
                    run.steps.append(Step("answer", final[:200], "accepted"))
                    break
            else:
                obs = 'Reply with one JSON object with an "action" of search, read, calculate or answer.'
                run.steps.append(Step("invalid", json.dumps(d)[:120], obs))
            transcript.append(f"You: {json.dumps(d)[:600]}\nResult: {obs}")
        if final is None and not run.guard:
            # out of steps: one last call where only an answer is accepted
            d = self._llm(_fit(transcript) + '\n\n(No steps left. Reply only with {"action": "answer", "text": ...}.)', run)
            if str(d.get("action", "")).lower() == "answer" and str(d.get("text", "")).strip():
                final = str(d["text"]).strip()
                if not CITE.search(final):
                    used = [lb for lb, ns in page_numbers.items() if numbers_in(final) & ns]
                    if used:
                        final, run.auto_cited = final + " " + "".join(f"[{lb}]" for lb in used), True
                problem = self._check(final, labels, run, evidence)
                run.steps.append(Step("answer", final[:200], problem or "accepted"))
                if problem:
                    run.guard, final = problem, REFUSAL_TEXT
        if final is None:
            run.guard, final = run.guard or "ran out of steps", REFUSAL_TEXT
        a = run.answer
        a.text = final
        cited = [labels[f"P{n}"] for n in dict.fromkeys(CITE.findall(final)) if f"P{n}" in labels]
        a.citations = [Citation(f"P{i}", d, p) for i, (d, p) in enumerate(cited, 1)]
        a.latency_s = time.perf_counter() - t0
        return run

    @staticmethod
    def _check(text: str, labels: dict, run: Run, evidence: set[float]) -> str:
        if not text:
            return "the answer is empty"
        if REFUSAL_TEXT.lower().rstrip(".") in text.lower():
            return ""
        cited = {labels.get(f"P{n}") for n in CITE.findall(text)}
        if not cited:
            return "it cites no pages; cite the pages you read like [P2]"
        unread = [f"P{n}" for n in CITE.findall(text) if labels.get(f"P{n}") not in run.pages_read]
        if unread:
            return f"it cites {', '.join(sorted(set(unread)))} which you did not read"
        bad = unsupported_numbers(text, evidence)
        if bad:
            return f"these numbers are not on any page you read and did not come from calculate: {', '.join(bad[:5])}"
        return ""
