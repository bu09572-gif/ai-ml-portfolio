"""Answer numeric questions about SEC filings by finding each number, checking it, and
doing the math in Python instead of asking the model to do it.

  1. plan      the model lists the line items the question needs (name, statement, year)
               and a formula over them, e.g. capex / revenue
  2. find      for each line item: search the filing, show the model the top pages,
               and ask for the value, the page it came from, and the table's scale
  3. check     the number must actually appear on that page; if not, try the next page
  4. compute   evaluate the formula in Python, convert to the unit the question asks for
  5. answer    the result, plus every input with its page, as [S#] citations

Questions with no formula (not numeric) fall back to the plain Q&A answer.
"""
from __future__ import annotations

import ast
import json
import operator
import re
import time
from dataclasses import dataclass, field

import pandas as pd

from docqa import llm
from docqa.answer import Answer, Citation
from docqa.answer import answer as plain_answer
from docqa.retrieve import Retriever

PAGES_PER_ITEM = 3
SCALE = {"units": 1, "thousands": 1e3, "millions": 1e6, "billions": 1e9}
OUT_SCALE = {"usd": 1, "usd thousands": 1e3, "usd millions": 1e6, "usd billions": 1e9}

PLAN_SYSTEM = """You plan calculations for questions about company financial statements.
Return JSON only:
{"items": [{"name": "a", "line_item": "<line item as it appears in a filing>",
            "statement": "income statement" | "balance sheet" | "cash flow statement",
            "fiscal_year": <year>}],
 "formula": "<python arithmetic over the item names>",
 "unit": "usd" | "usd thousands" | "usd millions" | "usd billions" | "percent" | "ratio" | "days",
 "decimals": <int>}
Rules:
- One item per number you need. Use names a, b, c, ... A value in two different years is two items.
- For an average of two years, list both years and write (a + b) / 2.
- For a percent, the formula gives the plain ratio. Do not multiply by 100.
- If the answer needs no calculation and is a single number, the formula is just "a".
- If the question is not about numbers from the statements, return {"items": [], "formula": ""}."""

FIND_SYSTEM = """You read pages from a company filing and pull out one number.
Return JSON only: {"value": <number as printed, no commas, or null>, "source": "S1" | "S2" | "S3" | null,
                   "scale": "units" | "thousands" | "millions" | "billions"}
- "value" is the number exactly as printed for the requested year. Numbers in parentheses are negative.
- "scale" is what the table says its numbers are in (for example "in millions"). If it doesn't say, use "units".
- If the line item is not on any page, return {"value": null, "source": null, "scale": "units"}."""

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr: str, env: dict[str, float]) -> float:
    """Evaluate + - * / and parentheses over named numbers. Nothing else is allowed."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name) and n.id in env:
            return float(env[n.id])
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.operand))
        raise ValueError(f"not allowed in a formula: {ast.dump(n)[:60]}")
    return ev(ast.parse(expr, mode="eval"))


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in model output")
    return json.loads(m.group(0))


def to_number(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "").replace("$", "")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        x = float(s)
    except ValueError:
        return None
    return -x if neg else x


def appears_on_page(value: float, page_text: str) -> bool:
    """Is this number printed on the page (ignoring commas, sign and trailing zeros)?"""
    text = page_text.replace(",", "")
    v = abs(value)
    forms = {f"{v:.0f}"} if v == int(v) else set()
    forms |= {f"{v:.1f}", f"{v:.2f}", f"{v:g}"}
    return any(re.search(rf"(?<![\d.]){re.escape(f)}(?![\d])", text) for f in forms if f not in {"0", "0.0"})


@dataclass
class Input:
    name: str
    line_item: str
    fiscal_year: int | None
    value: float | None = None
    scale: str = "units"
    doc_id: str | None = None
    page_num: int | None = None
    pages_tried: int = 0
    grounded: bool = False


@dataclass
class Result:
    answer: Answer
    plan: dict = field(default_factory=dict)
    inputs: list[Input] = field(default_factory=list)
    value: float | None = None
    route: str = "calculator"          # or "fallback"
    llm_calls: int = 0


class Analyst:
    def __init__(self, retriever: Retriever, pages: pd.DataFrame):
        self.r = retriever
        self.pages = pages.set_index(["doc_id", "page_num"]).text

    def _call(self, system: str, user: str, res: Result, **kw):
        c = llm.chat(system, user, json_mode=True, **kw)
        res.llm_calls += 1
        res.answer.input_tokens += c.input_tokens
        res.answer.output_tokens += c.output_tokens
        res.answer.model, res.answer.provider = c.model, c.provider
        return c.text

    def _find(self, question: str, item: Input, res: Result) -> None:
        query = f"{question} {item.line_item} {item.fiscal_year or ''}"
        hits, _ = self.r.search(query, k=PAGES_PER_ITEM)
        pages = [(h.doc_id, h.page_num) for h in hits]
        texts = [self.pages.get(p, "") for p in pages]
        src = "\n\n".join(f"[S{i}] page {p[1] + 1}\n{t[:4000]}" for i, (p, t) in enumerate(zip(pages, texts), 1))
        ask = f"Line item: {item.line_item}\nFiscal year: {item.fiscal_year}\n\n{src}"
        try:
            d = parse_json(self._call(FIND_SYSTEM, ask, res, max_tokens=120))
        except (ValueError, json.JSONDecodeError):
            return
        v = to_number(d.get("value"))
        m = re.match(r"S(\d)", str(d.get("source") or ""))
        if v is None or not m or not (1 <= int(m.group(1)) <= len(pages)):
            return
        i = int(m.group(1)) - 1
        order = [i] + [j for j in range(len(pages)) if j != i]   # check the cited page first, then the rest
        for j in order:
            item.pages_tried += 1
            if appears_on_page(v, texts[j]):
                item.value, item.doc_id, item.page_num, item.grounded = v, *pages[j], True
                item.scale = d.get("scale") if d.get("scale") in SCALE else "units"
                return

    def ask(self, question: str) -> Result:
        t0 = time.perf_counter()
        res = Result(answer=Answer(text="", citations=[], input_tokens=0, output_tokens=0, latency_s=0,
                                   model=llm.model_name(), provider=llm.PROVIDER))
        try:
            plan = parse_json(self._call(PLAN_SYSTEM, question, res, max_tokens=400))
        except (ValueError, json.JSONDecodeError):
            plan = {"items": []}
        res.plan = plan
        items = [Input(str(x.get("name")), str(x.get("line_item")), _year(x.get("fiscal_year")))
                 for x in plan.get("items") or [] if x.get("name") and x.get("line_item")]
        if not items or not plan.get("formula"):
            return self._fallback(question, res, t0)
        res.inputs = items
        for it in items:
            self._find(question, it, res)
        if not all(it.grounded for it in items):
            return self._fallback(question, res, t0)
        try:
            raw = safe_eval(str(plan["formula"]), {it.name: it.value * SCALE[it.scale] for it in items})
        except (ValueError, ZeroDivisionError, SyntaxError):
            return self._fallback(question, res, t0)
        unit = str(plan.get("unit", "ratio")).lower()
        decimals = int(plan.get("decimals", 2)) if str(plan.get("decimals", "")).lstrip("-").isdigit() else 2
        if unit in OUT_SCALE:
            value, shown = raw / OUT_SCALE[unit], unit.replace("usd", "USD")
        elif unit == "percent":
            value, shown = raw * 100, "%"
        else:
            value, shown = raw, unit if unit != "ratio" else ""
        res.value = round(value, decimals)
        cites, lines = [], []
        for n, it in enumerate(items, 1):
            cites.append(Citation(f"S{n}", it.doc_id, it.page_num))
            lines.append(f"- {it.name} = {it.line_item}, FY{it.fiscal_year}: {it.value:g} ({it.scale}) [S{n}]")
        res.answer.text = (f"{res.value:,.{max(decimals, 0)}f}{'%' if shown == '%' else (' ' + shown if shown else '')}\n"
                           f"Formula: {plan['formula']}\n" + "\n".join(lines))
        res.answer.citations = cites
        res.answer.latency_s = time.perf_counter() - t0
        return res

    def _fallback(self, question: str, res: Result, t0: float) -> Result:
        hits, _ = self.r.search(question, k=5)
        a = plain_answer(question, hits)
        a.input_tokens += res.answer.input_tokens
        a.output_tokens += res.answer.output_tokens
        a.latency_s = time.perf_counter() - t0
        res.answer, res.route = a, "fallback"
        res.llm_calls += 1
        return res


def _year(y) -> int | None:
    m = re.search(r"(19|20)\d\d", str(y))
    return int(m.group(0)) if m else None
