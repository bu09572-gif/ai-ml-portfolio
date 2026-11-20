import json

import analyst as A
import pandas as pd
import pytest

from docqa import llm
from docqa.retrieve import Retriever


def test_safe_eval_allows_only_arithmetic():
    assert A.safe_eval("(a + b) / 2", {"a": 3, "b": 5}) == 4
    with pytest.raises(ValueError):
        A.safe_eval("__import__('os')", {})
    with pytest.raises(ValueError):
        A.safe_eval("a ** 2", {"a": 2})


@pytest.mark.parametrize("raw,want", [("1,577", 1577), ("(222)", -222), ("$8.7", 8.7), (12, 12), ("n/a", None)])
def test_to_number(raw, want):
    assert A.to_number(raw) == want


def test_appears_on_page():
    page = "Purchases of property, plant and equipment (PP&E) (1,577) (1,373)"
    assert A.appears_on_page(-1577, page)
    assert not A.appears_on_page(1578, page)
    assert not A.appears_on_page(157, page)   # part of a bigger number doesn't count


def _corpus():
    rows = [
        ("acme_2019", 0, "Revenue 1,000 900 Cost of revenue 400"),
        ("acme_2019", 1, "Purchases of property and equipment (50) (40) Depreciation 30"),
    ]
    chunks = pd.DataFrame([{"chunk_id": f"{d}:{p}", "doc_id": d, "page_num": p, "part": 0, "company": "Acme",
                            "doc_type": "10K", "period": 2019, "sector": "", "header": f"Acme | 10K 2019 | page {p + 1}",
                            "text": t, "n_words": len(t.split())} for d, p, t in rows])
    pages = pd.DataFrame([{"doc_id": d, "page_num": p, "text": t, "n_chars": len(t)} for d, p, t in rows])
    return Retriever(chunks), pages


def fake_llm(replies):
    it = iter(replies)

    def chat(system, user, **kw):
        return llm.Completion(json.dumps(next(it)), 10, 5, 0.01, "fake", "ollama")
    return chat


def test_calculator_path_grounds_and_computes(monkeypatch):
    r, pages = _corpus()
    monkeypatch.setattr(llm, "chat", fake_llm([
        {"items": [{"name": "a", "line_item": "purchases of property and equipment", "fiscal_year": 2019},
                   {"name": "b", "line_item": "revenue", "fiscal_year": 2019}],
         "formula": "-a / b", "unit": "percent", "decimals": 1},
        {"value": "(50)", "source": "S1", "scale": "millions"},
        {"value": "1,000", "source": "S1", "scale": "millions"},
    ]))
    res = A.Analyst(r, pages).ask("What is Acme's FY2019 capex as a % of revenue?")
    assert res.route == "calculator" and res.value == 5.0
    assert res.answer.text.startswith("5.0%")
    assert {c.page_num for c in res.answer.citations} == {0, 1}


def test_number_not_on_page_falls_back(monkeypatch):
    r, pages = _corpus()
    monkeypatch.setattr(llm, "chat", fake_llm([
        {"items": [{"name": "a", "line_item": "revenue", "fiscal_year": 2019}], "formula": "a", "unit": "usd millions"},
        {"value": "1234", "source": "S1", "scale": "millions"},          # made up: not on any page
    ]))
    monkeypatch.setattr(A, "plain_answer", lambda q, hits: A.Answer("Not enough information in the provided filings."))
    res = A.Analyst(r, pages).ask("What is Acme's FY2019 revenue?")
    assert res.route == "fallback" and not res.inputs[0].grounded


def test_scale_comes_from_the_page():
    assert A.page_scale("CONSOLIDATED BALANCE SHEETS (In millions, except per share data)") == "millions"
    assert A.page_scale("($ in thousands)") == "thousands"
    assert A.page_scale("Revenue 1,000") == "units"


def test_unit_conversions_are_removed_from_formulas():
    assert A.CONVERSION_RE.sub("", "(a - b) / 1000000") == "(a - b)"
    assert A.CONVERSION_RE.sub("", "(a + b) / 2") == "(a + b) / 2"
