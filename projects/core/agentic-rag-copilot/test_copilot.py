import json

import copilot as C
import pandas as pd

from docqa import llm
from docqa.retrieve import Retriever


def corpus():
    rows = [("acme_2019", 0, "Consolidated statement of cash flows. Purchases of property and equipment (1,577) (1,373)"),
            ("acme_2019", 1, "Net sales 32,765 31,657 Cost of sales 16,682")]
    chunks = pd.DataFrame([{"chunk_id": f"{d}:{p}", "doc_id": d, "page_num": p, "part": 0, "company": "Acme",
                            "doc_type": "10K", "period": 2019, "sector": "", "header": f"Acme | 10K 2019 | page {p + 1}",
                            "text": t, "n_words": len(t.split())} for d, p, t in rows])
    pages = pd.DataFrame([{"doc_id": d, "page_num": p, "text": t, "n_chars": len(t)} for d, p, t in rows])
    return Copilot(chunks, pages)


def Copilot(chunks, pages):
    return C.Copilot(Retriever(chunks), pages)


def script(monkeypatch, actions):
    it = iter(actions)
    seen = []

    def chat(system, user, **kw):
        seen.append(user)
        return llm.Completion(json.dumps(next(it)), 10, 5, 0.01, "fake", "ollama")
    monkeypatch.setattr(llm, "chat", chat)
    return seen


def test_unsupported_numbers():
    ev = {1577.0, 32765.0, 4.8131}
    assert C.unsupported_numbers("Capex was $1,577 million [P1], 4.8% of sales [P2] in 2019.", ev) == []
    assert C.unsupported_numbers("Capex was $1.58 billion [P1]", ev) == []
    assert C.unsupported_numbers("Capex was $1,800 million [P1]", ev) == ["$1,800"]


def test_safe_calc_rejects_code():
    assert C.safe_calc("(1,577 / 32765) * 100") == 1577 / 32765 * 100
    for bad in ["__import__('os')", "2**10", "a+1"]:
        try:
            C.safe_calc(bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_happy_path_with_calculate(monkeypatch):
    script(monkeypatch, [
        {"action": "search", "query": "capex"},
        {"action": "read", "source": "P1"},
        {"action": "read", "source": "P2"},
        {"action": "calculate", "expression": "1577 / 32765 * 100"},
        {"action": "answer", "text": "Capex was $1,577 million [P1], or 4.8% of net sales [P2]."},
    ])
    run = corpus().ask("What was Acme's FY2019 capex as a share of sales?")
    assert run.answer.text.startswith("Capex was") and not run.guard
    assert {c.page_num for c in run.answer.citations} == {0, 1}


def test_made_up_number_gets_one_retry_then_refused(monkeypatch):
    script(monkeypatch, [
        {"action": "search", "query": "capex"},
        {"action": "read", "source": "P1"},
        {"action": "answer", "text": "Capex was $2,000 million [P1]."},
        {"action": "answer", "text": "Capex was $2,100 million [P1]."},
    ])
    run = corpus().ask("What was Acme's FY2019 capex?")
    assert run.revised and run.answer.refused and "not on any page" in run.guard


def test_citing_an_unread_page_is_caught(monkeypatch):
    seen = script(monkeypatch, [
        {"action": "search", "query": "capex"},
        {"action": "answer", "text": "Capex was $1,577 million [P1]."},
        {"action": "read", "source": "P1"},
        {"action": "answer", "text": "Capex was $1,577 million [P1]."},
    ])
    run = corpus().ask("What was Acme's FY2019 capex?")
    assert "did not read" in seen[2] and not run.guard and not run.answer.refused


def test_step_limit(monkeypatch):
    script(monkeypatch, [{"action": "search", "query": "x"}] * (C.MAX_STEPS + 1))
    run = corpus().ask("What was Acme's FY2019 capex?")
    assert run.answer.refused and run.guard == "ran out of steps"


def test_model_error_ends_question_not_run(monkeypatch):
    def boom(system, user, **kw):
        raise RuntimeError("500 from ollama")
    monkeypatch.setattr(llm, "chat", boom)
    run = corpus().ask("What was Acme's FY2019 capex?")
    assert run.answer.refused and run.guard.startswith("model error")


def test_prompt_stays_short():
    big = "x" * 4000
    t = ["Question: q"] + [f'You: {{"action": "read"}}\nResult: [P{i}] page {i} of doc:\n{big}' for i in range(10)]
    out = C._fit(t)
    assert len(out) <= C.PROMPT_CHARS and out.startswith("Question: q")


def test_missing_citation_is_added_from_pages_read(monkeypatch):
    script(monkeypatch, [
        {"action": "search", "query": "capex"},
        {"action": "read", "source": "P1"},
        {"action": "answer", "text": "Capex was $1,577 million."},
    ])
    run = corpus().ask("What was Acme's FY2019 capex?")
    assert run.auto_cited and not run.guard and run.answer.citations[0].page_num == 0
