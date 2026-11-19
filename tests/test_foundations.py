"""Unit tests. Tiny hand-written inputs check the logic; the real numbers come from
running the pipeline and evals on FinanceBench (see README)."""
from types import SimpleNamespace

import pandas as pd
import pytest

from docpipe.chunk import build_chunks, chunk_page
from docpipe.quality import run_checks
from docqa.answer import Answer, answer, build_content
from docqa.retrieve import Retriever, detect_filters, expand_query, rrf
from evalkit import metrics as M


def _chunks():
    rows = [
        ("a1", "ACME_2022_10K", 10, "ACME", 2022, "Net sales were $1,577 million in 2022 driven by industrial demand."),
        ("a2", "ACME_2022_10K", 40, "ACME", 2022, "Purchases of property, plant and equipment were $300 million."),
        ("b1", "BETA_2021_10K", 12, "Beta Corp", 2021, "Net sales were $900 million. Purchases of property plant and equipment $50 million."),
    ]
    return pd.DataFrame([{"chunk_id": c, "doc_id": d, "page_num": p, "part": 0, "company": co, "doc_type": "10K",
                          "period": y, "sector": "Industrials", "header": f"{co} | 10K {y} | page {p}", "text": t,
                          "n_words": len(t.split())} for c, d, p, co, y, t in rows])


# ---------- pipeline ----------
def test_chunk_page_windows_overlap():
    words = " ".join(f"w{i}" for i in range(500))
    parts = chunk_page(words, size=220, overlap=40)
    assert len(parts) == 3 and parts[1].split()[0] == "w180"
    assert chunk_page("too short") == []


def test_build_chunks_keeps_page_and_metadata():
    pages = pd.DataFrame([{"doc_id": "D", "page_num": 3, "text": "word " * 30, "n_chars": 150}])
    man = pd.DataFrame([{"doc_id": "D", "company": "ACME", "doc_type": "10K", "period": 2022, "sector": "X"}])
    c = build_chunks(pages, man)
    assert c.page_num.tolist() == [3] and c.header[0] == "ACME | 10K 2022 | page 3"


def test_quality_checks_catch_missing_pages():
    man = pd.DataFrame([{"doc_id": "D1", "sha256": "x"}, {"doc_id": "D2", "sha256": "y"}])
    pages = pd.DataFrame([{"doc_id": "D1", "page_num": 0, "text": "t" * 100, "n_chars": 100}])
    checks = {c.name: c for c in run_checks(man, pages, _chunks().assign(doc_id="D1"))}
    assert not checks["every_doc_has_pages"].passed and checks["every_doc_has_pages"].observed == ["D2"]


# ---------- retrieval ----------
def test_filters_detect_company_and_year():
    assert detect_filters("What was Beta Corp's FY2021 capex?", ["ACME", "Beta Corp"]) == {"company": "Beta Corp", "period": 2021}


def test_query_expansion_bridges_vocabulary():
    assert "purchases of property plant and equipment" in expand_query("FY2022 capital expenditure")


def test_retriever_uses_filters_and_expansion():
    r = Retriever(_chunks())
    hits, f = r.search("What is ACME's FY2022 capital expenditure?", k=1)
    assert f == {"company": "ACME", "period": 2022} and hits[0].chunk_id == "a2"


def test_rrf_rewards_agreement():
    import numpy as np
    fused = rrf([np.array([0, 1, 2]), np.array([0, 2, 1])])
    assert fused.argmax() == 0


# ---------- answer ----------
def test_answer_parses_citations_with_fake_client():
    hits, _ = Retriever(_chunks()).search("ACME FY2022 net sales", k=2)
    cit = SimpleNamespace(document_index=0, cited_text="Net sales were $1,577 million")
    resp = SimpleNamespace(stop_reason="end_turn", model="claude-opus-5-5",
                           usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
                           content=[SimpleNamespace(type="text", text="$1,577 million.", citations=[cit])])
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: resp)))
    a = answer("ACME FY2022 net sales?", hits, client=client)
    assert a.citations[0].page_num == hits[0].page_num
    assert a.cost_usd == pytest.approx((1000 * 4 + 100 * 20) / 1e6)
    assert build_content("q", hits)[0]["citations"] == {"enabled": True}


def test_refusal_detection():
    assert Answer("Not enough information in the provided filings.").refused


# ---------- eval metrics ----------
@pytest.mark.parametrize("ans,gold,exp", [
    ("Capex was $1,577 million.", "$1577.00", 1.0),
    ("About 1.7 billion", "$1577.00", 0.0),           # outside 1% tolerance
    ("Margin of 24.5%", "24.5%", 1.0),
    ("Capex was $1.577 billion", "$1577.00", 1.0),
    ("No", "Yes, it is capital intensive", None),
])
def test_numeric_match(ans, gold, exp):
    assert M.numeric_match(ans, gold) == exp


def test_hit_and_mrr():
    assert M.hit_at_k(["a", "b"], {"b"}, 1) == 0 and M.hit_at_k(["a", "b"], {"b"}, 2) == 1
    assert M.mrr(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert M.citation_page_precision(["x", "y"], {"x"}) == 0.5
