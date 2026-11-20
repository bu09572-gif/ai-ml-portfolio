import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import run_eval as re_  # noqa: E402
import serving as sv  # noqa: E402

from docqa.retrieve import Hit  # noqa: E402

TEXT = ("The company is based in Ohio. Capital expenditures were $1,577 million in 2018. "
        "The board met four times. Revenue grew in every segment. Employees number about 93,000.")


def test_compress_keeps_the_relevant_sentence_and_shrinks():
    h = Hit("c1", "3M_2018_10K", 5, 1.0, TEXT, "3M | 10K 2018 | page 5")
    out = sv.compress("What were capital expenditures in 2018?", [h])[0]
    assert "1,577" in out.text and len(out.text.split()) < len(TEXT.split())
    assert out.doc_id == h.doc_id and out.page_num == h.page_num


def test_unsupported_numbers_allows_scale_changes():
    assert sv.unsupported_numbers("Capex was $1.577 billion [S1].", TEXT) == 0
    assert sv.unsupported_numbers("Capex was $1,900 million.", TEXT) == 1
    assert sv.unsupported_numbers("There are 3 segments.", TEXT) == 0


def test_router_escalates_and_adds_cost():
    base = {"k": 5, "compress": False, "retrieval_hit": 1.0, "output_tokens": 10}
    df = pd.DataFrame([
        base | {"id": "a", "model": re_.SMALL, "refused": True, "citations": [], "unsupported_numbers": 0,
                "numeric_match": 0.0, "input_tokens": 100, "latency_s": 2.0},
        base | {"id": "a", "model": re_.LARGE, "refused": False, "citations": ["p"], "unsupported_numbers": 0,
                "numeric_match": 1.0, "input_tokens": 100, "latency_s": 5.0},
        base | {"id": "b", "model": re_.SMALL, "refused": False, "citations": ["p"], "unsupported_numbers": 0,
                "numeric_match": 1.0, "input_tokens": 100, "latency_s": 2.0},
        base | {"id": "b", "model": re_.LARGE, "refused": False, "citations": ["p"], "unsupported_numbers": 0,
                "numeric_match": 1.0, "input_tokens": 100, "latency_s": 5.0}])
    r = re_.router(df).set_index("id")
    assert r.loc["a", "escalated"] and r.loc["a", "latency_s"] == 7.0 and r.loc["a", "numeric_match"] == 1.0
    assert not r.loc["b", "escalated"] and r.loc["b", "latency_s"] == 2.0


def test_pareto_drops_dominated():
    t = pd.DataFrame([{"model": "a", "k": 3, "compress": False, "numeric_accuracy": 0.2, "cpu_seconds_mean": 5},
                      {"model": "b", "k": 3, "compress": False, "numeric_accuracy": 0.1, "cpu_seconds_mean": 9},
                      {"model": "c", "k": 3, "compress": False, "numeric_accuracy": 0.3, "cpu_seconds_mean": 20}])
    assert [p["model"] for p in re_.pareto(t)] == ["a", "c"]
