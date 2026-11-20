import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import criteria as cr  # noqa: E402

NOTE = "Patient has dermatochalasis.\nVisual field testing shows superior field loss of 30 degrees with lids taped."


def test_verified_needs_exact_words():
    assert cr.verified("visual field testing shows superior field loss", NOTE)
    assert cr.verified("Visual  field testing, shows superior", NOTE)        # case, spacing, punctuation
    assert not cr.verified("visual field testing shows inferior field loss", NOTE)
    assert not cr.verified("field loss", NOTE)                               # too short to count


def test_passages_split_on_lines():
    text = "\n".join(["word " * 100] * 7)
    ps = cr.passages(text, words=350)
    assert len(ps) == 3 and all(len(p.split()) <= 350 for p in ps)
    assert sum(len(p.split()) for p in ps) == 700


def test_check_unverified_quote_becomes_not_documented(monkeypatch):
    class C:
        text, latency_s = '{"status": "met", "quote": "visual fields are normal"}', 0.1
    monkeypatch.setattr(cr.llm, "chat", lambda *a, **k: C)
    r = cr.check(NOTE, "visual field loss", "m")
    assert r["model_status"] == "met" and not r["quote_verified"] and r["status"] == "not_documented"


def test_check_verified_quote_kept(monkeypatch):
    class C:
        text, latency_s = '{"status": "met", "quote": "superior field loss of 30 degrees"}', 0.1
    monkeypatch.setattr(cr.llm, "chat", lambda *a, **k: C)
    assert cr.check(NOTE, "visual field loss", "m")["status"] == "met"


def test_extract_drops_invented_criteria(monkeypatch):
    class C:
        text, latency_s = '{"criteria": ["Visual field testing shows superior field loss", "Patient is over 65 years old"]}', 0.1
    monkeypatch.setattr(cr.llm, "chat", lambda *a, **k: C)
    out = cr.extract(NOTE, "m")
    assert [c["verified"] for c in out] == [True, False]


def test_decision():
    assert cr.decision([{"status": "met"}, {"status": "met"}]) == "ready to submit"
    assert cr.decision([{"status": "met"}, {"status": "not_documented"}]) == "missing documentation"
    assert cr.decision([{"status": "not_met"}, {"status": "not_documented"}]) == "criterion not met"
    assert cr.decision([{"status": None}]) == "error"


def test_relevant_keeps_matching_criteria():
    crits = [f"unrelated criterion number {i}" for i in range(10)] + ["superior visual field loss documented"]
    assert "superior visual field loss documented" in cr.relevant(crits, NOTE, k=3)


def test_check_numbered_quotes_are_note_sentences(monkeypatch):
    class C:
        text, latency_s = '{"status": "met", "evidence": [2, 9]}', 0.1
    monkeypatch.setattr(cr.llm, "chat", lambda *a, **k: C)
    r = cr.check_numbered(NOTE, "superior visual field loss", "m")
    assert r["status"] == "met" and r["evidence"] == [2]          # 9 is out of range and dropped
    assert cr.verified(r["quote"], NOTE)


def test_check_numbered_without_evidence_is_not_documented(monkeypatch):
    class C:
        text, latency_s = '{"status": "not_met", "evidence": []}', 0.1
    monkeypatch.setattr(cr.llm, "chat", lambda *a, **k: C)
    assert cr.check_numbered(NOTE, "visual field loss", "m")["status"] == "not_documented"
