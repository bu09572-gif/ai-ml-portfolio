import redact as R
from data import gold_spans


def test_rules_find_dates_codes_and_money():
    t = "The applicant lodged application no. 36110/97 on 3 March 1997 and claimed EUR 5,000."
    found = {t[s:e] for s, e, _ in R.rules(t)}
    assert "36110/97" in " ".join(found)
    assert "3 March 1997" in found
    assert any("5,000" in f for f in found)


def test_merge_keeps_longest_label():
    assert R.merge([(0, 5, "A"), (3, 12, "B"), (20, 25, "C")]) == [(0, 12, "B"), (20, 25, "C")]


def test_redact_replaces_spans():
    assert R.redact("Mr Galip Yalman lives in Ankara.", [(3, 15, "PERSON"), (25, 31, "LOC")]) == \
        "Mr [PERSON] lives in [LOC]."


def test_gold_spans_keeps_only_masked_mentions():
    doc = {"annotations": {"a1": {"entity_mentions": [
        {"start_offset": 0, "end_offset": 4, "entity_type": "PERSON", "identifier_type": "DIRECT"},
        {"start_offset": 10, "end_offset": 16, "entity_type": "ORG", "identifier_type": "NO_MASK"},
        {"start_offset": 20, "end_offset": 24, "entity_type": "DATETIME", "identifier_type": "QUASI"}]}}}
    assert gold_spans(doc) == [(0, 4, "PERSON"), (20, 24, "DATETIME")]
