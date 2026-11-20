"""Tests on the real CodiEsp data."""
import coder
import data
import pytest


def test_corpus_shape():
    df = data.cases()
    assert df.groupby("split").size().to_dict() == {"dev": 250, "test": 250, "train": 500}
    assert (df.codes.map(len) > 0).all() and df.text.str.len().min() > 200


def test_average_precision_matches_definition():
    assert coder.average_precision(["A", "B", "C"], {"A", "C"}) == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert coder.average_precision(["X"], {"A"}) == 0


def test_ranker_learns_from_real_cases():
    df = data.cases()
    tr, dv = df[df.split == "train"], df[df.split == "dev"].head(40)
    m = coder.Coder(data.code_descriptions()).fit(tr.text.tolist(), tr.codes.tolist())
    s = m.scores(dv.text.tolist())
    freq = coder.evaluate(m.rank(s["frequency"]), dv.codes.tolist())["MAP"]
    clf = coder.evaluate(m.rank(s["classifier"]), dv.codes.tolist())["MAP"]
    assert clf > freq
    # every dictionary suggestion comes with a sentence from the note itself
    top = m.rank(s["dictionary"], top=3)[0]
    assert all(m.evidence[0][c] in dv.text.iloc[0] for c in top if c in m.evidence[0])
