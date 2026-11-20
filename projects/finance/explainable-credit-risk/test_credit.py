"""Tests on the real UCI data (1.6 MB, downloaded once from this repo's data-v1 release)."""
import credit
import data
import numpy as np
import pytest
import run


@pytest.fixture(scope="module")
def setup():
    df = data.load()
    x = data.features(df)
    tr, va = df.split == "train", df.split == "valid"
    m = credit.fit("xgb_monotone", x[tr], df.default[tr], x[va], df.default[va])
    return df, x, m


def test_data_and_no_protected_inputs(setup):
    df, x, _ = setup
    assert len(df) == 30_000 and df.default.mean() == pytest.approx(0.2212, abs=1e-4)
    assert not set(x.columns) & {"sex", "marriage", "age", "age_group", "education"}


def test_monotone_in_delay_and_limit(setup):
    """More months late can only raise predicted risk; a higher limit can only lower it."""
    _, x, m = setup
    rows = x.sample(200, random_state=0)
    for col, values, direction in [("delay_sep", [-2, -1, 0, 1, 2, 3, 6], 1),
                                   ("limit", [10_000, 50_000, 200_000, 500_000], -1)]:
        preds = np.array([credit.predict(m, rows.assign(**{col: v})) for v in values])
        assert (direction * np.diff(preds, axis=0) >= -1e-7).all(), col


def test_reasons_are_positive_and_one_per_family(setup):
    _, x, m = setup
    rows = x[x.max_delay > 0].head(20)
    sv = credit.shap_values(m, rows)
    for r, s in zip(credit.reasons(m, rows), sv):
        assert 1 <= len(r) <= 4
        assert len({credit.reason_family(t) for t in r}) == len(r)
        assert (s > 0).sum() >= len(r)


def test_adverse_impact_ratio():
    import pandas as pd
    t = pd.DataFrame({"sex": ["a"] * 4 + ["b"] * 4, "age_group": "x", "marriage": "y", "default": 0})
    approve = np.array([1, 1, 1, 1, 1, 1, 0, 0], bool)
    out = run.fairness(t, approve, np.zeros(8))
    assert out["sex"]["b"]["adverse_impact_ratio"] == 0.5 and out["sex"]["a"]["adverse_impact_ratio"] == 1.0
