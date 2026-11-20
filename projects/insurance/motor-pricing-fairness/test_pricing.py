"""Tests on a 3% slice of the real freMTPL2 data (downloaded once, about 9 MB)."""
import data
import numpy as np
import pricing
import pytest
import run


@pytest.fixture(scope="module")
def slice_():
    df = data.load().sample(frac=0.03, random_state=1)
    return df[~df.test], df[df.test]


def test_cleaning_rules():
    df = data.load()
    assert len(df) == 677_991 and df.IDpol.is_unique
    assert df.ClaimNb.max() == 4 and df.Exposure.max() <= 1 and df.ClaimAmount.max() <= 200_000
    assert not ((df.ClaimNb > 0) & (df.ClaimAmount == 0)).any()
    assert 0.19 < df.test.mean() < 0.21


def test_prices_are_positive_and_balanced(slice_):
    train, _ = slice_
    models = pricing.fit_prices(train, "glm")      # the GBM path is the same code, and slow to fit
    p = pricing.price(models, train)
    assert (p > 0).all().all()
    w = train.Exposure
    total = (p.aware * w).sum()
    for col in ["unaware", "discrimination_free"]:
        assert (p[col] * w).sum() == pytest.approx(total, rel=1e-6)


def test_discrimination_free_price_ignores_age(slice_):
    """Two policies that differ only in driver age must get the same discrimination-free price."""
    train, _ = slice_
    models = pricing.fit_prices(train, "glm")
    x = train.head(1)
    young, old = x.assign(DrivAge=20), x.assign(DrivAge=70)
    assert models["aware"].pure_premium(young)[0] != pytest.approx(models["aware"].pure_premium(old)[0])
    a = pricing.discrimination_free(models["aware"], young, models["p_d"])
    b = pricing.discrimination_free(models["aware"], old, models["p_d"])
    assert a[0] == pytest.approx(b[0])


def test_gini_perfect_and_random():
    rng = np.random.default_rng(0)
    exposure = np.ones(2000)
    actual = rng.gamma(0.2, 500, 2000)
    assert run.gini(actual, actual, exposure) == pytest.approx(1.0)
    assert abs(run.gini(actual, rng.random(2000), exposure)) < 0.1
