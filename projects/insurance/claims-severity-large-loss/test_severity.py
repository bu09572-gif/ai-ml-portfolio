"""Tests on the real freMTPL2 claims."""
import data
import numpy as np
import pytest
import run
from scipy import stats


@pytest.fixture(scope="module")
def claims():
    return data.claims()


def test_join_keeps_every_claim(claims):
    assert len(claims) == 26_444 and claims.ClaimAmount.max() > 4_000_000     # uncapped here
    assert claims[data.FEATURES].notna().all().all()


def test_fixed_fee_amounts_found_on_train_only(claims):
    tr = claims[~claims.test]
    comp, flat = run.composition(tr, claims)
    assert 1204.0 in flat and len(flat) >= 5
    assert 0.3 < comp["fixed_fee_share_of_claims"] < 0.6


def test_gpd_fit_recovers_known_shape():
    rng = np.random.default_rng(0)
    exc = stats.genpareto.rvs(0.5, scale=1000, size=5000, random_state=rng)
    xi, _, sigma = stats.genpareto.fit(exc, floc=0)
    assert xi == pytest.approx(0.5, abs=0.08) and sigma == pytest.approx(1000, rel=0.1)


def test_gini_bounds():
    a = np.arange(1, 101, dtype=float)
    assert run.lorenz_gini(a, a) == pytest.approx(1.0)
    assert run.lorenz_gini(a, -a) == pytest.approx(-1.0, abs=0.01)
