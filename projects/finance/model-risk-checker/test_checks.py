"""Tests on the real UCI credit data with a quick logistic model: each check must flag its fault."""
import checks
import numpy as np
import pandas as pd
import pytest
import run
from sklearn.linear_model import LogisticRegression


@pytest.fixture(scope="module")
def base():
    data = run._load("explainable-credit-risk", "data", "credit_data_t")
    df = data.load()
    x, y = data.features(df)[["limit", "delay_sep", "max_delay", "utilization_sep", "paid_sep"]], df.default.to_numpy()
    tr, te = (df.split == "train").to_numpy(), (df.split == "test").to_numpy()
    lr = LogisticRegression(max_iter=2000).fit(x[tr], y[tr])
    return checks.ModelUnderTest(
        name="test", predict=lambda z: lr.predict_proba(z[x.columns])[:, 1], x_train=x[tr], y_train=y[tr],
        x_test=x[te], y_test=y[te], protected_test=df.loc[te, ["sex"]], decision_share=0.8), df


def status(findings, name):
    return next(f.status for f in findings if f.check.startswith(name))


def test_psi():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert checks.psi(a, a) == pytest.approx(0, abs=1e-9)
    assert checks.psi(a, a + 1) > 0.25


def test_clean_model_passes_core_checks(base):
    m, _ = base
    f = checks.validate(m)
    for name in ["missing values", "single-feature leakage", "score PSI", "observed / expected"]:
        assert status(f, name) == "PASS", name


def test_leak_duplicates_drift_and_calibration_are_flagged(base):
    m, _ = base
    leak = m.x_train.assign(flag=m.y_train)
    assert status(checks.data_checks(checks.ModelUnderTest(**{**m.__dict__, "x_train": leak})),
                  "single-feature leakage") == "FAIL"
    dup = checks.ModelUnderTest(**{**m.__dict__, "x_train": pd.concat([m.x_train, m.x_test]),
                                         "y_train": np.r_[m.y_train, m.y_test]})
    assert status(checks.data_checks(dup), "test rows also in train") == "FAIL"
    drift = checks.ModelUnderTest(**{**m.__dict__, "x_test": m.x_test.assign(limit=m.x_test.limit * 0.3)})
    assert status(checks.stability_checks(drift), "feature PSI") == "FAIL"
    hot = checks.ModelUnderTest(**{**m.__dict__, "predict": lambda z: np.clip(m.predict(z) * 1.6, 0, 1)})
    assert status(checks.calibration_checks(hot), "observed / expected") == "FAIL"


def test_fairness_flags_a_penalised_group(base):
    m, df = base
    male = (df.sex == "male").to_numpy()
    biased = checks.ModelUnderTest(**{**m.__dict__, "predict": lambda z: m.predict(z) + 0.3 * male[z.index]})
    assert status(checks.fairness_checks(biased), "adverse impact ratio by sex") == "FAIL"


def test_monotone_check(base):
    m, _ = base
    ok = checks.ModelUnderTest(**{**m.__dict__, "monotone": {"delay_sep": 1}})
    wrong = checks.ModelUnderTest(**{**m.__dict__, "monotone": {"delay_sep": -1}})
    assert status(checks.robustness_checks(ok), "monotone") == "PASS"
    assert status(checks.robustness_checks(wrong), "monotone") == "FAIL"
