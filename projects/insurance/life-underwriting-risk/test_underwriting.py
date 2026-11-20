import data
import numpy as np
import pandas as pd
import underwriting as uw


def test_rule_class():
    d = pd.DataFrame({"bmi": [24, 24, 36, 24], "sbp": [120, 120, 120, 150], "dbp": [80, 80, 80, 85],
                      "current_smoker": [0, 1, 0, 0], **{h: [0, 0, 0, 0] for h in uw.HISTORY}})
    assert uw.rule_class(d).tolist() == ["preferred", "smoker", "standard", "standard"]
    d.loc[0, "stroke"] = 1
    assert uw.rule_class(d)[0] == "standard"


def test_scores_and_actual_to_expected():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    p = np.array([.9, .1, .1, .1, .8, .1, .1, .1, .1, .1])
    s = uw.scores(y, p)
    assert s["auc"] == 1.0 and s["deaths_in_riskiest_10pct"] == 1.0
    ae = uw.actual_to_expected(y, p, pd.Series(["a"] * 5 + ["b"] * 5))
    assert ae["a"]["deaths"] == 2 and ae["a"]["actual_to_expected"] == 1.0
    assert ae["b"]["deaths"] == 0 and ae["b"]["actual_to_expected"] == 0.0


def test_real_cohort():
    d = data.cohort()
    assert len(d) == 9670 and d.died_8y.sum() == 1124
    assert d.age.between(40, 79).all() and set(d.cycle) == {2005, 2007, 2009}
    assert d[uw.FULL_LABS].notna().mean().min() > 0.85
