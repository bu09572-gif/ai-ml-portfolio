import data
import denials as dn
import numpy as np
import pandas as pd


def test_real_file_loads_and_totals():
    d = data.plans()
    assert len(d) == 5050 and d.State.nunique() == 32
    c = dn.composition(d)
    assert 0.15 < c["issuer_in_network_denial_rate"] < 0.17
    assert abs(sum(c["denial_reason_share"].values()) - 1) < 0.01
    assert c["internal_appeals_overturned"] < c["internal_appeals_filed"]


def test_between_issuer_share_extremes():
    p = pd.DataFrame({"Issuer_ID": ["a", "a", "b", "b"], "claims": [1, 1, 1, 1], "denial_rate": [.1, .1, .3, .3]})
    assert np.isclose(dn.between_issuer_share(p), 1.0)
    p["denial_rate"] = [.1, .3, .1, .3]
    assert np.isclose(dn.between_issuer_share(p), 0.0)


def test_wquantiles_weighted():
    assert dn.wquantiles(np.array([1., 2., 3.]), np.array([1., 1., 10.]), [0.5]) == [3.0]


def test_appeal_score():
    import appeal
    note = "Visual field testing shows superior field loss of 30 degrees with lids taped."
    ev = [{"criterion": "field loss", "quote": "superior field loss of 30 degrees"}]
    letter = ('The record states "superior field loss of 30 degrees" and "visual acuity is 20/400 bilaterally". '
              "Policy L34411 applies. Surgery was on 12 March.")
    s = appeal.score(letter, note, "Policy text.", ev)
    assert (s["quotes"], s["quotes_verified"], s["evidence_used"]) == (2, 1, 1.0)
    assert "12" in s["unsupported_numbers"] and "34411" not in s["unsupported_numbers"] and "30" not in s["unsupported_numbers"]
