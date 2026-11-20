import data
import governance as g

PAGE = ("Section 5 Governance. The insurer must establish a governance structure overseen by the board of directors. "
        "This regulation was adopted in 2023 by the Division of Insurance. "
        "Insurers should test their models for unfair discrimination against protected classes.")


def test_requirements_keep_obligations_and_tag_themes():
    reqs = g.requirements([PAGE], "x")
    assert len(reqs) == 2                                         # the "was adopted" sentence is not an obligation
    assert "governance and accountability" in reqs[0]["themes"]
    assert set(reqs[1]["themes"]) >= {"unfair discrimination and bias testing", "validation, testing and monitoring"}
    assert all(r["page"] == 1 for r in reqs)


def test_crosswalk_counts():
    cw = g.crosswalk(g.requirements([PAGE], "x"))
    assert cw["governance and accountability"]["x"] == 1 and cw["third-party vendors"]["x"] == 0


def test_real_rules_and_adoption_map():
    a = data.adoption()
    assert len(a["adopted"]) == 30 and a["own_rules"] == ["California", "Colorado", "New York"]
    co = g.requirements(data.text("colorado"), "colorado")
    assert any("board of directors" in r["text"] for r in co)


def test_evidence_on_a_real_model_folder():
    ev = g.evidence("projects/finance/explainable-credit-risk")
    assert set(ev) == set(g.THEMES)
    assert ev["consumer notice and adverse decisions"]["status"] == "found"
    assert ev["governance and accountability"]["status"] == "gap"
