"""Unit tests with a scripted model (no LLM needed) and the real question files."""
import clinical_eval as ce
import data
import pytest

from docqa import llm


def test_swap_sex():
    s = "A 45-year-old man presents with his wife. He says she noticed it. The man's father gave her a note."
    out = ce.swap_sex(s)
    assert out == "A 45-year-old woman presents with her husband. She says he noticed it. The woman's mother gave him a note."
    assert ce.swap_sex("Her blood pressure is high.") == "His blood pressure is high."


def test_sex_neutral_filter():
    assert ce.sex_neutral_question("A 30-year-old man has chest pain.", {"A": "MI"})
    assert not ce.sex_neutral_question("A 30-year-old woman who is pregnant has chest pain.", {"A": "MI"})
    assert not ce.sex_neutral_question("A man has testicular pain.", {"A": "torsion"})
    assert not ce.sex_neutral_question("A patient has chest pain.", {"A": "MI"})


def test_shuffle_moves_every_option_and_maps_back():
    opts = {"A": "a", "B": "b", "C": "c", "D": "d"}
    new, back = ce.shuffled(opts, 3)
    assert all(new[k] != opts[k] for k in "ABCD")
    assert all(new[k] == opts[back[k]] for k in "ABCD")


def test_parsers():
    assert ce.parse_letter("B") == "B" and ce.parse_letter("The answer is C.") == "C" and ce.parse_letter("none") is None
    assert ce.parse_yn("Yes.") == "yes" and ce.parse_yn("Maybe") == "maybe"


def test_medqa_row_with_a_model_that_knows_the_answer(monkeypatch):
    """A model that always picks the gold option text must be consistent under shuffling and sex swap."""
    row = data.medqa().iloc[0]
    gold_text = row.options[row.answer_idx]

    def chat(system, user, **kw):
        for line in user.splitlines():
            if line[3:] == gold_text:
                return llm.Completion(line[0], 1, 1, 0, "scripted", "test")
        return llm.Completion("A", 1, 1, 0, "scripted", "test")
    monkeypatch.setattr(llm, "chat", chat)
    r = ce.medqa_row(row, "scripted", 0)
    assert r["correct"] and r["correct_shuffled"] and r["order_consistent"]


@pytest.mark.parametrize("n,expected", [(1273, "medqa"), (500, "pubmedqa")])
def test_datasets(n, expected):
    d = data.medqa() if expected == "medqa" else data.pubmedqa()
    assert len(d) == n
