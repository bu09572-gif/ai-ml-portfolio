import pandas as pd
import redteam as rt
import run_eval


def test_prompts_fill_goal_and_target():
    p = rt.prompt("prefix_injection", "Explain X", "Sure, here is X")
    assert p.startswith("Explain X") and p.endswith('"Sure, here is X"')
    assert all("{goal}" in t for t in rt.ATTACKS.values())


def test_refusal_detection():
    assert rt.is_refusal("I'm sorry, but I can't help with that.")
    assert rt.is_refusal("I cannot provide instructions for this.")
    assert not rt.is_refusal("Here is a short history of the printing press.")


def test_self_check_reads_verdict(monkeypatch):
    class C:
        text = "HARMFUL"
    monkeypatch.setattr(rt.llm, "chat", lambda *a, **k: C)
    assert rt.self_check("x", "m") is True
    C.text = "SAFE"
    assert rt.self_check("x", "m") is False


def test_summary_counts_blocks_and_unsafe():
    def row(kind, attack, blocked, unsafe_none, refusal_none, unsafe_sp=False):
        return {"model": "m", "kind": kind, "behavior": "b", "category": "c", "attack": attack,
                "self_check_blocked": blocked, "none": {"unsafe": unsafe_none, "refusal": refusal_none},
                "safety_prompt": {"unsafe": unsafe_sp, "refusal": True}}
    df = pd.DataFrame([row("harmful", "direct", True, True, False), row("harmful", "direct", False, False, True),
                       row("benign", "direct", True, False, False), row("benign", "direct", False, False, False)])
    s = run_eval.summarize(df)["models"]["m"]
    assert s["attack_success_rate"]["none"]["direct"] == 0.5
    assert s["attack_success_rate"]["self_check"]["direct"] == 0.0       # the unsafe one was blocked first
    assert s["benign_refusal_rate"]["self_check"] == 0.5 and s["benign_refusal_rate"]["none"] == 0.0
