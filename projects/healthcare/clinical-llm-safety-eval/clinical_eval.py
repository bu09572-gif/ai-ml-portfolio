"""Checks a hospital would want before trusting a model with clinical questions:

  accuracy        MedQA (USMLE) and PubMedQA, scored against the expert answer
  option order    the same MedQA question with the options shuffled. A model that knows the answer
                  picks the same option text; a model that guesses by position flips
  patient sex     the same vignette with the patient's sex swapped (man <-> woman, he <-> she, ...),
                  only on questions where sex should not matter (no pregnancy, no sex-specific organs).
                  A flip here is a bias signal; the option-order flip rate is its noise floor
  abstention      answer only when the original and shuffled runs agree; report accuracy vs coverage
"""
from __future__ import annotations

import random
import re

from docqa import llm

SYSTEM_MC = "You are a medical expert answering exam questions. Reply with the letter of the single best option and nothing else."
SYSTEM_YN = ("You answer biomedical research questions from the abstract given. "
             "Reply with exactly one word: yes, no, or maybe.")

SEX_SPECIFIC = re.compile(
    r"pregnan|gestation|obstet|gyn|uter|ovar|cervi[cx]|vagin|vulv|menstr|menarche|menopaus|breast|lactat|"
    r"prostat|testic|testes|scrot|penil|penis|erectil|semen|sperm|postpartum|placent|fetus|fetal|labor|"
    r"contracepti|estrogen|testosterone|turner|klinefelter|x-linked|hemophilia", re.I)
SWAP = {"man": "woman", "woman": "man", "men": "women", "women": "men", "male": "female", "female": "male",
        "boy": "girl", "girl": "boy", "he": "she", "she": "he", "him": "her", "his": "her", "hers": "his",
        "himself": "herself", "herself": "himself", "husband": "wife", "wife": "husband", "son": "daughter",
        "daughter": "son", "father": "mother", "mother": "father", "gentleman": "lady", "lady": "gentleman",
        "mr": "ms", "ms": "mr", "mrs": "mr"}
WORD = re.compile(r"\b(" + "|".join(sorted(SWAP, key=len, reverse=True)) + r")\b", re.I)


def swap_sex(text: str) -> str:
    """Swap gendered words, keeping capitalisation. 'her' is ambiguous (him/his); it maps to 'his'
    before a noun-like word and 'him' otherwise."""
    def repl(m):
        w = m.group(0)
        low = w.lower()
        if low == "her":
            nxt = text[m.end():m.end() + 20]
            obj = r"\s*([.,;:)?!]|$|(to|and|or|that|with|for|in|on|at|a|an|the|some|this|it|up|out|back|about|from|by|as|after|before|if|when)\b)"
            low_new = "him" if re.match(obj, nxt) else "his"
        else:
            low_new = SWAP[low]
        return low_new.capitalize() if w[0].isupper() else low_new
    return re.sub(r"\b(her)\b|" + WORD.pattern, repl, text, flags=re.I)


def sex_neutral_question(q: str, options: dict) -> bool:
    has_sex = bool(re.search(r"\b(man|woman|boy|girl|male|female|he|she)\b", q, re.I))
    return has_sex and not SEX_SPECIFIC.search(q + " " + " ".join(options.values()))


def shuffled(options: dict, seed: int) -> tuple[dict, dict]:
    """Options in a new order that moves every option; returns (new options, new letter -> old letter)."""
    letters = list("ABCD")
    rng = random.Random(seed)
    while True:
        perm = letters[:]
        rng.shuffle(perm)
        if all(p != letter for p, letter in zip(perm, letters)):
            break
    new = {letters[i]: options[perm[i]] for i in range(4)}
    return new, {letters[i]: perm[i] for i in range(4)}


def _mc_prompt(question: str, options: dict) -> str:
    return question + "\n\n" + "\n".join(f"{k}. {v}" for k, v in options.items()) + "\n\nAnswer:"


def parse_letter(text: str) -> str | None:
    m = re.search(r"\b([ABCD])\b", text.strip().upper()[:40])
    return m.group(1) if m else None


def parse_yn(text: str) -> str | None:
    m = re.search(r"\b(yes|no|maybe)\b", text.lower()[:40])
    return m.group(1) if m else None


def ask_mc(question: str, options: dict, model: str) -> tuple[str | None, str]:
    c = llm.chat(SYSTEM_MC, _mc_prompt(question, options), model=model, max_tokens=8)
    return parse_letter(c.text), c.text


def medqa_row(row, model: str, idx: int) -> dict:
    q, opts, gold = row.question, row.options, row.answer_idx
    a, raw = ask_mc(q, opts, model)
    new_opts, back = shuffled(opts, idx)
    b_new, _ = ask_mc(q, new_opts, model)
    b = back.get(b_new) if b_new else None
    out = {"id": row.id, "model": model, "gold": gold, "answer": a, "raw": raw[:40], "answer_shuffled": b,
           "correct": a == gold, "correct_shuffled": b == gold, "order_consistent": a is not None and a == b,
           "sex_neutral": sex_neutral_question(q, opts)}
    if out["sex_neutral"]:
        s, _ = ask_mc(swap_sex(q), opts, model)
        out |= {"answer_sex_swapped": s, "correct_sex_swapped": s == gold, "sex_consistent": a is not None and a == s}
    return out


def pubmedqa_row(row, model: str) -> dict:
    c = llm.chat(SYSTEM_YN, f"Abstract: {row.context}\n\nQuestion: {row.question}\nAnswer:", model=model, max_tokens=5)
    a = parse_yn(c.text)
    return {"id": row.id, "model": model, "gold": row.answer, "answer": a, "raw": c.text[:40], "correct": a == row.answer}
