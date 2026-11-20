"""Three ways to find what to mask in a document, from simplest to strongest.

  rules     regular expressions for dates, case/application numbers, money, ages, emails, phones
  ner       off-the-shelf spaCy NER (en_core_web_md), masking people, places, organisations,
            dates, nationalities and numbers
  trained   a spaCy NER model trained on the TAB training cases to predict which spans need
            masking (people, codes, places, organisations, dates, demographics, quantities, misc)
  trained+rules  union of the two
  trained+ner    union of the trained model and off-the-shelf NER (highest recall)

Each returns (start, end) character spans. `redact()` replaces spans with [TYPE] tags.
"""
from __future__ import annotations

import re
from pathlib import Path

import spacy

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model" / "model-best"

MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
RULES = {
    "DATETIME": [rf"\b\d{{1,2}}\s+{MONTHS}\s+\d{{4}}\b", rf"\b{MONTHS}\s+\d{{4}}\b", rf"\b{MONTHS}\s+\d{{1,2}},\s+\d{{4}}\b",
                 r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b", r"\b(?:19|20)\d{2}\b"],
    "CODE": [r"\b\d{2,6}/\d{2,4}\b", r"\bno\.\s*[A-Z0-9][\w./-]{2,}", r"\b[A-Z]{1,4}-?\d{3,}[A-Z0-9/-]*\b"],
    "QUANTITY": [r"(?:EUR|USD|GBP|CHF|TRY|RUB|PLN|UAH|€|\$|£)\s?\d[\d,. ]*\d", r"\b\d[\d,. ]*\s?(?:euros?|dollars|pounds|roubles|lira|zlotys)\b",
                 r"\b(?:aged|age of)\s+\d{1,3}\b"],
    "CONTACT": [r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", r"\+?\d[\d ()-]{8,}\d"],
}
RULES_RE = [(t, re.compile(p)) for t, ps in RULES.items() for p in ps]
NER_MASK = {"PERSON", "GPE", "LOC", "ORG", "FAC", "NORP", "DATE", "TIME", "MONEY", "CARDINAL", "QUANTITY", "PERCENT"}


def merge(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Merge overlapping spans; the longer span's label wins."""
    out: list[tuple[int, int, str]] = []
    for s, e, t in sorted(spans):
        if out and s <= out[-1][1]:
            ps, pe, pt = out[-1]
            out[-1] = (ps, max(pe, e), pt if pe - ps >= e - s else t)
        else:
            out.append((s, e, t))
    return out


def rules(text: str) -> list[tuple[int, int, str]]:
    return merge([(m.start(), m.end(), t) for t, r in RULES_RE for m in r.finditer(text)])


class Ner:
    def __init__(self, model: str = "en_core_web_md", labels: set[str] | None = NER_MASK):
        self.nlp = spacy.load(model, disable=["lemmatizer"])
        self.nlp.max_length = 2_000_000
        self.labels = labels

    def __call__(self, text: str) -> list[tuple[int, int, str]]:
        doc = self.nlp(text)
        return [(e.start_char, e.end_char, e.label_) for e in doc.ents if self.labels is None or e.label_ in self.labels]


def trained() -> Ner:
    return Ner(str(MODEL_DIR), labels=None)


def redact(text: str, spans: list[tuple[int, int, str]]) -> str:
    out, last = [], 0
    for s, e, t in merge(spans):
        out += [text[last:s], f"[{t}]"]
        last = e
    return "".join(out + [text[last:]])
