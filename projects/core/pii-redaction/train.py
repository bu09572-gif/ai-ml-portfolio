"""Train a spaCy NER model on the TAB training cases to predict spans that must be masked.

    python train.py            # writes model/model-best (about 1 hour on 4 CPU cores)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans

from data import gold_spans, load

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"


def to_docbin(split: str) -> tuple[int, int]:
    nlp = spacy.blank("en")
    db, kept, dropped = DocBin(), 0, 0
    for d in load(split):
        doc = nlp.make_doc(d["text"])
        ents = []
        for s, e, t in gold_spans(d):
            span = doc.char_span(s, e, label=t, alignment_mode="expand")
            if span is None:
                dropped += 1
            else:
                ents.append(span)
        doc.ents = filter_spans(ents)      # spaCy entities can't overlap
        kept += len(doc.ents)
        db.add(doc)
    CORPUS.mkdir(exist_ok=True)
    db.to_disk(CORPUS / f"{split}.spacy")
    return kept, dropped


def main():
    for s in ["train", "dev"]:
        print(s, "entities kept/dropped:", to_docbin(s), flush=True)
    cmd = [sys.executable, "-m", "spacy", "train", str(HERE / "config.cfg"), "--output", str(HERE / "model"),
           "--paths.train", str(CORPUS / "train.spacy"), "--paths.dev", str(CORPUS / "dev.spacy")]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
