"""Rank ICD-10 diagnosis codes for a clinical note, so a human coder reviews a short list.

Four rankers, each giving every candidate code a score per note:
  frequency    codes ranked by how often they appear in training (the floor any method must beat)
  classifier   TF-IDF + one-vs-rest logistic regression, for codes seen in at least 2 training notes
  neighbours   codes of the 20 most similar training notes, weighted by similarity
  dictionary   best match between a sentence of the note and the code's official English
               description, so it can suggest codes never seen in training
  ensemble     weighted sum of the four after scaling each to [0, 1]; weights tuned on dev
For each suggested code the dictionary ranker also returns the sentence that matched: the evidence
a coder checks before accepting the code.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

SENT = re.compile(r"(?<=[.;:])\s+|\n+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT.split(text) if len(s.strip()) > 15]


def _scale(m: np.ndarray) -> np.ndarray:
    hi = m.max(axis=1, keepdims=True)
    return np.divide(m, hi, out=np.zeros_like(m), where=hi > 0)


class Coder:
    def __init__(self, descriptions: pd.Series, min_docs: int = 2, k: int = 20):
        self.desc, self.min_docs, self.k = descriptions, min_docs, k

    def fit(self, texts: list[str], codes: list[list[str]]) -> Coder:
        self.freq = Counter(c for cs in codes for c in cs)
        # candidates: codes seen in training, plus codes in the same 3-character ICD-10 categories
        cats = {c[:3] for c in self.freq}
        self.dict_codes = sorted(set(self.freq) | {c for c in self.desc.index if c[:3] in cats})
        self.vocab = self.dict_codes
        self.idx = {c: i for i, c in enumerate(self.vocab)}
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
        x = self.tfidf.fit_transform(texts)
        keep = [c for c, n in self.freq.items() if n >= self.min_docs]
        self.mlb = MultiLabelBinarizer(classes=sorted(keep))
        y = self.mlb.fit_transform([[c for c in cs if c in set(keep)] for cs in codes])
        self.clf = OneVsRestClassifier(LogisticRegression(C=10, max_iter=2000, class_weight="balanced"), n_jobs=-1)
        self.clf.fit(x, y)
        self.train_x, self.train_y = x, MultiLabelBinarizer(classes=self.vocab).fit_transform(codes)
        # dictionary: code descriptions and notes in one TF-IDF space
        texts_d = [self._describe(c) for c in self.dict_codes]
        self.dict_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit(texts_d + texts)
        self.dict_m = self.dict_vec.transform(texts_d)
        return self

    def _describe(self, code: str) -> str:
        if code in self.desc.index:
            return self.desc[code]
        child = self.desc[self.desc.index.str.startswith(code)]
        return child.iloc[0] if len(child) else code

    def scores(self, texts: list[str]) -> dict[str, np.ndarray]:
        n, v = len(texts), len(self.vocab)
        out = {}
        f = np.zeros(v)
        for c, cnt in self.freq.items():
            f[self.idx[c]] = cnt
        out["frequency"] = np.tile(f / f.max(), (n, 1))
        x = self.tfidf.transform(texts)
        p = self.clf.predict_proba(x)
        m = np.zeros((n, v))
        m[:, [self.idx[c] for c in self.mlb.classes_]] = p
        out["classifier"] = m
        sim = (x @ self.train_x.T).toarray()
        top = np.argsort(-sim, axis=1)[:, : self.k]
        nb = np.zeros((n, v))
        for i in range(n):
            nb[i] = sim[i, top[i]] @ self.train_y[top[i]]
        out["neighbours"] = _scale(nb)
        d, self.evidence = np.zeros((n, v)), []
        for i, t in enumerate(texts):
            sents = sentences(t) or [t]
            s = (self.dict_vec.transform(sents) @ self.dict_m.T).toarray()      # sentences x codes
            best = s.argmax(axis=0)
            cols = [self.idx[c] for c in self.dict_codes]
            d[i, cols] = s.max(axis=0)
            self.evidence.append({c: sents[best[j]] for j, c in enumerate(self.dict_codes) if s[best[j], j] > 0})
        out["dictionary"] = d
        return out

    @staticmethod
    def ensemble(s: dict[str, np.ndarray], w: dict[str, float]) -> np.ndarray:
        return sum(w[k] * s[k] for k in w)

    def rank(self, scores: np.ndarray, top: int = 20) -> list[list[str]]:
        return [[self.vocab[j] for j in np.argsort(-row)[:top]] for row in scores]


def average_precision(ranked: list[str], gold: set[str]) -> float:
    """CodiEsp's MAP: precision at each correct code, summed, divided by the number of gold codes."""
    hits, total = 0, 0.0
    for i, c in enumerate(ranked, 1):
        if c in gold:
            hits += 1
            total += hits / i
    return total / len(gold) if gold else 0.0


def evaluate(ranked: list[list[str]], gold: list[list[str]]) -> dict:
    g = [set(x) for x in gold]
    return {"MAP": round(float(np.mean([average_precision(r, s) for r, s in zip(ranked, g)])), 4),
            "precision_at_5": round(float(np.mean([len(set(r[:5]) & s) / 5 for r, s in zip(ranked, g)])), 4),
            "recall_at_10": round(float(np.mean([len(set(r[:10]) & s) / len(s) for r, s in zip(ranked, g) if s])), 4),
            "recall_at_20": round(float(np.mean([len(set(r[:20]) & s) / len(s) for r, s in zip(ranked, g) if s])), 4)}

