"""Answer a question from retrieved chunks with Claude, with page-level citations.

Each retrieved chunk is passed as its own `document` content block with
citations enabled, so every sentence of the answer comes back linked to the
exact chunk (and therefore filing + page) it was drawn from. If the sources
don't contain the answer, the model is told to say so instead of guessing.

Needs ANTHROPIC_API_KEY (set it as a Codespaces secret; never commit it).
Model and effort are configurable: DOCQA_MODEL, DOCQA_EFFORT.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from .retrieve import Hit

MODEL = os.environ.get("DOCQA_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("DOCQA_EFFORT", "medium")
# USD per million tokens (input, output); used for the cost dashboard. Update if pricing changes.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}

SYSTEM = (
    "You answer questions about public-company SEC filings using ONLY the provided documents. "
    "Show the numbers you use and the arithmetic for any calculation. "
    "Give the final answer in the first sentence. "
    "If the documents do not contain enough information, reply exactly: "
    "\"Not enough information in the provided filings.\""
)
REFUSAL_TEXT = "Not enough information in the provided filings."


@dataclass
class Citation:
    chunk_id: str
    doc_id: str
    page_num: int
    cited_text: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    model: str = MODEL
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    stop_reason: str | None = None

    @property
    def cost_usd(self) -> float:
        pin, pout = PRICES.get(self.model, (0.0, 0.0))
        return (self.input_tokens * pin + self.output_tokens * pout) / 1e6

    @property
    def refused(self) -> bool:
        return REFUSAL_TEXT.lower().rstrip(".") in self.text.lower()


def build_content(question: str, hits: list[Hit]) -> list[dict]:
    blocks = [{
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": h.text},
        "title": h.header,
        "context": f"chunk_id={h.chunk_id}",
        "citations": {"enabled": True},
    } for h in hits]
    blocks.append({"type": "text", "text": question})
    return blocks


def _client():
    import anthropic
    return anthropic.Anthropic()


def answer(question: str, hits: list[Hit], client=None, model: str = MODEL) -> Answer:
    client = client or _client()
    t0 = time.perf_counter()
    resp = client.beta.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM,
        output_config={"effort": EFFORT},
        betas=["server-side-fallback-2026-07-01"],   # if a request is declined, the server retries on a fallback model
        fallbacks="default",
        messages=[{"role": "user", "content": build_content(question, hits)}],
    )
    latency = time.perf_counter() - t0
    if resp.stop_reason == "refusal":
        return Answer("[declined by model safety system]", model=resp.model, stop_reason=resp.stop_reason, latency_s=latency)
    texts, cites = [], []
    for block in resp.content:
        if block.type != "text":
            continue
        texts.append(block.text)
        for c in getattr(block, "citations", None) or []:
            h = hits[c.document_index]
            cites.append(Citation(h.chunk_id, h.doc_id, h.page_num, c.cited_text))
    return Answer(text="".join(texts).strip(), citations=cites, model=resp.model,
                  input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
                  latency_s=latency, stop_reason=resp.stop_reason)
