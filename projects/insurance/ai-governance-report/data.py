"""The texts of the state AI rules for insurers, mirrored into this repo's data-v2 release.

  naic_bulletin   NAIC Model Bulletin "Use of Artificial Intelligence Systems by Insurers", the
                  October 13, 2023 committee draft (NAIC's adopted PDF link is broken; see README)
  adoption_map    NAIC's map of states that adopted the bulletin, status November 13, 2026
  new_york        NY DFS Insurance Circular Letter No. 7 (2024): AI and external data in underwriting and pricing
  colorado        Colorado Amended Regulation 10-1-1 (3 CCR 702-10): governance and risk management for
                  life, private passenger auto and health insurers' use of external data, algorithms and models
"""
from __future__ import annotations

import hashlib
import html
import io
import os
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[3]
RAW = Path(os.environ.get("PUBLIC_V2_DIR", ROOT / "data" / "raw" / "public_v2"))
BASE = "https://github.com/bu09572-gif/ai-ml-portfolio/releases/download/data-v2/"
FILES = {
    "naic_bulletin": ("naic_ai_model_bulletin.pdf", "588e4ff5db54536f78dd1116270acf05d88a198dfce25a4d9f06963e20a0b787"),
    "adoption_map": ("naic_ai_bulletin_adoption_map.pdf", "0315fd25f9dda54befad84cad9008e5a284f91f084b6d3577581d0d6904770d6"),
    "new_york": ("nydfs_circular_letter_2024_07.html", "1eed7ab646f400c728f4265331e6915bcb1e052fe7a860c903fb4d8b36ef04ee"),
    "colorado": ("colorado_reg_10_1_1_amended.pdf", "0e2edba3467bc90e0c8eaca02b51b0e0c97b580d28a662f9072ae25b08c6cc0e"),
}
RULES = ["naic_bulletin", "new_york", "colorado"]


def _bytes(key: str) -> bytes:
    name, sha = FILES[key]
    p = RAW / name
    if not p.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        p.write_bytes(requests.get(BASE + name, timeout=300).content)
    b = p.read_bytes()
    if hashlib.sha256(b).hexdigest() != sha:
        raise RuntimeError(f"{name} does not match the mirrored copy")
    return b


def _pdf_pages(b: bytes) -> list[str]:
    import pypdfium2
    doc = pypdfium2.PdfDocument(io.BytesIO(b))
    return [doc[i].get_textpage().get_text_range().replace("\r", "") for i in range(len(doc))]


def _html_text(b: bytes) -> str:
    h = b.decode("utf-8", errors="ignore")
    m = re.search(r"<main.*?</main>", h, re.S | re.I)
    h = m.group(0) if m else h
    h = re.sub(r"<(script|style|nav|header|footer)\b.*?</\1>", " ", h, flags=re.S | re.I)
    h = re.sub(r"<(br|/p|/li|/h\d|/div)[^>]*>", "\n", h, flags=re.I)
    t = html.unescape(re.sub(r"<[^>]+>", " ", h))
    return re.sub(r"\n\s*\n+", "\n", re.sub(r"[ \t]+", " ", t)).strip()


def text(key: str) -> list[str]:
    """The document as a list of pages (an HTML page counts as one page)."""
    b = _bytes(key)
    if FILES[key][0].endswith(".pdf"):
        pages = _pdf_pages(b)
        if key == "naic_bulletin":   # drop the draft's running header
            pages = [re.sub(r"10\.13\.2023\s*\nCOMMITTEE EXPOSURE DRAFT\s*\n\d+\s*\n(DRAFT\s*\n)?", "", p) for p in pages]
        return pages
    t = _html_text(b)
    start = t.find("The New York State Department of Financial Services")
    return [t[start:] if start > 0 else t]


def adoption() -> dict:
    """States that adopted the NAIC bulletin, and states with their own insurance AI rules."""
    t = _pdf_pages(_bytes("adoption_map"))[1]
    adopted_part, own_part = t.split("Insurance Specific Regulation/Guidance")
    states = lambda s: re.findall(r"•\s*([A-Z][A-Za-z .]+?):", s)  # noqa: E731
    return {"status_date": re.search(r"Status as of ([A-Za-z]+ \d+, \d{4})", _pdf_pages(_bytes("adoption_map"))[0]).group(1),
            "adopted": states(adopted_part.split("Adopted States", 1)[1]), "own_rules": states(own_part)}
