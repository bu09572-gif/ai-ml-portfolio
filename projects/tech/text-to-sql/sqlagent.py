"""Text-to-SQL with a small free model, safe execution, and self-correction.

  1. schema   each table as a CREATE TABLE line with column types, primary and foreign keys,
              and a few example values per text column (so the model sees 'M' vs 'Male')
  2. generate the model writes one SQLite SELECT, given the schema, the question and BIRD's
              evidence hint (domain knowledge such as "eligible free rate = Free Meal Count / Enrollment")
  3. execute  read-only: the file is opened with mode=ro and an SQLite authorizer denies
              anything but reads, a 30 s time limit, at most 10,000 rows returned
  4. repair   if SQLite raises an error, the error goes back to the model (up to 2 more tries)
"""
from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from docqa import llm

MAX_ROWS = 10_000
TIME_LIMIT_S = 30
MAX_REPAIRS = 2
SAMPLES = 3

SYSTEM = """You write SQLite queries. Reply with one SQL query in a ```sql code block and nothing else.
Use only tables and columns from the schema. Quote column names that contain spaces or symbols
with double quotes. Return only the columns the question asks for."""

READ_OK = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
SCHEMA_CHARS = 24_000      # about 6k tokens, so the prompt fits the 8k context window


def _authorizer(action, *_):
    return sqlite3.SQLITE_OK if action in READ_OK else sqlite3.SQLITE_DENY


def connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return con


def execute(path: Path, sql: str, time_limit: float = TIME_LIMIT_S) -> list[tuple]:
    """Run one read-only query with a time limit. Raises sqlite3.Error on failure."""
    con = connect(path)
    con.set_authorizer(_authorizer)
    deadline = time.monotonic() + time_limit
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = con.execute(sql)
        return cur.fetchmany(MAX_ROWS)
    finally:
        con.close()


def _q(name: str) -> str:
    return f'"{name}"' if not re.fullmatch(r"[A-Za-z_]\w*", name) else name


@lru_cache(maxsize=32)
def schema(path: Path, samples: int = SAMPLES) -> str:
    con = connect(path)
    lines = []
    tables = [r[0] for r in con.execute("select name from sqlite_master where type='table' and name not like 'sqlite_%'")]
    for t in tables:
        cols = con.execute(f"pragma table_info({_q(t)})").fetchall()
        fks = con.execute(f"pragma foreign_key_list({_q(t)})").fetchall()
        parts = []
        for _, name, typ, _, _, pk in cols:
            p = f"  {_q(name)} {typ or ''}".rstrip() + (" PRIMARY KEY" if pk == 1 else "")
            if samples and (typ or "").upper() in ("TEXT", ""):
                try:
                    vals = [r[0] for r in con.execute(
                        f"select distinct {_q(name)} from {_q(t)} where {_q(name)} is not null limit {samples}")]
                    vals = [v for v in vals if isinstance(v, str) and len(v) <= 40]
                    if vals:
                        p += "  -- e.g. " + ", ".join(repr(v) for v in vals)
                except sqlite3.Error:
                    pass
            parts.append(p)
        for fk in fks:
            parts.append(f"  FOREIGN KEY ({_q(fk[3])}) REFERENCES {_q(fk[2])}({_q(fk[4] or fk[3])})")
        lines.append(f"CREATE TABLE {_q(t)} (\n" + ",\n".join(parts) + "\n);")
    con.close()
    out = "\n".join(lines)
    if len(out) > SCHEMA_CHARS and samples:      # big schemas: drop the example values first
        return schema(path, 0)
    return out


def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    sql = (m.group(1) if m else text).strip()
    return sql.rstrip(";").strip()


@dataclass
class Attempt:
    sql: str
    error: str | None
    rows: list[tuple] | None = None


@dataclass
class Result:
    attempts: list[Attempt] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    @property
    def final(self) -> Attempt:
        return self.attempts[-1]


def answer(question: str, evidence: str, db: Path, use_evidence: bool = True, repairs: int = MAX_REPAIRS,
           model: str | None = None) -> Result:
    t0 = time.perf_counter()
    res = Result()
    hint = f"\nHint: {evidence}" if use_evidence and evidence else ""
    prompt = f"Schema:\n{schema(db)}\n\nQuestion: {question}{hint}"
    for _ in range(1 + repairs):
        c = llm.chat(SYSTEM, prompt, model=model, max_tokens=400)
        res.input_tokens += c.input_tokens
        res.output_tokens += c.output_tokens
        sql = extract_sql(c.text)
        try:
            res.attempts.append(Attempt(sql, None, execute(db, sql)))
            break
        except sqlite3.Error as e:
            res.attempts.append(Attempt(sql, str(e)[:300]))
            prompt += f"\n\nYour query:\n```sql\n{sql}\n```\nfailed with: {str(e)[:300]}\nWrite a corrected query."
    res.latency_s = time.perf_counter() - t0
    return res


def same_result(pred: list[tuple] | None, gold: list[tuple]) -> bool:
    """BIRD's execution accuracy: the two result sets are equal (row order ignored)."""
    return pred is not None and set(pred) == set(gold)
