"""Unit tests: a small SQLite file and a scripted model, so no download or model is needed."""
import sqlite3

import pytest
import run_eval
import sqlagent

from docqa import llm


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "school.sqlite"
    con = sqlite3.connect(p)
    con.executescript("""
        create table school (id integer primary key, name text, county text);
        create table score (school_id integer references school(id), "Avg Math" real);
        insert into school values (1, 'Lincoln High', 'Alameda'), (2, 'Grant High', 'Fresno');
        insert into score values (1, 520.5), (2, 480.0);
        create table "order" (order_id integer primary key, amount real);
        insert into "order" values (1, 9.5);
    """)
    con.commit()
    con.close()
    sqlagent.schema.cache_clear()
    return p


@pytest.fixture
def scripted(monkeypatch):
    """Replace the model with replies given in order; record the prompts it was sent."""
    prompts, replies = [], []

    def chat(system, user, **kw):
        prompts.append(user)
        return llm.Completion(replies.pop(0), 10, 5, 0.0, "scripted", "test")

    monkeypatch.setattr(llm, "chat", chat)
    return prompts, replies


def test_execute_is_read_only(db):
    assert sqlagent.execute(db, "select count(*) from school") == [(2,)]
    for bad in ["delete from school", "drop table school", "insert into school values (3, 'x', 'y')",
                "attach database ':memory:' as x", "pragma writable_schema = 1"]:
        with pytest.raises(sqlite3.Error):
            sqlagent.execute(db, bad)
    assert sqlagent.execute(db, "select count(*) from school") == [(2,)]


def test_execute_allows_cte_and_functions(db):
    sql = "with recursive n(i) as (select 1 union all select i + 1 from n where i < 3) select sum(i) from n"
    assert sqlagent.execute(db, sql) == [(6,)]
    assert sqlagent.execute(db, "select upper(name) from school where id = 1") == [("LINCOLN HIGH",)]


def test_execute_time_limit(db):
    slow = "with recursive n(i) as (select 1 union all select i + 1 from n) select count(*) from n"
    with pytest.raises(sqlite3.OperationalError):
        sqlagent.execute(db, slow, time_limit=0.2)


def test_schema_shows_keys_quotes_and_examples(db):
    s = sqlagent.schema(db)
    assert "id INTEGER PRIMARY KEY" in s
    assert '"Avg Math" REAL' in s
    assert "FOREIGN KEY (school_id) REFERENCES school(id)" in s
    assert "'Lincoln High'" in s
    assert 'CREATE TABLE "order"' in s          # a keyword as a table name must be quoted


def test_extract_sql():
    assert sqlagent.extract_sql("Here:\n```sql\nSELECT 1;\n```\nDone") == "SELECT 1"
    assert sqlagent.extract_sql("select 2;") == "select 2"


def test_repair_loop_sends_error_back(db, scripted):
    prompts, replies = scripted
    replies += ["```sql\nselect avg_math from score\n```", '```sql\nselect "Avg Math" from score where school_id = 1\n```']
    r = sqlagent.answer("Math score of Lincoln High?", "", db)
    assert len(r.attempts) == 2
    assert "no such column" in r.attempts[0].error
    assert "no such column" in prompts[1] and "select avg_math" in prompts[1]
    assert r.final.rows == [(520.5,)] and r.final.error is None


def test_repairs_stop_after_limit(db, scripted):
    _, replies = scripted
    replies += ["select nope from school"] * 3
    r = sqlagent.answer("q", "", db)
    assert len(r.attempts) == 1 + sqlagent.MAX_REPAIRS and r.final.error


def test_hint_only_when_asked(db, scripted):
    prompts, replies = scripted
    replies += ["select 1", "select 1"]
    sqlagent.answer("q", "Alameda is a county", db, use_evidence=True)
    sqlagent.answer("q", "Alameda is a county", db, use_evidence=False)
    assert "Hint: Alameda" in prompts[0] and "Hint" not in prompts[1]


def test_same_result_ignores_order_only():
    assert sqlagent.same_result([(1,), (2,)], [(2,), (1,)])
    assert not sqlagent.same_result([(1,)], [(1,), (2,)])
    assert not sqlagent.same_result([(1, "a")], [(1,)])
    assert not sqlagent.same_result(None, [])


def test_score_one_and_summary(db, scripted, monkeypatch):
    _, replies = scripted
    monkeypatch.setattr(run_eval.data, "db_path", lambda _: db)
    q = {"question_id": 7, "db_id": "school", "difficulty": "simple", "evidence": "",
         "question": "Which schools are in Alameda?", "SQL": "select name from school where county = 'Alameda'"}
    # with hint: wrong first, fixed by repair. no hint: right first time
    replies += ["select name frm school", "select name from school where county = 'Alameda'",
                "select name from school where county = 'Alameda'"]
    row = run_eval.score_one(q)
    assert row["with_hint"]["tries"] == 2 and not row["with_hint"]["first_correct"] and row["with_hint"]["final_correct"]
    assert row["no_hint"]["first_correct"] and row["gold_rows"] == 1
    bad = dict(q, question_id=8, SQL="select missing from school")
    replies += ["select 1", "select 1"]
    row2 = run_eval.score_one(bad)
    assert row2["gold_error"]
    import pandas as pd
    s = run_eval.summarize(pd.DataFrame([row, row2]))
    assert s["scored"] == 1 and s["reference_sql_failed"] == 1
    assert s["configs"]["with_hint"]["execution_accuracy"] == 1.0
    assert s["configs"]["with_hint"]["execution_accuracy_first_try"] == 0.0
    assert s["configs"]["with_hint"]["fixed_by_repair"] == 1
