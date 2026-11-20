import pandas as pd
import pytest
from cdc import Source, Target, sync

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


def ev(op, at, fid, status, dep=None, arr=None):
    return {"op": op, "at": T(at), "flight_id": fid, "carrier": "UA", "flight": 1, "tailnum": "N1",
            "origin": "EWR", "dest": "ORD", "sched_dep_utc": T("2013-01-02 12:00"), "status": status,
            "dep_delay": dep, "arr_delay": arr, "air_time": None}


def run(events):
    s, t = Source(), Target()
    s.apply(pd.DataFrame(events))
    sync(s, t)
    return s, t


def test_history_has_every_version_and_current_matches():
    s, t = run([ev("i", "2013-01-01 05:00", 1, "scheduled"),
                ev("u", "2013-01-02 12:10", 1, "departed", dep=10),
                ev("u", "2013-01-02 14:30", 1, "arrived", dep=10, arr=5)])
    h = t.db.execute("select status, valid_to is null, is_current from flights_history order by lsn").fetchall()
    assert h == [("scheduled", False, False), ("departed", False, False), ("arrived", True, True)]
    assert t.current().status.tolist() == ["arrived"]
    assert t.as_of("2013-01-02 13:00").status.tolist() == ["departed"]


def test_versions_split_across_batches_line_up():
    s, t = Source(), Target()
    s.apply(pd.DataFrame([ev("i", "2013-01-01 05:00", 1, "scheduled")]))
    sync(s, t)
    s.apply(pd.DataFrame([ev("u", "2013-01-02 12:00", 1, "cancelled")]))
    sync(s, t)
    rows = t.db.execute("select valid_from, valid_to from flights_history order by lsn").fetchall()
    assert rows[0][1] == rows[1][0]


def test_delete_removes_from_current_but_keeps_history():
    s, t = run([ev("i", "2013-01-01 05:00", 1, "scheduled"), ev("i", "2013-01-01 05:00", 2, "scheduled")])
    s.db.execute("delete from flights where flight_id = 2")
    s.db.commit()
    sync(s, t)
    assert t.current().flight_id.tolist() == [1]
    assert t.db.execute("select count(*) from flights_history where flight_id = 2").fetchone()[0] == 2


def test_redelivery_is_ignored():
    s, t = run([ev("i", "2013-01-01 05:00", 1, "scheduled"), ev("u", "2013-01-02 12:00", 1, "cancelled")])
    assert t.apply(s.read_changes(0)) == 0
    assert t.db.execute("select count(*) from flights_history").fetchone()[0] == 2


def test_crash_mid_batch_rolls_back_then_retry_succeeds(monkeypatch):
    s = Source()
    s.apply(pd.DataFrame([ev("i", "2013-01-01 05:00", 1, "scheduled")]))
    t = Target()
    real = t._run

    def flaky(sql, params=None):
        if sql.lstrip().startswith("delete from flights_current"):
            raise RuntimeError("consumer died")
        return real(sql, params)

    monkeypatch.setattr(t, "_run", flaky)
    with pytest.raises(RuntimeError):
        sync(s, t)
    monkeypatch.setattr(t, "_run", real)
    assert t.checkpoint == 0
    assert t.db.execute("select count(*) from flights_history").fetchone()[0] == 0
    sync(s, t)
    assert t.checkpoint == 1 and len(t.current()) == 1
