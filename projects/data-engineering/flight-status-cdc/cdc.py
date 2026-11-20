"""Change data capture from an operational SQLite database into a DuckDB warehouse.

Source (SQLite, stands in for the airline's ops database)
  flights      one row per flight, updated in place as the flight progresses
  change_log   written by triggers on every insert / update / delete; lsn is the order

Target (DuckDB)
  flights_current   same rows as the source, kept in sync
  flights_history   SCD type 2: every version of every flight with valid_from / valid_to
  cdc_checkpoint    last lsn applied; updated in the same transaction as the data

Because the checkpoint commits together with the data and the consumer skips any lsn it
has already applied, delivering a batch twice (after a crash, say) changes nothing.
"""
from __future__ import annotations

import sqlite3

import duckdb
import pandas as pd

from flightsdata.events import COLS

DATA_COLS = COLS[1:]  # everything but the key

SOURCE_DDL = f"""
create table if not exists flights (
    flight_id integer primary key, carrier text, flight integer, tailnum text, origin text,
    dest text, sched_dep_utc text, status text, dep_delay real, arr_delay real, air_time real,
    updated_at text
);
create table if not exists change_log (
    lsn integer primary key autoincrement, op text, changed_at text,
    {", ".join(f"{c} {'integer' if c == 'flight_id' else ''}" for c in COLS)}
);
create trigger if not exists flights_ins after insert on flights begin
    insert into change_log (op, changed_at, {", ".join(COLS)})
    values ('i', new.updated_at, {", ".join("new." + c for c in COLS)});
end;
create trigger if not exists flights_upd after update on flights begin
    insert into change_log (op, changed_at, {", ".join(COLS)})
    values ('u', new.updated_at, {", ".join("new." + c for c in COLS)});
end;
create trigger if not exists flights_del after delete on flights begin
    insert into change_log (op, changed_at, {", ".join(COLS)})
    values ('d', strftime('%Y-%m-%dT%H:%M:%SZ', 'now'), {", ".join("old." + c for c in COLS)});
end;
"""

TARGET_DDL = f"""
create table if not exists flights_current (
    flight_id integer primary key, carrier varchar, flight integer, tailnum varchar, origin varchar,
    dest varchar, sched_dep_utc timestamptz, status varchar, dep_delay double, arr_delay double,
    air_time double, updated_at timestamptz, lsn bigint
);
create table if not exists flights_history (
    flight_id integer, {", ".join(f"{c} {t}" for c, t in zip(DATA_COLS, [
        "varchar", "integer", "varchar", "varchar", "varchar", "timestamptz", "varchar",
        "double", "double", "double"]))},
    valid_from timestamptz, valid_to timestamptz, is_current boolean, is_deleted boolean, lsn bigint
);
create table if not exists cdc_checkpoint (id integer primary key, lsn bigint);
insert into cdc_checkpoint values (1, 0) on conflict do nothing;
"""


def _ts(s: pd.Series) -> pd.Series:
    return s.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class Source:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path)
        self.db.executescript(SOURCE_DDL)

    def apply(self, ev: pd.DataFrame) -> None:
        """Apply application writes in order. The triggers produce the change log."""
        if ev.empty:
            return
        ev = ev.assign(sched_dep_utc=_ts(ev.sched_dep_utc), updated_at=_ts(ev["at"]))
        ev = ev.astype(object).where(ev.notna(), None)
        with self.db:
            for op, chunk in _runs(ev):
                if op == "i":
                    self.db.executemany(
                        f"insert into flights ({', '.join(COLS)}, updated_at) values ({', '.join('?' * (len(COLS) + 1))})",
                        chunk[[*COLS, "updated_at"]].itertuples(index=False, name=None))
                elif op == "u":
                    self.db.executemany(
                        "update flights set status=?, dep_delay=?, arr_delay=?, air_time=?, updated_at=? where flight_id=?",
                        chunk[["status", "dep_delay", "arr_delay", "air_time", "updated_at", "flight_id"]]
                        .itertuples(index=False, name=None))
                else:
                    self.db.executemany("delete from flights where flight_id=?",
                                        chunk[["flight_id"]].itertuples(index=False, name=None))

    def read_changes(self, after_lsn: int, limit: int = 50_000) -> pd.DataFrame:
        return pd.read_sql_query(
            f"select lsn, op, changed_at, {', '.join(COLS)} from change_log where lsn > ? order by lsn limit ?",
            self.db, params=(after_lsn, limit))

    def snapshot(self) -> pd.DataFrame:
        return pd.read_sql_query(f"select {', '.join(COLS)} from flights order by flight_id", self.db)


def _runs(ev: pd.DataFrame):
    """Split into consecutive runs of the same op, keeping order."""
    run_id = (ev.op != ev.op.shift()).cumsum()
    for _, chunk in ev.groupby(run_id, sort=False):
        yield chunk.op.iloc[0], chunk


class Target:
    def __init__(self, path: str = ":memory:"):
        self.db = duckdb.connect(path)
        self._run("set TimeZone='UTC'")
        for stmt in filter(str.strip, TARGET_DDL.split(";")):
            self._run(stmt)

    def _run(self, sql: str, params: list | None = None) -> list:
        """Execute and drain the result, so the connection is free for the next statement."""
        return self.db.execute(sql, params or []).fetchall()

    @property
    def checkpoint(self) -> int:
        return self._run("select lsn from cdc_checkpoint")[0][0]

    def apply(self, changes: pd.DataFrame) -> int:
        """Apply a batch of changes. Returns how many were new (not seen before)."""
        if changes.empty:
            return 0
        db = self.db
        db.register("batch_raw", changes)
        self._run("begin")
        try:
            self._run("""
                create or replace temp table batch as
                select * replace (cast(changed_at as timestamptz) as changed_at,
                                  cast(sched_dep_utc as timestamptz) as sched_dep_utc)
                from batch_raw where lsn > (select lsn from cdc_checkpoint)
            """)
            n = self._run("select count(*) from batch")[0][0]
            if n:
                cols = ", ".join(DATA_COLS)
                # close the open version of every flight touched in this batch
                self._run("""
                    update flights_history h set valid_to = b.first_at, is_current = false
                    from (select flight_id, min(changed_at) first_at from batch group by 1) b
                    where h.flight_id = b.flight_id and h.is_current
                """)
                # add every new version; within the batch, a version ends where the next starts
                self._run(f"""
                    insert into flights_history
                    select flight_id, {cols}, changed_at,
                           lead(changed_at) over w,
                           lead(lsn) over w is null and op <> 'd',
                           op = 'd', lsn
                    from batch window w as (partition by flight_id order by lsn)
                """)
                self._run("delete from flights_current where flight_id in (select flight_id from batch)")
                self._run(f"""
                    insert into flights_current
                    select flight_id, {cols}, changed_at, lsn from batch
                    where op <> 'd'
                    qualify row_number() over (partition by flight_id order by lsn desc) = 1
                """)
                self._run("update cdc_checkpoint set lsn = (select max(lsn) from batch)")
            self._run("commit")
        except Exception:
            self._run("rollback")
            raise
        finally:
            db.unregister("batch_raw")
        return n

    def as_of(self, ts: str) -> pd.DataFrame:
        """Every flight as it looked at time ts, rebuilt from history."""
        return self.db.execute(f"""
            select flight_id, {", ".join(DATA_COLS)} from flights_history
            where valid_from <= cast(? as timestamptz)
              and (valid_to is null or valid_to > cast(? as timestamptz)) and not is_deleted
            order by flight_id
        """, [ts, ts]).df()

    def current(self) -> pd.DataFrame:
        return self.db.execute(f"select {', '.join(COLS)} from flights_current order by flight_id").df()


def sync(source: Source, target: Target, batch: int = 50_000) -> int:
    """Pull everything new from the source change log. Returns changes applied."""
    total = 0
    while True:
        ch = source.read_changes(target.checkpoint, batch)
        if ch.empty:
            return total
        total += target.apply(ch)
