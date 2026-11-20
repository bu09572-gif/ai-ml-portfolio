"""The work behind each task of the daily flights pipeline, kept free of orchestration
code so it can be tested on its own.

One run handles one day (the run's logical date):

  extract     copy that day's flights and weather from the raw files into staging
  check       compute the day's quality metrics and score them against earlier days
  publish     load the day into the warehouse, replacing any earlier load of that day
  quarantine  keep the day out of the warehouse and record why

Everything is keyed by day, so re-running a day gives the same result.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from flightsdata.dq import KIND, RAW, Z_ALERT, daily_metrics, prepare, score_day

ROOT = Path(__file__).resolve().parents[4]
WORK = Path(os.environ.get("PIPELINE_DIR", ROOT / "data" / "airflow"))
DB = WORK / "warehouse.duckdb"

DDL = """
create table if not exists daily_metrics (
    day date primary key, flights_scheduled double, cancel_rate double,
    missing_arr_delay_rate double, missing_tail_rate_flown double, median_dep_delay double,
    p90_dep_delay double, weather_rows double
);
create table if not exists day_status (
    day date primary key, status varchar, data_alerts varchar, flights integer, loaded_at timestamptz
);
"""


def _connect() -> duckdb.DuckDBPyConnection:
    WORK.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB))
    for stmt in filter(str.strip, DDL.split(";")):
        con.execute(stmt).fetchall()
    return con


def _stage(ds: str) -> Path:
    return WORK / "staging" / ds  # no "key=value" folder: DuckDB would read it as a hive partition column


def extract(ds: str, raw: Path = RAW) -> dict:
    d = pd.Timestamp(ds)
    out = _stage(ds)
    out.mkdir(parents=True, exist_ok=True)
    where = f"year = {d.year} and month = {d.month} and day = {d.day}"
    counts = {}
    for name in ["flights", "weather"]:
        path = out / f"{name}.parquet"
        duckdb.sql(f"select * from read_csv_auto('{raw}/{name}.csv', nullstr='NA') where {where}").write_parquet(str(path))
        counts[name] = duckdb.sql(f"select count(*) from '{path}'").fetchall()[0][0]
    if counts["flights"] == 0:
        raise ValueError(f"no flights in the source for {ds}")
    return counts


def check(ds: str) -> dict:
    """Metrics for the day, saved, then scored against the days before it."""
    st = _stage(ds)
    today = daily_metrics(flights=duckdb.sql(f"select * from '{st / 'flights.parquet'}'").df(),
                          weather=duckdb.sql(f"select * from '{st / 'weather.parquet'}'").df())
    with _connect() as con:
        con.execute("delete from daily_metrics where day = ?", [ds]).fetchall()
        con.register("today", today)
        con.execute("insert into daily_metrics by name select * from today").fetchall()
        hist = con.execute("select * from daily_metrics where day <= ? order by day", [ds]).df()
    hist = prepare(hist)
    scores = score_day(hist, len(hist) - 1)
    data_alerts = [{"metric": m, "value": round(float(v), 4), "z": round(float(z), 2)}
                   for _, m, v, z in scores if KIND[m] == "data" and pd.notna(z) and abs(z) >= Z_ALERT]
    ops_alerts = [m for _, m, _, z in scores if KIND[m] == "ops" and pd.notna(z) and abs(z) >= Z_ALERT]
    return {"day": ds, "data_alerts": data_alerts, "ops_alerts": ops_alerts}


def _record(con, ds: str, status: str, alerts: list, flights: int) -> None:
    con.execute("delete from day_status where day = ?", [ds]).fetchall()
    con.execute("insert into day_status values (?, ?, ?, ?, ?)",
                [ds, status, json.dumps(alerts), flights, datetime.now(timezone.utc)]).fetchall()


def publish(ds: str) -> int:
    path = _stage(ds) / "flights.parquet"
    with _connect() as con:
        con.execute(f"create table if not exists flights as select * from '{path}' limit 0").fetchall()
        con.execute("begin").fetchall()
        con.execute("delete from flights where make_date(year, month, day) = ?", [ds]).fetchall()
        con.execute(f"insert into flights by name select * from '{path}'").fetchall()
        n = con.execute("select count(*) from flights where make_date(year, month, day) = ?", [ds]).fetchall()[0][0]
        _record(con, ds, "published", [], n)
        con.execute("commit").fetchall()
    return n


def quarantine(ds: str, alerts: list) -> None:
    with _connect() as con:
        con.execute("begin").fetchall()
        if con.execute("select count(*) from information_schema.tables where table_name = 'flights'").fetchall()[0][0]:
            con.execute("delete from flights where make_date(year, month, day) = ?", [ds]).fetchall()
        _record(con, ds, "quarantined", alerts, 0)
        con.execute("commit").fetchall()
