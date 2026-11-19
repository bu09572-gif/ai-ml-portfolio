"""Daily data-quality monitor for the flights feed.

For every day it computes a handful of metrics and compares each one with that
metric's recent history:

  volume metrics (flights scheduled)  vs. the same weekday over the previous 8 weeks
  rate metrics (cancel rate, missing arrival delay, ...)  vs. the previous 28 days

The comparison is a robust z-score: (value - median) / (1.4826 * MAD), so one bad
day doesn't distort the baseline for the next. |z| >= 4 raises an alert.

    python monitor.py                      # writes results/alerts.csv, results/daily_metrics.csv, results/summary.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = Path(os.environ.get("RAW_DIR", HERE.parents[2] / "data" / "raw" / "nycflights13"))
OUT = HERE / "results"
Z_ALERT = 4.0

DAILY_SQL = """
with f as (
    select make_date(year, month, day) as d, *
    from read_csv_auto('{raw}/flights.csv', nullstr='NA')
),
w as (
    -- local date, same as the flights file (time_hour is UTC)
    select make_date(year, month, day) as d, count(*) as weather_rows
    from read_csv_auto('{raw}/weather.csv', nullstr='NA') group by 1
)
select
    f.d                                                as day,
    count(*)                                           as flights_scheduled,
    avg((dep_time is null)::int)                       as cancel_rate,
    avg((dep_time is not null and arr_delay is null)::int) as missing_arr_delay_rate,
    avg(case when dep_time is not null then (tailnum is null)::int end) as missing_tail_rate_flown,
    median(dep_delay)                                  as median_dep_delay,
    quantile_cont(dep_delay, 0.9)                      as p90_dep_delay,
    any_value(w.weather_rows)                          as weather_rows
from f left join w on w.d = f.d
group by f.d order by f.d
"""

# Two kinds of checks. "data" checks catch problems with the feed itself (missing rows or
# fields); "ops" checks catch real-world disruptions (storms, holidays) that a downstream
# model or dashboard should know about.
VOLUME = {"flights_scheduled": "ops", "weather_rows": "data"}
RATES = {"cancel_rate": "ops", "median_dep_delay": "ops", "p90_dep_delay": "ops",
         "missing_arr_delay_rate": "data", "missing_tail_rate_flown": "data"}
KIND = {**VOLUME, **RATES}
# Smallest spread treated as meaningful, per metric. Without it, a metric that sits at
# exactly the same value for weeks (median delay = -2 min) alerts on a 1-minute change.
MIN_SCALE = {"flights_scheduled": 10, "weather_rows": 1, "cancel_rate": 0.01, "median_dep_delay": 2,
             "p90_dep_delay": 10, "missing_arr_delay_rate": 0.003, "missing_tail_rate_flown": 0.002}


def robust_z(x: float, base: pd.Series, min_scale: float = 0.0) -> float:
    base = pd.to_numeric(base, errors="coerce").dropna()
    if pd.isna(x) or len(base) < 4:
        return np.nan
    med = base.median()
    mad = (base - med).abs().median() * 1.4826
    mad = max(mad, min_scale, 1e-9)
    return float((x - med) / mad)


def daily_metrics(raw: Path = RAW, flights: pd.DataFrame | None = None,
                  weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily metrics from the CSVs, or from in-memory frames (used by the fault tests)."""
    sql = DAILY_SQL.format(raw=raw)
    con = duckdb.connect()
    if flights is not None:
        con.register("flights_df", flights)
        sql = sql.replace(f"read_csv_auto('{raw}/flights.csv', nullstr='NA')", "flights_df")
    if weather is not None:
        con.register("weather_df", weather)
        sql = sql.replace(f"read_csv_auto('{raw}/weather.csv', nullstr='NA')", "weather_df")
    df = con.sql(sql).df()
    num = df.columns.drop("day")
    df[num] = df[num].astype("float64")
    return df


def score_day(df: pd.DataFrame, i: int, row: pd.Series | None = None) -> list[tuple]:
    """z-scores for day i. `row` replaces that day's values (used to test injected faults)."""
    r = df.iloc[i] if row is None else row
    out = []
    for m in VOLUME:
        hist = df[(df.index < i) & (df.weekday == df.weekday.iloc[i])].tail(8)[m]
        out.append((df.day.iloc[i], m, r[m], robust_z(r[m], hist, MIN_SCALE[m])))
    for m in RATES:
        hist = df.iloc[max(0, i - 28):i][m]
        out.append((df.day.iloc[i], m, r[m], robust_z(r[m], hist, MIN_SCALE[m])))
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("day").reset_index(drop=True)
    df["weekday"] = pd.to_datetime(df.day).dt.day_of_week
    return df


def score(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare(df)
    rows = [t for i in range(len(df)) for t in score_day(df, i)]
    out = pd.DataFrame(rows, columns=["day", "metric", "value", "z"])
    out["kind"] = out.metric.map(KIND)
    return out


def main():
    OUT.mkdir(exist_ok=True)
    daily = daily_metrics()
    scores = score(daily)
    alerts = scores[scores.z.abs() >= Z_ALERT].copy()
    alerts["direction"] = np.where(alerts.z > 0, "high", "low")
    daily.to_csv(OUT / "daily_metrics.csv", index=False)
    alerts.assign(value=alerts.value.round(4), z=alerts.z.round(2)).to_csv(OUT / "alerts.csv", index=False)
    summary = {
        "days": len(daily),
        "metrics_per_day": len(KIND), "z_threshold": Z_ALERT,
        "checks": int(scores.z.notna().sum()),
        "alerts": len(alerts),
        "days_with_alerts": int(alerts.day.nunique()),
        "alerts_by_metric": alerts.metric.value_counts().to_dict(),
        "days_with_alerts_by_kind": alerts.groupby("kind").day.nunique().to_dict(),
        "top_days": [
            {"day": str(pd.Timestamp(d).date()), "max_abs_z": round(float(g.z.abs().max()), 1),
             "metrics": sorted(g.metric.unique().tolist())}
            for d, g in sorted(alerts.groupby("day"), key=lambda kv: -kv[1].z.abs().max())[:12]],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
