"""Replay a year of real flight changes through the CDC pipeline and check the result.

The source database receives the changes hour by hour, in the order they happened.
After each hour the consumer pulls whatever is new from the change log.

Checks at the end:
  1. the warehouse's current table equals the source table, row for row
  2. point-in-time: at a few moments in the year I saved a copy of the source table;
     the history table rebuilt "as of" that moment must equal the copy
  3. history is well formed: one current row per flight, no gaps or overlaps
  4. re-sending batches the consumer already applied changes nothing

    python pipeline.py              # whole year (~1M changes)
    python pipeline.py --month 2    # one month, for a quick run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from cdc import COLS, Source, Target, sync

from flightsdata.events import build_events, load_flights, order_problems

OUT = Path(__file__).resolve().parent / "results"
# moments to compare (New York time): a blizzard evening, a holiday, Thanksgiving eve
SNAPSHOTS = ["2013-01-15 09:00", "2013-02-08 19:00", "2013-07-04 12:00", "2013-11-27 17:00"]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLS].copy()
    df["sched_dep_utc"] = pd.to_datetime(df.sched_dep_utc, utc=True).dt.strftime("%Y-%m-%dT%H:%M")
    for c in ["dep_delay", "arr_delay", "air_time"]:
        df[c] = pd.to_numeric(df[c]).astype("float64")
    for c in ["flight_id", "flight"]:
        df[c] = df[c].astype("int64")
    for c in ["carrier", "tailnum", "origin", "dest", "status"]:
        df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df.sort_values("flight_id").reset_index(drop=True)


def diff_rows(a: pd.DataFrame, b: pd.DataFrame) -> int:
    """Rows present in one frame and not the other (0 means identical)."""
    a, b = normalize(a), normalize(b)
    m = a.merge(b, how="outer", indicator=True, on=COLS)
    return int((m._merge != "both").sum())


def history_problems(t: Target) -> dict:
    q = lambda sql: int(t.db.execute(sql).fetchone()[0])  # noqa: E731
    return {
        "flights_without_one_current_row": q("""
            select count(*) from (select flight_id from flights_history group by 1
                                  having sum(is_current::int) <> 1)"""),
        "versions_with_gap_or_overlap": q("""
            select count(*) from (
              select valid_to, lead(valid_from) over (partition by flight_id order by lsn) nxt
              from flights_history) where nxt is not null and valid_to <> nxt"""),
        "versions_ending_before_they_start": q(
            "select count(*) from flights_history where valid_to < valid_from"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", type=int)
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    ev = build_events(load_flights(month=a.month))
    build_s = time.perf_counter() - t0
    snaps = {s: pd.Timestamp(s, tz="America/New_York").tz_convert("UTC") for s in SNAPSHOTS}
    snaps = {k: v for k, v in snaps.items() if ev["at"].min() < v < ev["at"].max()}

    src, tgt = Source(), Target()
    saved, src_s, cdc_s, polls, redelivered_new = {}, 0.0, 0.0, 0, 0
    hours = ev["at"].dt.floor("h")
    pending = sorted(snaps.items(), key=lambda kv: kv[1])
    for i, (hour, chunk) in enumerate(ev.groupby(hours, sort=True)):
        while pending and pending[0][1] <= hour:   # everything before this moment is applied
            name, ts = pending.pop(0)
            saved[name] = (ts, src.snapshot())
        t1 = time.perf_counter()
        src.apply(chunk)
        t2 = time.perf_counter()
        sync(src, tgt)
        t3 = time.perf_counter()
        src_s += t2 - t1
        cdc_s += t3 - t2
        polls += 1
        if i % 500 == 250:  # pretend the consumer crashed and re-reads the last 2,000 changes
            redelivered_new += tgt.apply(src.read_changes(max(0, tgt.checkpoint - 2000), 2000))

    checks = {
        "changes_out_of_lifecycle_order": order_problems(ev),
        "current_vs_source_row_diffs": diff_rows(tgt.current(), src.snapshot()),
        "point_in_time_row_diffs": {name: diff_rows(tgt.as_of((ts - pd.Timedelta(seconds=1)).isoformat()), snap)
                                    for name, (ts, snap) in saved.items()},
        **history_problems(tgt),
        "redelivered_changes_applied_again": redelivered_new,
    }
    n_changes = int(src.db.execute("select count(*) from change_log").fetchone()[0])
    st = tgt.current().status.value_counts().to_dict()
    report = {
        "scope": f"month {a.month}" if a.month else "2013 full year",
        "flights": int(ev.flight_id.nunique()),
        "changes": n_changes,
        "changes_by_status": ev.status.value_counts().to_dict(),
        "history_rows": int(tgt.db.execute("select count(*) from flights_history").fetchone()[0]),
        "final_status": st,
        "polls": polls,
        "seconds": {"build_events": round(build_s, 1), "source_writes": round(src_s, 1),
                    "cdc_apply": round(cdc_s, 1)},
        "cdc_changes_per_second": round(n_changes / cdc_s),
        "checks": checks,
        "passed": all(v == 0 for k, v in checks.items() if not isinstance(v, dict))
                  and all(v == 0 for v in checks["point_in_time_row_diffs"].values()),
    }
    OUT.mkdir(exist_ok=True)
    name = f"cdc_month{a.month}.json" if a.month else "cdc_full_year.json"
    (OUT / name).write_text(json.dumps(report, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
