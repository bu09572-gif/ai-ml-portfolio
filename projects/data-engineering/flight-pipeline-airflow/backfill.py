"""Run the daily DAG for a range of days, in order, the way a backfill would, then
compare what it quarantined with what the standalone data quality monitor flags.

Each day runs through `dag.test()`, which executes the real tasks in this process
without needing a scheduler. Use `airflow backfill create` on a running deployment.

    python backfill.py 2013-01-01 2013-12-31
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb
import flights_tasks as ft
import pandas as pd
from airflow.dag_processing.dagbag import DagBag

from flightsdata.dq import KIND, Z_ALERT, daily_metrics, score

OUT = Path(__file__).resolve().parent / "results"


def main(start: str, end: str) -> int:
    dag = DagBag(dag_folder=str(Path(__file__).resolve().parent / "dags")).get_dag("flights_daily")
    days = pd.date_range(start, end, freq="D", tz="UTC")
    failed, t0 = [], time.perf_counter()
    for d in days:
        run = dag.test(logical_date=d.to_pydatetime())
        if run.state != "success":
            failed.append(str(d.date()))
    secs = time.perf_counter() - t0

    con = duckdb.connect(str(ft.DB), read_only=True)
    status = con.execute("select day, status, data_alerts, flights from day_status order by day").df()
    published_rows = con.execute("select count(*) from flights").fetchone()[0]
    con.close()

    # the standalone monitor, run over the same days
    scores = score(daily_metrics())
    scores = scores[(scores.day >= days[0].tz_localize(None)) & (scores.day <= days[-1].tz_localize(None))]
    flagged = scores[(scores.kind == "data") & (scores.z.abs() >= Z_ALERT)]
    monitor_days = sorted(str(pd.Timestamp(x).date()) for x in flagged.day.unique())
    quarantined = sorted(str(pd.Timestamp(x).date()) for x in status[status.status == "quarantined"].day)

    report = {
        "days": len(days), "failed_runs": failed, "seconds": round(secs, 1),
        "seconds_per_run": round(secs / len(days), 2),
        "published_days": int((status.status == "published").sum()),
        "quarantined_days": quarantined,
        "published_flight_rows": int(published_rows),
        "monitor_data_alert_days": monitor_days,
        "matches_monitor": quarantined == monitor_days,
        "data_metrics": sorted(m for m, k in KIND.items() if k == "data"),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"backfill_{start}_{end}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if not failed and report["matches_monitor"] else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
