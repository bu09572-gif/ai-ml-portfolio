"""Daily flights pipeline: extract a day, check it, then publish it or quarantine it.

A day is published only if none of its data checks fire. Ops alerts (storms, holidays)
are recorded but don't block anything, because that data is correct.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import flights_tasks as ft
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import Asset, dag, task

WAREHOUSE = Asset("duckdb://flights-warehouse/flights")


@dag(
    schedule="@daily",
    start_date=datetime(2013, 1, 1, tzinfo=timezone.utc),
    end_date=datetime(2013, 12, 31, tzinfo=timezone.utc),
    catchup=False,           # history is loaded with an explicit backfill
    max_active_runs=1,       # each day is scored against the days before it
    default_args={"retries": 2, "retry_delay": timedelta(seconds=30)},
    tags=["flights", "data-quality"],
)
def flights_daily():
    @task
    def extract(ds=None):
        return ft.extract(ds)

    @task(depends_on_past=True)
    def check(_counts, ds=None):
        return ft.check(ds)

    @task.branch
    def decide(result):
        return "quarantine" if result["data_alerts"] else "publish"

    @task(outlets=[WAREHOUSE])
    def publish(ds=None):
        return ft.publish(ds)

    @task
    def quarantine(result, ds=None):
        ft.quarantine(ds, result["data_alerts"])

    done = EmptyOperator(task_id="done", trigger_rule="none_failed_min_one_success")
    result = check(extract())
    decide(result) >> [publish(), quarantine(result)] >> done


flights_daily()
