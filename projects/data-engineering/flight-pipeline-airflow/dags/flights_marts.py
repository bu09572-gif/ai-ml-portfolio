"""Rebuild the dbt warehouse (models + tests) whenever the daily pipeline publishes."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import Asset, dag

DBT_DIR = Path(__file__).resolve().parents[2] / "flight-ops-dbt-warehouse"
DBT = os.environ.get("DBT_BIN", "dbt")


@dag(schedule=[Asset("duckdb://flights-warehouse/flights")],
     start_date=datetime(2013, 1, 1, tzinfo=timezone.utc), catchup=False, tags=["flights", "dbt"])
def flights_marts():
    BashOperator(task_id="dbt_build", cwd=str(DBT_DIR),
                 bash_command=f"{DBT} build --profiles-dir . --no-use-colors")


flights_marts()
