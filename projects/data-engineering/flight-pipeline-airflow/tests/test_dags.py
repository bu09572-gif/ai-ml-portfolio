"""DAG integrity: the files import, the graph is what I expect, and the settings that
make backfills safe are in place."""
from pathlib import Path

import pytest
from airflow.dag_processing.dagbag import DagBag

DAGS = Path(__file__).resolve().parents[1] / "dags"


@pytest.fixture(scope="module")
def bag():
    return DagBag(dag_folder=str(DAGS))


def test_no_import_errors(bag):
    assert bag.import_errors == {}
    assert set(bag.dag_ids) == {"flights_daily", "flights_marts"}


def test_daily_graph(bag):
    d = bag.get_dag("flights_daily")
    edges = {(t.task_id, down) for t in d.tasks for down in t.downstream_task_ids}
    assert edges == {("extract", "check"), ("check", "decide"), ("check", "quarantine"),
                     ("decide", "publish"), ("decide", "quarantine"),
                     ("publish", "done"), ("quarantine", "done")}


def test_backfill_safety(bag):
    d = bag.get_dag("flights_daily")
    assert d.max_active_runs == 1 and not d.catchup
    assert d.get_task("check").depends_on_past
    assert all(t.retries >= 1 for t in d.tasks if t.task_id != "done")
    assert d.get_task("done").trigger_rule == "none_failed_min_one_success"


def test_marts_runs_when_warehouse_updates(bag):
    publish = bag.get_dag("flights_daily").get_task("publish")
    marts = bag.get_dag("flights_marts")
    assert [o.uri for o in publish.outlets] == ["duckdb://flights-warehouse/flights"]
    assert "duckdb://flights-warehouse/flights" in str(marts.timetable.asset_condition)
