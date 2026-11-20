# Daily Flight Pipeline on Airflow

An Airflow 3 pipeline that loads one day of flight data at a time and only lets a day into the warehouse if its data checks pass. I backfilled all of 2013 (365 daily runs) on the real NYC flights data.

```
flights_daily  (one run per day)

extract ──▶ check ──▶ decide ──▶ publish ─────┐
               │                              ├──▶ done
               └────────────────▶ quarantine ─┘

publish updates the asset  duckdb://flights-warehouse/flights
                                      │
flights_marts  (runs when that asset changes)   dbt build: models + tests
```

| Task | What it does |
|---|---|
| extract | copies that day's flights and weather from the raw files into staging |
| check | computes the day's quality metrics, saves them, and scores them against earlier days (same robust z-score as the [Flight Data Quality Monitor](../flight-data-quality-monitor/README.md)) |
| decide | sends the day to quarantine if any **data** check fires. Ops alerts like storms or holidays are recorded but don't block anything, because that data is correct |
| publish | replaces that day's rows in the warehouse in one transaction and records the status |
| quarantine | keeps the day out of the warehouse and records which checks fired and their z-scores |

Settings that make a backfill safe:

- `max_active_runs=1` and `depends_on_past` on `check`, because each day is scored against the days before it
- every write is keyed by day (delete the day, then insert it), so re-running a day gives the same result. I re-ran the same day twice in a row and got the same 943 rows.
- 2 retries per task, and `done` uses `none_failed_min_one_success`, so the run succeeds whichever branch it takes
- the task logic lives in `dags/flights_tasks.py` with no Airflow imports, so it can be tested on its own

## Results: 2013 backfill

From `results/backfill_2013-01-01_2013-12-31.json`:

- 365 runs, 0 failed, about 1 s per run on a 4-core machine (348 s in total)
- 353 days published (325,594 flight rows), 12 days quarantined
- the 12 quarantined days are exactly the 12 days the standalone quality monitor flags when it runs over the whole year at once

| Quarantined day | Why |
|---|---|
| 2013-03-18, 04-10, 06-07, 07-07, 07-10, 09-12 | too many departed flights with no arrival delay (up to 3.7%, z up to 11.1) |
| 2013-08-22, 10-25, 10-26, 11-02, 12-30 | weather file missing 4 to 15 hours |
| 2013-11-01 | both |

The matching result is the main check here. The DAG only ever sees one day plus the metrics it stored on earlier runs. The monitor sees the whole year. Getting the same 12 days shows the day-by-day state is handled correctly.

`tests/test_dags.py` checks that both DAGs import with no errors, that the task graph has the expected edges, the backfill settings above, and that `flights_marts` is scheduled on the asset that `publish` updates.

## Running it

```bash
pip install "apache-airflow==3.4.0" duckdb pandas numpy pytest \
  --constraint https://raw.githubusercontent.com/apache/airflow/constraints-3.4.0/constraints-3.13.txt
export PYTHONPATH=../../../src:$PWD/dags AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow db migrate && airflow dags reserialize

pytest -q tests
python backfill.py 2013-01-01 2013-12-31    # runs each day with dag.test(), then compares with the monitor
```

`backfill.py` runs the real tasks in one process, without a scheduler, so it works in CI. On a running deployment, use `airflow backfill create --dag-id flights_daily --from-date 2013-01-01 --to-date 2013-12-31`.

The raw CSVs come from `src/flightsdata` (see [Flight Operations Warehouse](../flight-ops-dbt-warehouse/README.md)).
