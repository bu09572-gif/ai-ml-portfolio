# Flight Status CDC Pipeline with SCD2 History

An airline's operations database keeps one row per flight and overwrites it as the flight moves along: scheduled, then departed, then arrived (or cancelled). Analysts need more than the latest value. They need to ask what the board looked like at 7 pm on a given evening. This project captures every change from the ops database and builds both a current table and a full history (SCD type 2) in a warehouse.

I replay the real 2013 NYC flights through it: 336,776 flights and 1,000,898 changes, applied in the order they happened.

```
SQLite (ops database)                    DuckDB (warehouse)
  flights  ──triggers──▶ change_log  ──▶  flights_current   latest row per flight
                         (lsn order)      flights_history   every version, valid_from / valid_to
                                          cdc_checkpoint    last lsn applied
```

## Where the changes come from

`src/flightsdata/events.py` turns each flight into its real lifecycle:

| Change | When |
|---|---|
| insert, status scheduled | day before departure, 00:00 New York time (my choice; the data has no publish time) |
| update, status departed | scheduled departure + recorded departure delay |
| update, status arrived | scheduled arrival + recorded arrival delay |
| update, status cancelled | scheduled departure, for flights that never left |

Scheduled times are local, so I convert them with each airport's time zone. As a sanity check, arrival minus departure minus air time (taxi time) has a median of 28 minutes, and only 7 of 327,346 flights come out negative, which is noise in the source data.

## How the capture works

- Triggers on the source table write every insert, update and delete into `change_log`, numbered by `lsn`. The application never knows CDC is there.
- The consumer pulls everything after its last `lsn`, closes the open history version of each flight it touches, inserts the new versions, refreshes `flights_current`, and moves the checkpoint. All of that is one transaction.
- Because the checkpoint commits with the data, and changes at or below the checkpoint are skipped, getting a batch twice (a retry after a crash, for example) changes nothing.

## Results: full year

From `results/cdc_full_year.json`. The source gets each hour of changes, then the consumer syncs, 8,742 times in total.

| Check | Result |
|---|---|
| changes out of lifecycle order | 0 |
| rows where warehouse current table ≠ source table | 0 of 336,776 |
| point-in-time rebuild from history ≠ saved source copy, at 4 moments (2013-01-15 09:00, 02-08 19:00 during the blizzard, 07-04 12:00, 11-27 17:00) | 0 rows at each |
| flights without exactly one current version | 0 |
| versions with a gap or overlap with the next one | 0 |
| changes applied again when 2,000 old changes were re-sent (18 times) | 0 |

- history rows: 1,000,898 (one per change)
- final state: 327,346 arrived, 8,255 cancelled, 1,175 departed with no arrival recorded (mostly diversions)
- consumer speed: about 4,500 changes per second on a 4-core machine (222 s for the year), source writes 50 s

The tests in `test_cdc.py` also check deletes, versions split across batches, and a consumer that dies halfway through a batch: nothing is written, the checkpoint doesn't move, and the retry gives the right result.

## Example: what did the board say at 7 pm during the blizzard?

```sql
select status, count(*)
from flights_history
where valid_from <= timestamptz '2013-02-09 00:00:00+00'        -- 19:00 in New York
  and (valid_to is null or valid_to > timestamptz '2013-02-09 00:00:00+00')
  and cast(timezone('America/New_York', sched_dep_utc) as date) = date '2013-02-08'
group by status;
```

For flights scheduled to leave NYC that day: 429 had arrived, 342 were cancelled, 130 were still scheduled and 29 were in the air. The current table can't answer this, because by the next morning every one of those rows had been overwritten.

## Running it

Needs the raw CSVs from `src/flightsdata` (see [Flight Operations Warehouse](../flight-ops-dbt-warehouse/README.md)).

```bash
pip install duckdb pandas numpy pytest
export PYTHONPATH=../../../src
python pipeline.py --month 2     # one month, ~20 s
python pipeline.py               # whole year, ~5 min
pytest -q test_cdc.py
```

In production the source would be Postgres or MySQL and the change log would come from the write-ahead log (Debezium into Kafka) instead of triggers. The consumer logic, checkpointing and history table stay the same.
