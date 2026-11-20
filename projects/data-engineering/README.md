# Data engineering projects

| Project | What it does | Key result |
|---|---|---|
| [Flight Operations Warehouse (dbt + DuckDB)](flight-ops-dbt-warehouse) | Star schema over real flights with dbt tests | 336,776 flights; 13 models and 17 tests, all passing in CI |
| [Flight Data Quality Monitor](flight-data-quality-monitor) | Daily anomaly checks, tested by breaking the data on purpose | 480 injected-fault trials; large problems caught every time |
| [Flight Status CDC Pipeline with SCD2 History](flight-status-cdc) | Change capture from an ops database into current and history tables | 1,000,898 changes replayed; all reconciliation checks at 0 |
| [Streaming Flight Lakehouse (Kafka, Spark, Delta Lake)](flight-streaming-lakehouse) | Bronze, silver and gold Delta tables from a Kafka stream, with a crash test | Killed with kill -9 mid-stream; after restart all 4 checks are 0 |
| [Daily Flight Pipeline on Airflow](flight-pipeline-airflow) | Extract, check, then publish or quarantine each day | 365 daily runs, 0 failed, 12 bad days quarantined |

[Back to all projects](../../README.md)
