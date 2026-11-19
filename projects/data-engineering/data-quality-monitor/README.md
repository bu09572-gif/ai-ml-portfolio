# Data quality monitor for a daily flights feed

Every day of the 2013 NYC flights data (336,776 flights, 26,115 hourly weather rows) gets seven metrics. Each metric is compared with its own recent history, and a day is flagged when it is far outside what is normal.

I split the checks into two kinds, because they go to different people:

- **data checks** look for a broken feed: rows missing, fields left blank
- **ops checks** look for real-world disruption: storms, holidays, bad delay days

| Metric | Kind | Compared with |
|---|---|---|
| flights scheduled | data* | same weekday, previous 8 weeks |
| weather rows | data | same weekday, previous 8 weeks |
| share of departed flights with no arrival delay | data | previous 28 days |
| share of departed flights with no tail number | data | previous 28 days |
| cancel rate | ops | previous 28 days |
| median departure delay | ops | previous 28 days |
| 90th percentile departure delay | ops | previous 28 days |

\* Volume is labeled ops in the output, because in this data every big drop turned out to be a holiday or storm, not a feed problem. In a live feed it is the first sign of a partial load.

The score is a robust z-score, `(value - median) / (1.4826 * MAD)`. Using the median and MAD means one bad day doesn't skew the baseline for the following days. Each metric has a smallest spread it will accept, so a value that sat at exactly -2 minutes for weeks doesn't alert on a 1-minute change. |z| ≥ 4 is an alert.

## Results on the real 2013 data

From `results/summary.json`:

- 365 days, 2,478 checks, 148 alerts on 98 days
- data alerts on 12 days; ops alerts on 92 days

The largest alerts match events I could confirm:

| Day | Metric | z | What happened |
|---|---|---|---|
| 2013-02-08, 02-09 | cancel rate | 49.5, 56.1 | blizzard in the Northeast (Winter Storm Nemo) |
| 2013-11-28, 11-29 | flights scheduled | -35.9, -32.9 | Thanksgiving and the day after |
| 2013-07-04 | flights scheduled | -25.2 | Independence Day |

The 12 data-alert days are real gaps in the files: 6 days where the weather file is missing between 4 and 15 hours, and 7 days where an unusual share of flights departed but have no arrival delay (up to 3.7% on 2013-09-12; the median day is 0.2%). Those are mostly diverted flights, which still matters for anyone averaging arrival delay.

92 ops days is a lot. Bad weather and holiday days are frequent in a year of NYC flying, and I'd send these as a daily digest rather than a page.

## Does it catch problems I put in on purpose?

`faults.py` picks 60 random days, breaks that day's real data in one way, recomputes the metrics, and checks whether the matching check fires. Days where that check was already alerting are left out (18 of the 480 trials).

| Injected problem | Caught | Median \|z\| |
|---|---|---|
| 10% of a day's flight rows missing | 90.6% | 8.2 |
| 25% of a day's flight rows missing | 100% | 22.3 |
| arrival delay blank on 1% of departed flights | 15.3% | 3.1 |
| arrival delay blank on 3% of departed flights | 100% | 9.6 |
| tail number blank on 1% of departed flights | 100% | 5.1 |
| tail number blank on 3% of departed flights | 100% | 15.0 |
| 3 weather hours missing at one airport | 5.1% | 3.0 |
| 6 weather hours missing at one airport | 100% | 6.0 |

What this tells me:

- Big problems are caught every time. Small ones depend on how noisy the metric is. Tail number is almost never blank, so 1% blank is obvious. Arrival delay is already blank for diverted flights, so 1% more hides in the normal noise.
- The weather file should have exactly 72 rows a day (24 hours × 3 airports). A z-score is the wrong tool for that: a plain "expect 72" rule would catch every missing hour. In the real data that rule would flag 25 days, since 340 of 365 days are complete.

## Running it

Needs the raw CSVs from `src/flightsdata` (see the dbt warehouse README).

```bash
pip install duckdb pandas numpy pytest
python monitor.py      # results/summary.json, alerts.csv, daily_metrics.csv  (~2 s)
python faults.py       # results/fault_detection.json, fault_detection.csv     (~20 s)
pytest -q test_monitor.py
```

## Data

nycflights13 (Wickham), public domain (CC0). It is built from the US Bureau of Transportation Statistics on-time data, FAA aircraft registry and NOAA weather. I download it from the PyPI package, not a copy in this repo.
