# Flights warehouse (dbt + DuckDB)

A small star-schema warehouse over every flight that left New York (JFK, LGA, EWR) in 2013, built with dbt on DuckDB, with tests on the staging and mart layers.

Data: 336,776 flights from the BTS on-time database, plus airlines, airports and aircraft (FAA) and hourly weather at each airport (NOAA), packaged as `nycflights13` (CC0). `python -m flightsdata.source` downloads it from PyPI and checks its sha256.

## Models

```
raw CSVs ─► staging (views)          ─► marts (tables)
            stg_flights                 fct_flights          one row per flight + origin weather at the scheduled hour
            stg_airlines / airports     dim_carrier, dim_airport, dim_aircraft
            stg_planes / weather        mart_carrier_monthly on-time rate, cancel rate, median / p90 delay
                                        mart_route_delays    routes with 100+ flights
                                        mart_weather_impact  delays and cancellations by weather at departure
```

`stg_flights` converts the source's integer clock times (517 = 05:17) into timestamps and builds a surrogate key from date, carrier, flight number, origin and scheduled time.

## Tests

30 nodes: 13 models and 17 tests (unique, not_null, accepted_values, relationships, a custom `accepted_range` test, and a row-count check that the weather join doesn't duplicate or drop flights).

Result of `dbt build`: **27 pass, 3 warn, 0 errors.** The warnings are real problems in the source data:

| Test | Rows flagged | What it is |
|---|---|---|
| flights.dest → airports | 7,602 | San Juan (SJU), Aguadilla (BQN), St. Thomas (STT), Ponce (PSE): the FAA airport extract has no Puerto Rico or Virgin Islands airports |
| flights.tail_number → planes | 50,094 | mostly Envoy (MQ, 25,395) and American (AA, 22,474); the registry extract doesn't include those fleets. Another 2,512 flights have no tail number at all |
| weather.wind_speed between 0 and 150 mph | 1 | EWR, 2013-02-12 08:00 UTC, 1,048 mph, a sensor or entry error |

I set these to `warn` instead of `error` because they are known gaps in the source, not bugs in the models. The reference tables would need to be patched upstream. The 1,048 mph reading should be nulled before anyone computes wind averages.

## Some results from the marts

On-time arrival rate (within 15 minutes) for carriers with 10,000+ flights, 2013:

| Carrier | Flights | On time | Cancelled |
|---|---|---|---|
| US Airways | 20,536 | 82.0% | 3.2% |
| Delta | 48,110 | 81.8% | 0.7% |
| American | 32,729 | 81.2% | 1.9% |
| United | 58,665 | 78.2% | 1.2% |
| Southwest | 12,275 | 75.2% | 1.6% |
| Endeavor | 18,460 | 74.9% | 5.7% |
| JetBlue | 54,635 | 73.8% | 0.9% |
| Envoy | 26,397 | 72.9% | 4.7% |
| ExpressJet | 54,173 | 68.5% | 5.2% |

Weather at the origin airport at the scheduled hour:

| Weather | Flights | Cancelled | Avg departure delay |
|---|---|---|---|
| visibility < 1 mile | 3,975 | 8.9% | 29.7 min |
| precipitation > 0.1 in/hr | 3,937 | 8.3% | 39.3 min |
| wind > 25 mph | 5,607 | 9.2% | 18.0 min |
| normal | 321,701 | 2.2% | 12.0 min |

Cancellations are about 4x higher in bad weather.

## Running it

```bash
pip install dbt-duckdb
make data      # from the repo root: download the CSVs
make build     # dbt build (models + tests), about 5 s
make docs      # lineage graph in the browser
```

The warehouse file is `data/warehouse/flights.duckdb`. `WAREHOUSE_PATH` and `RAW_DIR` override the locations.
