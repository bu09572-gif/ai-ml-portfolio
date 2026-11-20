"""Measure whether the monitor catches problems I inject on purpose.

For 60 random days (after the first 8 weeks, so there is history), I damage that day's
real data in one way, recompute the day's metrics, and check whether the matching
data check fires (|z| >= 4). The rest of the year stays untouched.

    python faults.py        # writes results/fault_detection.csv and results/fault_detection.json
"""
from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd
from monitor import OUT, RAW, Z_ALERT, daily_metrics, prepare, score_day

N_DAYS = 60
SEED = 7


def _null_frac(f: pd.DataFrame, col: str, frac: float, rng) -> pd.DataFrame:
    f = f.copy()
    flown = f.index[f.dep_time.notna() & f[col].notna()]
    hit = rng.choice(flown, size=max(1, int(round(frac * len(flown)))), replace=False)
    f.loc[hit, col] = None
    return f


def _drop_frac(f: pd.DataFrame, frac: float, rng) -> pd.DataFrame:
    return f.drop(rng.choice(f.index, size=int(round(frac * len(f))), replace=False))


def _drop_weather_hours(w: pd.DataFrame, hours: int, rng) -> pd.DataFrame:
    origin = rng.choice(w.origin.unique())
    rows = w.index[w.origin == origin]
    return w.drop(rng.choice(rows, size=min(hours, len(rows)), replace=False))


# name -> (metric that should fire, what to break)
FAULTS = {
    "flights: 10% of rows missing": ("flights_scheduled", lambda f, w, g: (_drop_frac(f, 0.10, g), w)),
    "flights: 25% of rows missing": ("flights_scheduled", lambda f, w, g: (_drop_frac(f, 0.25, g), w)),
    "arr_delay blank on 1% of flown": ("missing_arr_delay_rate", lambda f, w, g: (_null_frac(f, "arr_delay", 0.01, g), w)),
    "arr_delay blank on 3% of flown": ("missing_arr_delay_rate", lambda f, w, g: (_null_frac(f, "arr_delay", 0.03, g), w)),
    "tailnum blank on 1% of flown": ("missing_tail_rate_flown", lambda f, w, g: (_null_frac(f, "tailnum", 0.01, g), w)),
    "tailnum blank on 3% of flown": ("missing_tail_rate_flown", lambda f, w, g: (_null_frac(f, "tailnum", 0.03, g), w)),
    "weather: 3 hours missing at one airport": ("weather_rows", lambda f, w, g: (f, _drop_weather_hours(w, 3, g))),
    "weather: 6 hours missing at one airport": ("weather_rows", lambda f, w, g: (f, _drop_weather_hours(w, 6, g))),
}


def main():
    rng = np.random.default_rng(SEED)
    load = "select *, make_date(year, month, day) as d from read_csv_auto('{p}', nullstr='NA')"
    flights = duckdb.sql(load.format(p=RAW / "flights.csv")).df()
    weather = duckdb.sql(load.format(p=RAW / "weather.csv")).df()
    daily = prepare(daily_metrics())
    days = rng.choice(np.arange(56, len(daily)), size=N_DAYS, replace=False)

    rows = []
    for i in days:
        d = daily.day.iloc[i]
        f_day = flights[flights.d == d].drop(columns="d").reset_index(drop=True)
        w_day = weather[weather.d == d].drop(columns="d").reset_index(drop=True)
        clean = {m: z for _, m, _, z in score_day(daily, i)}
        for name, (metric, fn) in FAULTS.items():
            f2, w2 = fn(f_day, w_day, rng)
            new = daily_metrics(flights=f2, weather=w2).iloc[0]
            z = {m: z for _, m, _, z in score_day(daily, i, row=new)}[metric]
            rows.append({"day": str(pd.Timestamp(d).date()), "fault": name, "metric": metric,
                         "z_clean": round(clean[metric], 2), "z_faulted": round(z, 2),
                         "already_alerting": abs(clean[metric]) >= Z_ALERT,
                         "detected": abs(z) >= Z_ALERT})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "fault_detection.csv", index=False)
    fresh = res[~res.already_alerting]
    summary = {
        "days_tested": N_DAYS, "seed": SEED, "z_threshold": Z_ALERT,
        "detection_rate": {k: round(float(v), 3) for k, v in fresh.groupby("fault", sort=False).detected.mean().items()},
        "median_abs_z_faulted": {k: round(float(v), 1) for k, v in fresh.groupby("fault", sort=False).z_faulted.apply(lambda s: s.abs().median()).items()},
        "excluded_already_alerting": int(res.already_alerting.sum()),
    }
    (OUT / "fault_detection.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
