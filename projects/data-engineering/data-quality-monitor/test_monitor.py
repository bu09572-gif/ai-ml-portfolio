import numpy as np
import pandas as pd
from monitor import prepare, robust_z, score_day


def test_robust_z_ignores_one_outlier_in_history():
    base = pd.Series([10, 11, 10, 9, 10, 500])  # one bad day in the baseline
    assert abs(robust_z(10, base)) < 1
    assert robust_z(30, base) > 4


def test_min_scale_stops_alerts_on_flat_history():
    base = pd.Series([-2.0] * 20)
    assert robust_z(-1.0, base, min_scale=2) == 0.5


def test_not_enough_history_returns_nan():
    assert np.isnan(robust_z(1.0, pd.Series([1.0, 2.0])))


def test_volume_uses_same_weekday_only():
    days = pd.date_range("2013-01-01", periods=63, freq="D")
    vol = [1000 if d.day_of_week < 5 else 700 for d in days]  # weekends are always lower
    df = pd.DataFrame({"day": days, "flights_scheduled": vol, "weather_rows": 72.0})
    for m in ["cancel_rate", "median_dep_delay", "p90_dep_delay", "missing_arr_delay_rate",
              "missing_tail_rate_flown"]:
        df[m] = 0.0
    df = prepare(df)
    last_sat = df.index[df.weekday == 5][-1]
    z = {m: z for _, m, _, z in score_day(df, last_sat)}
    assert z["flights_scheduled"] == 0  # a normal Saturday is not an alert
