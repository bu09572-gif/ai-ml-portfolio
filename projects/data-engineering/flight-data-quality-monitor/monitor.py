"""Daily data-quality monitor for the flights feed.

For every day it computes a handful of metrics and compares each one with that
metric's recent history:

  volume metrics (flights scheduled)  vs. the same weekday over the previous 8 weeks
  rate metrics (cancel rate, missing arrival delay, ...)  vs. the previous 28 days

The comparison is a robust z-score: (value - median) / (1.4826 * MAD), so one bad
day doesn't distort the baseline for the next. |z| >= 4 raises an alert.

    PYTHONPATH=../../../src python monitor.py                      # writes results/alerts.csv, results/daily_metrics.csv, results/summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from flightsdata.dq import KIND, RAW, Z_ALERT, daily_metrics, prepare, robust_z, score, score_day  # noqa: F401

OUT = Path(__file__).resolve().parent / "results"


def main():
    OUT.mkdir(exist_ok=True)
    daily = daily_metrics()
    scores = score(daily)
    alerts = scores[scores.z.abs() >= Z_ALERT].copy()
    alerts["direction"] = np.where(alerts.z > 0, "high", "low")
    daily.to_csv(OUT / "daily_metrics.csv", index=False)
    alerts.assign(value=alerts.value.round(4), z=alerts.z.round(2)).to_csv(OUT / "alerts.csv", index=False)
    summary = {
        "days": len(daily),
        "metrics_per_day": len(KIND), "z_threshold": Z_ALERT,
        "checks": int(scores.z.notna().sum()),
        "alerts": len(alerts),
        "days_with_alerts": int(alerts.day.nunique()),
        "alerts_by_metric": alerts.metric.value_counts().to_dict(),
        "days_with_alerts_by_kind": alerts.groupby("kind").day.nunique().to_dict(),
        "top_days": [
            {"day": str(pd.Timestamp(d).date()), "max_abs_z": round(float(g.z.abs().max()), 1),
             "metrics": sorted(g.metric.unique().tolist())}
            for d, g in sorted(alerts.groupby("day"), key=lambda kv: -kv[1].z.abs().max())[:12]],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
