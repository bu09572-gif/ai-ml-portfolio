"""Find claims worth pursuing for subrogation: our driver was hit, and the other driver was at fault.

Framing. Every two-vehicle crash in the NYPD data becomes two claims, one from each vehicle's point of
view ("our insured"). The label comes from the police officer's contributing factors:
  recovery  the other driver has a driver-fault factor and ours has none
  no        ours has one (alone or both): no recovery, or a shared-fault fight
Only crashes where the officer named a driver-fault factor for at least one driver are labelled; the
rest ("Unspecified") have no answer to learn from.

Features are what a claims handler knows at first notice of loss, before any police report: both
vehicles' maneuvers, our point of impact and damage, vehicle types and ages, time, place and injuries.
The contributing factors themselves are never features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

DRIVER_FAULT = {
    "Driver Inattention/Distraction", "Following Too Closely", "Failure to Yield Right-of-Way", "Unsafe Speed",
    "Passing or Lane Usage Improper", "Driver Inexperience", "Passing Too Closely", "Traffic Control Disregarded",
    "Turning Improperly", "Backing Unsafely", "Unsafe Lane Changing", "Alcohol Involvement", "Aggressive Driving/Road Rage",
    "Fell Asleep", "Passenger Distraction", "Outside Car Distraction", "Failure to Keep Right", "Fatigued/Drowsy",
    "Drugs (illegal)", "Cell Phone (hand-Held)", "Cell Phone (hands-free)", "Texting", "Using On Board Navigation Device",
    "Other Electronic Device", "Eating or Drinking", "Listening/Using Headphones", "Oversized Vehicle"}
CAT = ["our_pre_crash", "our_impact", "our_damage", "our_type", "other_pre_crash", "other_impact", "other_type",
       "borough", "our_direction", "other_direction"]
NUM = ["our_vehicle_age", "other_vehicle_age", "our_occupants", "hour", "weekday", "injured", "month_day"]
REAR = r"Back End|Rear"


def _fault(v: pd.DataFrame) -> pd.Series:
    return v.contributing_factor_1.isin(DRIVER_FAULT) | v.contributing_factor_2.isin(DRIVER_FAULT)


def claims(crashes: pd.DataFrame, vehicles: pd.DataFrame) -> pd.DataFrame:
    """Two claims per two-vehicle crash, with features and the police-based label."""
    v = vehicles[vehicles.groupby("collision_id").collision_id.transform("size") == 2].copy()
    v = v.sort_values(["collision_id", "unique_id"])
    v["fault"] = _fault(v)
    v["vehicle_age"] = 2023 - pd.to_numeric(v.vehicle_year, errors="coerce").where(lambda s: s.between(1950, 2024))
    keep = ["collision_id", "pre_crash", "point_of_impact", "vehicle_damage", "vehicle_type", "vehicle_age",
            "travel_direction", "vehicle_occupants", "fault"]
    a = v.groupby("collision_id").nth(0)[keep]
    b = v.groupby("collision_id").nth(1)[keep]
    pairs = a.merge(b, on="collision_id", suffixes=("_a", "_b"))
    rows = []
    for ours, other in [("a", "b"), ("b", "a")]:
        rows.append(pd.DataFrame({
            "collision_id": pairs.collision_id,
            "our_pre_crash": pairs[f"pre_crash_{ours}"], "our_impact": pairs[f"point_of_impact_{ours}"],
            "our_damage": pairs[f"vehicle_damage_{ours}"], "our_type": pairs[f"vehicle_type_{ours}"],
            "our_vehicle_age": pairs[f"vehicle_age_{ours}"], "our_direction": pairs[f"travel_direction_{ours}"],
            "our_occupants": pd.to_numeric(pairs[f"vehicle_occupants_{ours}"], errors="coerce"),
            "other_pre_crash": pairs[f"pre_crash_{other}"], "other_impact": pairs[f"point_of_impact_{other}"],
            "other_type": pairs[f"vehicle_type_{other}"], "other_vehicle_age": pairs[f"vehicle_age_{other}"],
            "other_direction": pairs[f"travel_direction_{other}"],
            "our_fault": pairs[f"fault_{ours}"], "other_fault": pairs[f"fault_{other}"]}))
    c = pd.concat(rows, ignore_index=True)
    cr = crashes.set_index("collision_id")
    t = pd.to_datetime(cr.crash_date.str[:10] + " " + cr.crash_time, errors="coerce")
    c = c.join(pd.DataFrame({"date": t.dt.normalize(), "hour": t.dt.hour, "weekday": t.dt.weekday, "month_day": t.dt.day,
                             "borough": cr.borough, "injured": cr.number_of_persons_injured}), on="collision_id")
    c["labelled"] = c.our_fault | c.other_fault
    c["recovery"] = (c.other_fault & ~c.our_fault).astype(int)
    return c


def rule(c: pd.DataFrame) -> pd.Series:
    """The classic referral rule: we were parked, or stopped/slowing and hit from behind."""
    stopped = c.our_pre_crash.isin(["Stopped in Traffic", "Slowing or Stopping"])
    rear = c.our_impact.fillna("").str.contains(REAR)
    return (c.our_pre_crash.eq("Parked") | (stopped & rear)).astype(int)


def x(c: pd.DataFrame, top: int = 40) -> pd.DataFrame:
    """Features; each category keeps its 40 most common values (vehicle types have hundreds of spellings)."""
    out = c[CAT + NUM].copy()
    for col in CAT:
        s = out[col].fillna("missing").astype(str)
        common = s.value_counts().index[:top]
        out[col] = pd.Categorical(s.where(s.isin(common), "other"), categories=sorted(set(common) | {"other"}))
    return out


def model():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, max_depth=6, min_samples_leaf=50,
                                          l2_regularization=1.0, categorical_features="from_dtype",
                                          early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=0)


def referral_scores(y: np.ndarray, flag: np.ndarray) -> dict:
    tp = int((flag & (y == 1)).sum())
    return {"referred_share": round(float(flag.mean()), 3), "precision": round(tp / max(int(flag.sum()), 1), 3),
            "recall": round(tp / max(int(y.sum()), 1), 3)}


def ranked(y: np.ndarray, p: np.ndarray, share: float) -> dict:
    k = int(len(y) * share)
    flag = np.zeros(len(y), bool)
    flag[np.argsort(-p)[:k]] = True
    return referral_scores(y, flag)


def scores(y, p) -> dict:
    return {"auc": round(roc_auc_score(y, p), 4), "average_precision": round(average_precision_score(y, p), 4)}
