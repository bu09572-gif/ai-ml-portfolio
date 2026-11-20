"""Where claim denials come from on HealthCare.gov plans, and how predictable a plan's denial rate is.

  composition   denial rates, denial reasons, and how rarely denials are appealed
  variance      how much of the spread in plan denial rates is between issuers (insurers) vs within
  model         a gradient-boosted model of a plan's in-network denial rate, scored on issuers it has
                never seen (GroupKFold), against the same model allowed to know the issuer
"""
from __future__ import annotations

import data
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold

MIN_CLAIMS = 1000          # plans with fewer in-network claims have noisy rates
CAT = ["State", "Plan_Type", "Metal_Level", "New_or_Returning_Plan"]


def reporting_plans(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d.Plan_Number_Claims_Received_In_Network >= MIN_CLAIMS].copy()
    d["claims"] = d.Plan_Number_Claims_Received_In_Network
    d["denial_rate"] = d.Plan_Number_Claims_Denied_In_Network / d.claims
    return d.dropna(subset=["denial_rate"]).reset_index(drop=True)


def wquantiles(x, w, qs):
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return [float(x[o][np.searchsorted(cw, q)]) for q in qs]


def composition(d: pd.DataFrame) -> dict:
    p = reporting_plans(d)
    reasons = {v: float(np.nansum(d["denied_" + k.lower()])) for k, v in data.REASONS.items()}
    tot = sum(reasons.values())
    iss = d.drop_duplicates("Issuer_ID")
    denied = iss.Issuer_Claims_Denied_In_Network.fillna(0) + iss.Issuer_Claims_Denied_Out_of_Network.fillna(0)
    received = iss.Issuer_Claims_Received_In_Network.fillna(0) + iss.Issuer_Claims_Received_Out_of_Network.fillna(0)
    filed, over = iss.Issuer_Internal_Appeals_Filed, iss.Issuer_Number_Internal_Appeals_Overturned
    big = iss[filed >= 100]
    rate = big.Issuer_Number_Internal_Appeals_Overturned / big.Issuer_Internal_Appeals_Filed
    return {
        "medical_plans": len(d), "plans_with_at_least_1000_claims": len(p), "states": int(d.State.nunique()),
        "issuers": int(iss.Issuer_ID.nunique()),
        "issuer_claims_received": int(received.sum()), "issuer_claims_denied": int(denied.sum()),
        "issuer_denial_rate": round(float(denied.sum() / received.sum()), 4),
        "issuer_in_network_denial_rate": round(float(iss.Issuer_Claims_Denied_In_Network.sum() / iss.Issuer_Claims_Received_In_Network.sum()), 4),
        "issuer_out_of_network_denial_rate": round(float(iss.Issuer_Claims_Denied_Out_of_Network.sum() / iss.Issuer_Claims_Received_Out_of_Network.sum()), 4),
        "plan_in_network_denial_rate_p10_p50_p90": [round(v, 3) for v in wquantiles(p.denial_rate.to_numpy(), p.claims.to_numpy(), [.1, .5, .9])],
        "denial_reason_share": {k: round(v / tot, 3) for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])},
        "internal_appeals_filed": int(filed.sum()), "internal_appeals_overturned": int(over.sum()),
        "appeals_per_1000_denials": round(float(1000 * filed.sum() / denied.sum()), 2),
        "internal_appeal_overturn_rate": round(float(over.sum() / filed.sum()), 3),
        "issuers_with_100_plus_appeals": len(big),
        "overturn_rate_across_those_issuers_p10_p50_p90": [round(float(v), 3) for v in rate.quantile([.1, .5, .9])],
        "external_appeals_filed": int(iss.Issuer_External_Appeals_Filed.sum()),
        "external_appeals_overturned": int(iss.Issuer_Number_External_Appeals_Overturned.sum()),
    }


def between_issuer_share(p: pd.DataFrame) -> float:
    """Claim-weighted share of the variance in plan denial rates explained by the issuer."""
    w, y = p.claims, p.denial_rate
    mu = np.average(y, weights=w)
    im = p.groupby("Issuer_ID").apply(lambda g: np.average(g.denial_rate, weights=g.claims), include_groups=False)
    between = np.sum(w * (p.Issuer_ID.map(im) - mu) ** 2)
    return float(between / np.sum(w * (y - mu) ** 2))


def _x(p: pd.DataFrame, with_issuer: bool) -> pd.DataFrame:
    x = p[CAT].astype("category")
    x["log_issuer_claims"] = np.log1p(p.Issuer_Claims_Received_In_Network)
    x["out_of_network_share"] = p.Plan_Number_Claims_Received_Out_of_Network / (p.claims + p.Plan_Number_Claims_Received_Out_of_Network)
    if with_issuer:
        x["issuer"] = p.Issuer_ID.astype("category")
    return x


def _scores(y, pred, w) -> dict:
    mae = np.average(np.abs(y - pred), weights=w)
    r2 = 1 - np.sum(w * (y - pred) ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)
    return {"weighted_mae_points": round(100 * float(mae), 2), "weighted_r2": round(float(r2), 3)}


def model(p: pd.DataFrame, folds: int = 5) -> dict:
    y, w = p.denial_rate.to_numpy(), p.claims.to_numpy()
    out = {}
    setups = {"unseen_issuers": (GroupKFold(folds), False), "known_issuers": (KFold(folds, shuffle=True, random_state=0), True)}
    for name, (cv, with_issuer) in setups.items():
        x = _x(p, with_issuer)
        pred, base = np.zeros(len(p)), np.zeros(len(p))
        groups = p.Issuer_ID if isinstance(cv, GroupKFold) else None
        for tr, te in cv.split(x, y, groups=groups):
            m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=20,
                                              categorical_features="from_dtype", random_state=0)
            m.fit(x.iloc[tr], y[tr], sample_weight=w[tr])
            pred[te] = m.predict(x.iloc[te])
            base[te] = np.average(y[tr], weights=w[tr])
        out[name] = {"model": _scores(y, pred, w), "constant": _scores(y, base, w)}
    return out
