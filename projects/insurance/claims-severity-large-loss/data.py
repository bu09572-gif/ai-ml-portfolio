"""Individual motor claims from freMTPL2 (CASdatasets): 26,444 claim payments joined to the
policy and driver features of the policy they came from. The download, checksum and policy
table come from the motor pricing project next door (../motor-pricing-fairness/data.py).

Unlike that project, claim amounts are NOT capped here: the tail is the subject.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

_P = Path(__file__).resolve().parents[1] / "motor-pricing-fairness" / "data.py"
_spec = importlib.util.spec_from_file_location("fremtpl_data", _P)
fremtpl = importlib.util.module_from_spec(_spec)
sys.modules["fremtpl_data"] = fremtpl
_spec.loader.exec_module(fremtpl)

FEATURES = fremtpl.FEATURES
CATEGORICAL = fremtpl.CATEGORICAL


def claims() -> pd.DataFrame:
    policies = fremtpl.load()                                   # downloads and checks both files
    sev = fremtpl._read_rda(fremtpl._fetch("freMTPL2sev.rda"))
    sev["IDpol"] = sev.IDpol.astype(str).astype(float).astype(int)
    df = sev.merge(policies[["IDpol", "test", *FEATURES]], on="IDpol", how="inner")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    c = claims()
    print(len(c), "claims;", c.test.sum(), "in test policies;", f"total {c.ClaimAmount.sum() / 1e6:.1f}M EUR")
