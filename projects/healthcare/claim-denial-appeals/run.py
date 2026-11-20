"""Claim denial analytics on the CMS 2024 Transparency in Coverage file.

    python run.py      # -> results/denials.json, results/denial_reasons.png
"""
from __future__ import annotations

import json
from pathlib import Path

import data
import denials as dn

OUT = Path(__file__).resolve().parent / "results"


def chart(comp: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = comp["denial_reason_share"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(list(r)[::-1], [100 * v for v in list(r.values())[::-1]], color="#3b6ea8")
    ax.set_xlabel("share of reported denials (%)")
    ax.set_title("Why HealthCare.gov plans denied claims, 2024")
    fig.tight_layout()
    fig.savefig(OUT / "denial_reasons.png", dpi=120)
    plt.close(fig)


def main():
    d = data.plans()
    p = dn.reporting_plans(d)
    comp = dn.composition(d)
    res = {"composition": comp, "variance_between_issuers": round(dn.between_issuer_share(p), 3),
           "denial_rate_model": dn.model(p)}
    OUT.mkdir(exist_ok=True)
    (OUT / "denials.json").write_text(json.dumps(res, indent=2))
    chart(comp)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
