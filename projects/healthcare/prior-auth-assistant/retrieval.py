"""Find the right coverage policy (LCD) for a request with BM25, scored two ways:

  HCPCS test   each HCPCS code an LCD lists, queried by its description; correct if a listing LCD is found
  note test    each MTSamples note about a CMS prior-auth service; correct if an LCD with the service's
               policy title is found (several contractors publish the same policy under different ids)

    python retrieval.py      # -> results/retrieval.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import data
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from docqa.retrieve import BM25Index  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
KS = (1, 5, 10)


def index(lcds: pd.DataFrame, with_text: bool = True) -> BM25Index:
    return BM25Index(pd.DataFrame({"header": lcds.title, "text": lcds.indication.fillna("") if with_text else ""}))


def hit_rates(ix: BM25Index, ids: np.ndarray, queries, golds) -> dict:
    hits = {k: [] for k in KS}
    for q, g in zip(queries, golds):
        top = ids[np.argsort(-ix.scores(q), kind="stable")[:max(KS)]]
        for k in KS:
            hits[k].append(bool(set(top[:k]) & set(g)))
    return {f"hit_at_{k}": round(float(np.mean(v)), 3) for k, v in hits.items()} | {"queries": len(hits[1])}


def main():
    lcds = data.lcds()
    ids = lcds.lcd_id.to_numpy()
    full, title = index(lcds), index(lcds, with_text=False)
    h = data.hcpcs_links()
    s = data.service_notes()
    gold = {svc: lcds[lcds.title == t].lcd_id.tolist() for svc, (t, _) in data.SERVICES.items()}
    q = s.description + " " + s.text.str[:3000]
    res = {"lcds": len(lcds), "lcd_titles": int(lcds.title.nunique()),
           "hcpcs_test": {"title_only": hit_rates(title, ids, h.long_description, h.lcd_id),
                          "title_and_coverage_text": hit_rates(full, ids, h.long_description, h.lcd_id)},
           "note_test": {"title_only": hit_rates(title, ids, q, s.service.map(gold)),
                         "title_and_coverage_text": hit_rates(full, ids, q, s.service.map(gold)),
                         "notes_per_service": s.service.value_counts().to_dict()}}
    OUT.mkdir(exist_ok=True)
    (OUT / "retrieval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
