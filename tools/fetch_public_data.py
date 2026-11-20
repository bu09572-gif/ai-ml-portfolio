"""Download public datasets that are hosted outside GitHub (OpenML, PyG) and save them as parquet.
Run in GitHub Actions; the files are attached to a release so every project gets the same copy.

    python tools/fetch_public_data.py out/
"""
import hashlib
import json
import sys
from pathlib import Path

from sklearn.datasets import fetch_openml

OPENML = {
    # UCI "default of credit card clients" (Yeh & Lien 2009), 30,000 Taiwanese card holders
    "credit_default_taiwan": {"name": "default-of-credit-card-clients", "version": 1},
    # ULB credit card fraud (Dal Pozzolo et al. 2015), 284,807 European card transactions, Sept 2013
    "creditcard_fraud_ulb": {"name": "creditcard", "version": 1},
}


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for key, spec in OPENML.items():
        b = fetch_openml(**spec, as_frame=True, parser="auto")
        df = b.frame
        p = out / f"{key}.parquet"
        df.to_parquet(p, index=False)
        manifest[key] = {"source": f"openml:{spec['name']} v{spec['version']} (id {b.details['id']})",
                         "license": b.details.get("licence"), "rows": len(df), "columns": list(df.columns),
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        print(key, df.shape, b.details.get("licence"), flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
