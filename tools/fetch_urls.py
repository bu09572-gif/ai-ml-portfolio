"""Mirror public files that the cloud sandbox can't reach (Zenodo, CMS, Hugging Face) into a
GitHub release, so every project downloads one fixed copy. Run in GitHub Actions.

    python tools/fetch_urls.py out/
"""
import hashlib
import json
import sys
from pathlib import Path

import requests

URLS = {
    # CodiEsp gold standard, ICD-10 coded clinical cases (Miranda-Escalada et al. 2020), CC BY 4.0
    "codiesp.zip": "https://zenodo.org/api/records/3837305/files/codiesp.zip/content",
    # CMS Medicare Coverage Database exports: National and Local Coverage Determinations (public domain)
    "cms_ncd.zip": "https://downloads.cms.gov/medicare-coverage-database/downloads/exports/ncd.zip",
    "cms_current_lcd.zip": "https://downloads.cms.gov/medicare-coverage-database/downloads/exports/current_lcd.zip",
    # CMS Transparency in Coverage public use file 2024 (issuer-level claims and denials)
    "cms_tic_puf_2024.zip": "https://download.cms.gov/marketplace-puf/2024/transparency-in-coverage-puf.zip",
}
HF = {  # Hugging Face datasets: every parquet file of the default config
    "mtsamples": "harishnair04/mtsamples",
    "medqa_usmle_4opt": "GBaker/MedQA-USMLE-4-options",
}


def get(url: str, dest: Path) -> dict:
    r = requests.get(url, timeout=600, headers={"User-Agent": "portfolio-data-mirror"})
    r.raise_for_status()
    dest.write_bytes(r.content)
    return {"url": url, "bytes": len(r.content), "sha256": hashlib.sha256(r.content).hexdigest()}


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, url in URLS.items():
        manifest[name] = get(url, out / name)
        print(name, manifest[name]["bytes"], flush=True)
    for key, repo in HF.items():
        files = requests.get(f"https://huggingface.co/api/datasets/{repo}/parquet", timeout=60).json()
        print(repo, json.dumps(files)[:400], flush=True)
        configs = files if isinstance(files, dict) else {}
        cfg = "default" if "default" in configs else next(iter(configs))
        for split, urls in configs[cfg].items():
            for i, u in enumerate(urls):
                name = f"{key}__{split}__{i}.parquet"
                manifest[name] = get(u, out / name) | {"hf_repo": repo, "config": cfg, "split": split}
                print(name, manifest[name]["bytes"], flush=True)
    (out / "manifest_v2.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
