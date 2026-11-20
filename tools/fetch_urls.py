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
    # valid ICD-10 codes with descriptions for CodiEsp (Zenodo 3706838), CC BY 4.0
    "codiesp_codes.zip": "https://zenodo.org/api/records/3706838/files/codiesp_codes.zip/content",
    # CMS Medicare Coverage Database exports: National and Local Coverage Determinations (public domain)
    "cms_ncd.zip": "https://downloads.cms.gov/medicare-coverage-database/downloads/exports/ncd.zip",
    "cms_current_lcd.zip": "https://downloads.cms.gov/medicare-coverage-database/downloads/exports/current_lcd.zip",
    # CMS Transparency in Coverage public use file 2024 (issuer-level claims and denials)
    "cms_tic_puf_2024.zip": "https://download.cms.gov/marketplace-puf/2024/transparency-in-coverage-puf.zip",
    # Elliptic Bitcoin transaction graph (Weber et al. 2019): 203,769 real transactions, 234,355 payment
    # flows, 4,545 labelled illicit; copy hosted by PyTorch Geometric
    "elliptic_txs_features.csv.zip": "https://data.pyg.org/datasets/elliptic/elliptic_txs_features.csv.zip",
    "elliptic_txs_classes.csv.zip": "https://data.pyg.org/datasets/elliptic/elliptic_txs_classes.csv.zip",
    "elliptic_txs_edgelist.csv.zip": "https://data.pyg.org/datasets/elliptic/elliptic_txs_edgelist.csv.zip",
    # State AI rules for insurers. A list means: try each URL until one works.
    "naic_ai_model_bulletin.pdf": [   # the adopted file's name itself contains "%20", hence %2520
        "https://content.naic.org/sites/default/files/2023-12-4%2520Model%2520Bulletin_Adopted_0.pdf",
        "https://content.naic.org/sites/default/files/call_materials/Model%20Bulletin%2010.23%20Clean.pdf"],
    # Wisconsin's bulletin adopting the NAIC model text as final (March 2025)
    "wisconsin_ai_bulletin_2025.html": "https://oci.wi.gov/Pages/Regulation/Bulletin20250318AI.aspx",
    "naic_ai_bulletin_adoption_map.pdf": "https://content.naic.org/sites/default/files/legal-adoption-map-ai-model-bulletin.pdf",
    "nydfs_circular_letter_2024_07.html": "https://www.dfs.ny.gov/industry-guidance/circular-letters/cl2024-07",
    "colorado_reg_10_1_1_amended.pdf": [
        "https://www.insurereinsure.com/wp-content/uploads/sites/919/2025/08/Amended-Regulation-10-1-1.pdf"],
    "colorado_reg_10_1_1_notice.html": "https://doi.colorado.gov/announcements/notice-of-adoption-amended-regulation-10-1-1-governance-and-risk-management-framework",
}
# Files that change upstream (the weekly CMS export) must not be re-mirrored over the pinned copy.
# If tools/fetch-only.txt lists names, only those are fetched, into their own manifest.
ONLY = Path(__file__).with_name("fetch-only.txt")
HF = {  # Hugging Face datasets: every parquet file of the default config
    "mtsamples": "harishnair04/mtsamples",
    "medqa_usmle_4opt": "GBaker/MedQA-USMLE-4-options",
}


def get(url, dest: Path) -> dict:
    urls = url if isinstance(url, list) else [url]
    for u in urls:
        r = requests.get(u, timeout=600, headers={"User-Agent": "Mozilla/5.0 (portfolio-data-mirror)"})
        if r.ok:
            url = u
            break
        print("failed", r.status_code, u, flush=True)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return {"url": url, "bytes": len(r.content), "sha256": hashlib.sha256(r.content).hexdigest()}


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    only = ONLY.read_text().split() if ONLY.exists() else []
    if only:
        for name in only:
            try:
                manifest[name] = get(URLS[name], out / name)
                print(name, manifest[name]["bytes"], manifest[name]["sha256"], flush=True)
            except requests.RequestException as e:
                print("FAILED", name, e, flush=True)
        (out / f"manifest_{only[0].split('.')[0].split('_')[0]}.json").write_text(json.dumps(manifest, indent=2))
        return
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
