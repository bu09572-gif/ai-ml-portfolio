"""Turn state AI rules into a requirements register, and check my own models against it.

  requirements   split each rule into sentences; keep the obligations (shall, must, should, required,
                 expected to); tag each with the control themes its wording names
  crosswalk      which themes each rule covers, with sentence counts and the page of each requirement
  evidence       for a model folder in this repo, look for the evidence each theme asks for (README
                 sections, metrics, tests in CI, data checksums) and report found / partial / gap
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

OBLIGATION = re.compile(r"\b(shall|must|should|is required|are required|required to|expected to|is responsible|are responsible)\b", re.I)
THEMES = {
    "governance and accountability": r"governance|board of directors|senior management|accountab|committee|roles and responsibilit|oversight",
    "written program and policies": r"written (program|policies)|policies and procedures|AIS Program|risk management framework",
    "inventory and documentation": r"inventory|documentation|document(ed|ing)? |record[- ]?keeping|maintain records",
    "data quality and sources": r"data (quality|lineage|integrity|sources?|governance|practices|management|used)|external consumer data|ECDIS|external data",
    "unfair discrimination and bias testing": r"unfair(ly)? discriminat|\bbias|disparate|protected class|proxy|quantitative testing",
    "validation, testing and monitoring": r"validat|\btest(ing|s|ed)?\b|monitor|drift|performance|accuracy|reliab",
    "explainability and transparency": r"explain|transparen|interpretab|understand(able)?",
    "third-party vendors": r"third[- ]part|vendor|external (developer|provider)",
    "consumer notice and adverse decisions": r"notice|notif|adverse|disclos|consumer complaint|right to",
    "regulator reporting and examination": r"examination|investigation|report(ing)? to the (division|department|commissioner|superintendent)|upon request|submit .* report|attest",
}
THEME_RE = {k: re.compile(v, re.I) for k, v in THEMES.items()}
SENT = re.compile(r"(?<=[.;:])\s+(?=[A-Z(•\-])|\n(?=\s*(?:[a-z]\.|\d+\.|[A-Z]\.|•|\([a-z0-9]+\)))")


def sentences(page: str) -> list[str]:
    flat = re.sub(r"\s*\n\s*", " \n", page)
    out = []
    for s in SENT.split(flat):
        s = re.sub(r"\s+", " ", s).strip()
        if len(s.split()) >= 6:
            out.append(s)
    return out


def requirements(pages: list[str], rule: str) -> list[dict]:
    reqs = []
    for i, page in enumerate(pages, 1):
        for s in sentences(page):
            if OBLIGATION.search(s):
                themes = [k for k, r in THEME_RE.items() if r.search(s)]
                reqs.append({"rule": rule, "page": i, "text": s, "themes": themes})
    return reqs


def crosswalk(reqs: list[dict]) -> dict:
    rules = sorted({r["rule"] for r in reqs})
    out = {}
    for t in THEMES:
        out[t] = {rule: sum(t in r["themes"] for r in reqs if r["rule"] == rule) for rule in rules}
    return out


# ---------- evidence in this repo ----------

def _json_keys(obj, prefix="") -> list[str]:
    if isinstance(obj, dict):
        return [k2 for k, v in obj.items() for k2 in [prefix + str(k)] + _json_keys(v, prefix + str(k) + ".")]
    return []


def _found(pattern: str, haystack: str) -> bool:
    return bool(re.search(pattern, haystack, re.I | re.M))


def evidence(model_dir: str) -> dict:
    """Evidence per theme for one model folder: status found / partial / gap, with where it was found."""
    d = ROOT / model_dir
    readme = re.sub(r"\]\([^)]*\)", "]", (d / "README.md").read_text())     # drop link targets
    keys = " ".join(k for f in sorted((d / "results").glob("*.json")) for k in _json_keys(json.loads(f.read_text())))
    code = " ".join(p.read_text() for p in sorted(d.glob("*.py")))
    for sib in sorted({m for m in re.findall(r'"\.\./([\w-]+)', code)} | set(re.findall(r'/ "([\w-]+)" / "data\.py"', code))):
        code += " ".join(p.read_text() for p in sorted((d.parent / sib).glob("data.py")))   # data loaded from a sibling project
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    tests = [p.name for p in d.glob("test_*.py")]
    in_ci = [t for t in tests if t in ci]
    mrc = (ROOT / "projects/finance/model-risk-checker/run.py").read_text()     # models the validation toolkit checks
    name = d.name
    sections = re.findall(r"^## (.+)$", readme, re.M)

    def ev(status, where):
        return {"status": status, "evidence": where}

    out = {}
    out["governance and accountability"] = ev("gap", "solo project: no named owner, approval body or sign-off record")
    out["written program and policies"] = ev("gap", "no written AI program; the root README's rules (real data, tests in CI) are the closest")
    has_doc = all(_found(s, readme) for s in [r"^## Data", r"^## (Results|Models)"])
    out["inventory and documentation"] = ev("found" if has_doc and _found(r"^## (Limits|Running it)", readme) else "partial",
                                            f"README sections: {', '.join(sections)}")
    sha = _found(r"sha-?256|SHA\s*=|hexdigest", code + readme)
    out["data quality and sources"] = ev("found" if sha and _found(r"^## Data", readme) else "partial",
                                         "data source named in README; download checked against a pinned SHA-256" if sha else "data source named in README")
    bias_keys = _found(r"fair|discriminat|adverse_impact|disparate", keys)
    bias_text = _found(r"discrimination-free|four-fifths|adverse impact|fairness", readme)
    out["unfair discrimination and bias testing"] = ev("found" if bias_keys and bias_text else "partial" if bias_text else "gap",
                                                       "fairness metrics in results and README" if bias_keys and bias_text else
                                                       "discussed in README only" if bias_text else "no bias test for this model")
    perf = _found(r"auc|gini|deviance|mae|f1|precision|map|calibrat|predicted_over_actual", keys)
    drift = f'"{name}"' in mrc or _found(r"psi|drift", keys)
    out["validation, testing and monitoring"] = ev("found" if perf and in_ci and drift else "partial" if perf else "gap",
                                                   f"held-out metrics: {'yes' if perf else 'no'}; tests in CI: {', '.join(in_ci) or 'none'}; "
                                                   f"drift check: {'yes, in the model risk validation toolkit' if drift else 'no'}")
    expl = _found(r"\bshap\b|pred_contribs|decline reason|reason code|monoton", code + readme)
    out["explainability and transparency"] = ev("found" if expl else "gap", "per-decision reasons (SHAP or monotone reason codes)" if expl else "no per-decision explanation")
    out["third-party vendors"] = ev("found", "no vendor models or data: open-source libraries and public data only, named in README")
    notice = _found(r"decline reason|adverse action|reason code", readme)
    out["consumer notice and adverse decisions"] = ev("found" if notice else "gap",
                                                      "plain-language decline reasons" if notice else "no consumer-facing notice or reasons")
    out["regulator reporting and examination"] = ev("partial", "results, code and data are reproducible from the repo; no regulator-format report")
    return out
