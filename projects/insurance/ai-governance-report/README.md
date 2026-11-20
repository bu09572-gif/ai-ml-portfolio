# AI Governance Report for Insurance Models under State AI Rules

This project reads the main US rules on how insurers may use AI, turns them into a requirements register, and checks my own pricing and claims models in this repo against it. The output is a governance report that shows, for each model and each control theme, the evidence I have and the gaps. The point is to treat compliance as something you can test, like model accuracy.

## The rules

| Rule | What it covers |
|---|---|
| NAIC Model Bulletin, "Use of Artificial Intelligence Systems by Insurers" | A written AI program, governance, risk controls, third-party oversight and what examiners may ask for. Adopted by NAIC in December 2023 |
| New York DFS Insurance Circular Letter No. 7 (2024) | AI and external consumer data in underwriting and pricing, with a strong focus on testing for unfair discrimination |
| Colorado Amended Regulation 10-1-1 (3 CCR 702-10) | Governance and risk management for life, private passenger auto and health insurers that use external data and models. The 2025 amendment added auto insurers, which covers my motor pricing model |

NAIC's own map, as of November 13, 2026, lists **30 jurisdictions** that have adopted the bulletin (28 states, DC and Puerto Rico), and California, Colorado and New York with their own rules. `data.py` reads this list from the map.

**Sources, and two caveats.** All files are mirrored into this repo's `data-v2` release by `tools/fetch_urls.py`, with SHA-256 checks.
- NAIC's link to the adopted bulletin PDF returns 404. I use NAIC's October 13, 2023 committee draft, the version that went to the adoption vote, so a few words may differ from the final text.
- The Colorado text is the amended regulation as posted by a law firm's insurance blog, because the Colorado Division of Insurance site refused automated downloads. The New York letter is the DFS web page itself.

## How it works

1. **Requirements register** (`governance.py`). I split each text into sentences and keep the obligations: sentences with *shall, must, should, required* or *expected to*. That gives 118 requirements: 31 NAIC, 54 New York, 33 Colorado, each with its page.
2. **Themes.** Keyword rules tag each requirement with up to ten control themes, such as "unfair discrimination and bias testing" or "third-party vendors". 27 requirements match no theme. Most of these are about scope and dates.
   - I checked 40 random tags by hand, and **34 of 40 were right**. The 6 errors come from words used in another sense. Examples are "monitoring" in a sentence about who is responsible, "adverse" in "adverse consumer outcomes" rather than an adverse decision, and "reporting" errors to a vendor rather than to the regulator.
3. **Evidence check.** For each model folder, `evidence()` looks for what each theme asks for:
   - README sections for data, models, results and limits.
   - A pinned SHA-256 on the data.
   - Fairness metrics in the results files.
   - Held-out metrics, and tests that CI actually runs.
   - Whether the [model risk validation toolkit](../../finance/model-risk-checker) checks the model for drift.
   - Per-decision reasons, and consumer-facing decline reasons.

   Each theme is marked found, partial or gap, and the reason is written out.

## Results

From `results/governance_report.md` and `results/summary.json`.

**Requirement sentences per theme:**

| Theme | NAIC | New York | Colorado |
|---|---|---|---|
| governance and accountability | 8 | 9 | 6 |
| written program and policies | 14 | 5 | 6 |
| data quality and sources | 0 | 36 | 9 |
| unfair discrimination and bias testing | 1 | 7 | 2 |
| validation, testing and monitoring | 3 | 7 | 1 |
| consumer notice and adverse decisions | 5 | 9 | 2 |
| regulator reporting and examination | 1 | 4 | 9 |

The three rules stress different things:
- NAIC is mostly about having a written program and governance.
- New York is about external data and proving it does not discriminate.
- Colorado is about governance, plus reporting and attesting to the Division.

**My models against the themes:**

| Theme | Motor pricing | Claim severity | Credit (comparison) | Fraud (comparison) |
|---|---|---|---|---|
| governance and accountability | **gap** | **gap** | **gap** | **gap** |
| written program and policies | **gap** | **gap** | **gap** | **gap** |
| inventory and documentation | found | partial | found | found |
| data quality and sources | found | found | found | found |
| unfair discrimination and bias testing | found | **gap** | found | **gap** |
| validation, testing and monitoring | partial | partial | found | found |
| explainability and transparency | **gap** | **gap** | found | found |
| third-party vendors | found | found | found | found |
| consumer notice and adverse decisions | **gap** | **gap** | found | **gap** |
| regulator reporting and examination | partial | partial | partial | partial |

What I take from it:
- **The technical evidence is mostly there. The organisational evidence is not.** Every model has documented, checksummed data and held-out results with tests in CI. None has an owner, an approval record or a written AI program. Those are what NAIC and Colorado ask for first. A real insurer would close these gaps with a model inventory, named owners and a sign-off step, not with code.
- **The motor pricing model is the one Colorado now covers, and it has three gaps:**
  1. No drift monitoring.
  2. No per-policy explanation of the premium.
  3. No consumer notice.

  My next steps would be to add it to the model risk validation toolkit, return the GLM factor for each rating variable with each price, and draft the adverse-decision notice Colorado describes.
- **The claim severity model has no bias test.** The NAIC bulletin covers claims models too, so it would need the same fairness checks as the pricing model.

## Limits

- Keyword tagging is a first pass (34 of 40 right). A compliance team would review the register, not trust it as is.
- The evidence check looks for artefacts, not their quality. It can tell that fairness metrics exist, not that they are good enough. This is not legal advice.
- I only cover three texts. Each state that adopted the bulletin can add its own changes.

## Running it

```bash
pip install pypdfium2 requests
python run.py            # a few seconds
pytest -q test_governance.py
```
