# Prior Authorization Assistant over Medicare Coverage Policies

This project tests whether a free local model can do the paperwork side of prior authorization. Given a clinical note, it finds the right Medicare coverage policy, turns the policy into a checklist of quoted criteria, and checks the note against each one with a quote as evidence.

Finding the policy works well. Building the checklist mostly works. **Checking the note does not work at 3B size.** Every verdict has to cite the note, and that rule is what shows the failure.

## Data

- **CMS Medicare Coverage Database export:** 952 current Local Coverage Determinations (LCDs), the policies Medicare contractors use to decide what is "reasonable and necessary". Each one has coverage indications and documentation requirements, and some list HCPCS codes. It's a weekly CMS file, so I pinned the 2026-11-16 copy in this repo's `data-v2` release (SHA-256 in `data.py`).
- **MTSamples:** real transcribed procedure and consult notes. I picked 39 notes for six services on CMS's hospital outpatient prior-authorization list: blepharoplasty, botulinum toxin, varicose veins, cervical fusion, facet joint injections and sleep testing.

## How it works

1. **Find the policy** (`retrieval.py`): BM25 over each LCD's title and coverage text.
2. **Build the checklist** (`criteria.py`, `extract`):
   - Qwen2.5 3B reads each policy in pieces of about 350 words and copies out every coverage condition word for word.
   - A condition that isn't in the policy text is dropped, so every checklist item can be traced to the policy.
3. **Check the note** (`check`): for the 8 criteria closest to the note, the model answers met, not met or not documented, and must cite the note.
   - **Run 1:** the model writes a quote. A quote not found in the note turns the answer into "not documented".
   - **Run 2:** the model sees the 12 note sentences closest to the criterion, numbered, and cites their numbers. The evidence is then word for word from the note by construction.
4. **Control:** each note is also checked against an unrelated policy (a cervical fusion note against blepharoplasty criteria). Nothing there should come back met.

## Results

### Finding the policy (`results/retrieval.json`)

| | hit@1 | hit@5 | hit@10 |
|---|---|---|---|
| 1,789 HCPCS code descriptions, title only | 0.238 | 0.425 | 0.468 |
| 1,789 HCPCS code descriptions, title + coverage text | **0.617** | **0.757** | **0.828** |
| 39 clinical notes, title only | 0.256 | 0.462 | 0.487 |
| 39 clinical notes, title + coverage text | **0.462** | **0.692** | **0.821** |

Indexing the policy text, not just the title, nearly doubles hit@1. For 82% of notes the right policy is in the top 10, which is good enough for a person to pick from.

### Building the checklist (`results/criteria.json`)

- 306 criteria from six policies. **86.6% are word for word in the policy**; the rest were paraphrased or made up and were dropped.
- **Problem:** many items are exclusions ("surgeries performed solely for cosmetic reasons are not covered"), not things a note can show. In 9 of the 29 cited verdicts I reviewed below, the model read one of these backwards. A checklist needs a step that separates requirements from exclusions.

### Checking the note (296 criterion checks per pairing)

| | Run 1, model writes a quote | Run 2, model cites sentence numbers |
|---|---|---|
| verdicts with real note evidence, matched policy | 1.2% | 10.7% |
| final: met | 0.7% | 1.0% |
| final: not met | 0.3% | 8.8% |
| final: not documented | 99% | 90% |
| model said "not met" before the evidence rule (matched) | 84% | 91% |
| model said "not met" on the **unrelated** policy | 68% | 70% |
| median time per check (CPU) | 9.8 s | 6.0 s |

What I learned:
- **The model mostly doesn't look.** It answers "not met" 91% of the time on the right policy and 70% on an unrelated one. It almost never says "not documented", the honest answer for an unrelated policy.
- **The evidence rule catches it.**
  - In run 1 the model wrote a quote for 4 of 296 checks. Every other verdict had no evidence, and the rule turned it into "not documented".
  - Numbered sentences made citing easy: verdicts with evidence rose from 1% to 11%.
- **The cited verdicts are still mostly wrong.** I read all 29 matched verdicts with evidence in run 2:
  - About 10 are clearly wrong: the cited sentence *shows* the criterion is met. For example, "There were 306 apneas and hypopnea with apnea/hypopnea index 76" was cited correctly, but "symptoms of obstructive sleep apnea with snoring" was marked not met for a sleep-symptoms criterion.
  - About 9 are exclusion rules read backwards.
  - About 5 are correct, and about 5 I couldn't decide.
- **So what's usable:** policy retrieval plus a checklist quoted from the policy, with the matching note sentences highlighted for a reviewer. The automated met/not-met verdict is not reliable at this size. In a real system it needs a much stronger model, and a test like this one before anyone trusts it.

## Limits

- 39 notes, six services, one model. MTSamples are teaching transcriptions, not full charts: a real prior auth packet has many documents.
- My review of the 29 cited verdicts is one reader's judgment, not a clinical review.
- CMS's outpatient prior-auth list applies to hospital outpatient departments. Medicare Advantage and commercial plans have their own criteria.

## Run it

```bash
python retrieval.py                  # policy retrieval test, CPU, about a minute
pytest -q test_prior_auth.py         # unit tests, no model needed
# checklist + checks: change run-request.txt; GitHub Actions (prior-auth-eval) runs 10 shards and commits results/
# run 1 results are kept in results/run1_free_quote/
```
