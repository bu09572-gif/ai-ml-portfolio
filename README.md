# AI/ML and Data Engineering Portfolio

Applied AI, machine learning and data engineering projects. Every project uses real public data and free, open-source tools, reports numbers I measured, and has tests that run in CI.

**How to browse:** click a project name below to open its folder. Each folder has its own README with the data, method, results and how to run it.

## Featured projects

Each of these also has its own repository with tests in CI.

| Project | Area | Repository |
|---|---|---|
| Production RAG over SEC Filings, with a Golden-Set Evaluation Harness | Core AI | [document-qa-platform](https://github.com/bu09572-gif/document-qa-platform) |
| Clinical Note Summarizer with De-identification | Healthcare | [clinical-note-summarizer](https://github.com/bu09572-gif/clinical-note-summarizer) |
| Real-Time Card Fraud Scoring: an End-to-End MLOps Pipeline | Finance | [realtime-fraud-scoring](https://github.com/bu09572-gif/realtime-fraud-scoring) |
| MCP Agent with Guardrails and Human Approval | Tech / LLM | [llm-agent-guardrails](https://github.com/bu09572-gif/llm-agent-guardrails) |
| Streaming Flight Lakehouse (Kafka, Spark, Delta Lake) | Data engineering | [streaming-lakehouse](https://github.com/bu09572-gif/streaming-lakehouse) |
| Subrogation Opportunity Detection from Crash Reports | Insurance | [subrogation-detection](https://github.com/bu09572-gif/subrogation-detection) |

## Contents

- [Core AI platform](#core-ai-platform): 3 projects
- [Healthcare](#healthcare): 5 projects
- [Finance](#finance): 5 projects
- [Insurance](#insurance): 5 projects
- [Tech / LLM](#tech--llm): 5 projects
- [Data engineering](#data-engineering): 5 projects

## Core AI platform

Folder: [`projects/core`](projects/core)

| Project | What it does | Key result |
|---|---|---|
| [Production RAG over SEC Filings, with a Golden-Set Evaluation Harness](projects/core/document-qa-platform) | RAG over 84 real filings with page citations: filing routing, hybrid BM25 + dense search, a 50-question golden set, manual audit of the LLM judge, failure analysis, Docker service with Prometheus metrics | Accuracy 20% → 34% on the golden set; retrieval failures 34 → 17; the LLM judge alone would have said 82% |
| [Agentic RAG Copilot for SEC Filings](projects/core/agentic-rag-copilot) | Agent that searches, reads and calculates, with guardrails enforced in code | 20.7% vs 19.3% for plain Q&A; 34% vs 14% on open-ended questions |
| [PII Redaction Pipeline on Real Court Cases](projects/core/pii-redaction) | Trained NER model that masks personal identifiers in court judgments | 97.7% of direct identifiers fully masked, 90% precision |

## Healthcare

Folder: [`projects/healthcare`](projects/healthcare)

| Project | What it does | Key result |
|---|---|---|
| [ICD-10 Diagnosis Coding Assistant for Clinical Notes](projects/healthcare/icd10-coding-assistant) | Ranks ICD-10 codes for a clinical note, with the sentence that supports each code | MAP 0.326 on 250 expert-coded cases; 53% of top-5 suggestions correct |
| [Clinical LLM Evaluation: Accuracy, Robustness and Bias Checks](projects/healthcare/clinical-llm-safety-eval) | Two free 3B models on MedQA and PubMedQA, with option-order, patient-sex-swap and abstention checks | Llama 3.2 3B: 54.9% MedQA; 29% of answers change when options are reordered vs 5% when patient sex is swapped |
| [Clinical Note Summarizer with De-identification](projects/healthcare/clinical-note-summarizer) | Removes identifiers before a free local model summarizes 300 real clinical notes; checks leaks, invented numbers and an LLM judge | Identifier repeats fall from 22% to 4% of notes; the 3B judge's flags are wrong 81% of the time |
| [Prior Authorization Assistant over Medicare Coverage Policies](projects/healthcare/prior-auth-assistant) | Finds the right Medicare coverage policy (LCD) for a clinical note, turns it into a checklist of quoted criteria, and checks the note with cited evidence | Right policy in the top 10 for 82% of notes; a free 3B model's met/not-met verdicts fail the evidence check, so they go to a reviewer |
| [Claim Denial Analytics and Grounded Appeal Letters](projects/healthcare/claim-denial-appeals) | Denial and appeal analytics on the CMS 2024 Transparency in Coverage file (386M claims), plus a free local model drafting appeal letters from quoted chart evidence | Only 1.6 appeals per 1,000 denials, yet 43% win; grounded letters keep 98% of quotes real vs 75% when the model reads the whole chart |

## Finance

Folder: [`projects/finance`](projects/finance)

| Project | What it does | Key result |
|---|---|---|
| [Real-Time Card Fraud Scoring: an End-to-End MLOps Pipeline](projects/finance/realtime-fraud-scoring) | XGBoost fraud API with SHAP reasons, shipped end to end: MLflow registry and promotion gate, Docker, Prometheus and Grafana monitoring, drift checks, load and cost tests, CI/CD to a container registry | 80% alert precision; 6.4 ms median latency; $0.003 per million transactions in batches; score drift stayed low while input features drifted every hour |
| [Explainable Credit Default Model with Fairness Checks](projects/finance/explainable-credit-risk) | Monotone XGBoost with plain-language decline reasons and four-fifths rule checks | AUC 0.776; every adverse impact ratio at or above 0.90 |
| [Model Risk Validation Toolkit](projects/finance/model-risk-checker) | SR 11-7 style validation: leakage, challenger, calibration, drift, monotonicity, fairness | Caught 6 of 6 injected faults and 2 real issues in the fraud model |
| [SEC Filing Analyst](projects/finance/sec-filing-analyst) | Finds each input number on its page and does the math in Python, never in the model | 40% of calculator answers within 1%; 18% overall vs 20% for plain Q&A, with the reasons |
| [Anti-Money-Laundering Detection on a Real Bitcoin Transaction Graph](projects/finance/graph-aml) | Flags illicit transactions with graph features, a GCN, and a test of what happens when criminal behaviour changes | F1 0.826 (precision 0.96); recall falls from 86% to 2% after a dark market shutdown |

## Insurance

Folder: [`projects/insurance`](projects/insurance)

| Project | What it does | Key result |
|---|---|---|
| [Motor Insurance Pricing with Discrimination-Free Premiums](projects/insurance/motor-pricing-fairness) | Frequency and severity pricing models, and what removing driver age really changes | 678k real policies; GBM Gini 0.361 vs GLM 0.279 |
| [Motor Claim Severity Model with Extreme-Value Large-Loss Analysis](projects/insurance/claims-severity-large-loss) | Claim size models below a cap, a Pareto tail fit above it, and a large-loss load | 1.8% of claims carry 34.5% of cost; tail shape 1.07 |
| [AI Governance Report for Insurance Models under State AI Rules](projects/insurance/ai-governance-report) | Requirements register from the NAIC bulletin, New York CL 7 and Colorado Reg. 10-1-1, and an evidence check of my models against it | 118 requirements, 34 of 40 theme tags right; 30 jurisdictions adopted the NAIC bulletin |
| [Life Underwriting Mortality Model on Real Health Exams](projects/insurance/life-underwriting-risk) | 8-year death risk from the application, a nurse visit or a blood test, on NHANES with National Death Index records; what fast-tracking without labs costs | AUC 0.80 application only vs 0.83 with labs; calibration checked by race and income |
| [Subrogation Opportunity Detection from Crash Reports](projects/insurance/subrogation-detection) | Flags recoverable auto claims at first notice of loss, on 147,290 claims built from 2023 NYPD crash reports, against the classic parked/rear-ended rule | 96% vs 89% precision at the rule's volume; 63% of recoverable claims found at 30% referred vs 37% |

## Tech / LLM

Folder: [`projects/tech`](projects/tech)

| Project | What it does | Key result |
|---|---|---|
| [Text-to-SQL Assistant with Safe Execution and Self-Repair](projects/tech/text-to-sql) | Plain English to read-only SQL, with error-driven repair | 42.1% execution accuracy on all 1,534 BIRD dev questions with a free 3B model |
| [Movie Recommendations: Retrieval, Ranking and Honest Offline Evaluation](projects/tech/movie-recommendations) | Popularity, kNN, implicit ALS, EASE and a two-stage ranker on MovieLens 1M, ranking each user's next movie against the full catalogue | EASE on the last 5 movies: 19.1% Hit@10 vs 10.0% for full-history EASE and 17.9% for the ranker |
| [LLM Red-Team Suite: Jailbreak Attacks, Defences and Over-Refusal](projects/tech/llm-red-team) | JailbreakBench requests x 4 jailbreak patterns x 3 defences on two free 3B models, judged by Llama Guard, with over-refusal measured on matched harmless requests | Prefix injection jailbreaks Qwen2.5 3B 79% of the time; self-check cuts attacks 40% to 5% but refuses 42% of harmless requests |
| [Production RAG: Cost, Latency and Quality Dashboard](projects/tech/rag-cost-dashboard) | Every serving knob (3 free models, k = 3/5/10 chunks, context compression, a small-to-large router) measured on FinanceBench for tokens, CPU seconds, p95 latency and accuracy, with a Streamlit cost dashboard | Compression cuts tokens 40% and lifted the best setting to 22.5% accuracy; the router cost more and scored lower because the small model refuses 58-81% of the time |
| [MCP Agent with Guardrails and Human Approval](projects/tech/agent-injection-guardrails) | A finance agent that calls tools over MCP through a guardrail gateway (budgets, allowlist, schema checks, rules in code, human approval, injection filter), with an approval inbox API and a 1,080-episode evaluation against real attacker text hidden in SEC filings | Without guardrails a poisoned filing hijacks the agent 78-82% of the time; with them 0 of 216 attacks succeed, at no accuracy cost |

## Data engineering

Folder: [`projects/data-engineering`](projects/data-engineering)

| Project | What it does | Key result |
|---|---|---|
| [Flight Operations Warehouse (dbt + DuckDB)](projects/data-engineering/flight-ops-dbt-warehouse) | Star schema over real flights with dbt tests | 336,776 flights; 13 models and 17 tests, all passing in CI |
| [Flight Data Quality Monitor](projects/data-engineering/flight-data-quality-monitor) | Daily anomaly checks, tested by breaking the data on purpose | 480 injected-fault trials; large problems caught every time |
| [Flight Status CDC Pipeline with SCD2 History](projects/data-engineering/flight-status-cdc) | Change capture from an ops database into current and history tables | 1,000,898 changes replayed; all reconciliation checks at 0 |
| [Streaming Flight Lakehouse (Kafka, Spark, Delta Lake)](projects/data-engineering/flight-streaming-lakehouse) | Bronze, silver and gold Delta tables from a Kafka stream, with a crash test | Killed with kill -9 mid-stream; after restart all 4 checks are 0 |
| [Daily Flight Pipeline on Airflow](projects/data-engineering/flight-pipeline-airflow) | Extract, check, then publish or quarantine each day | 365 daily runs, 0 failed, 12 bad days quarantined |

## Shared code and data

- `src/`: the document Q&A platform (`docpipe`, `docqa`, `evalkit`) and shared flight data code (`flightsdata`), imported by several projects.
- Datasets hosted outside GitHub are mirrored into this repo's releases (`data-v1`, `data-v2`) by `tools/` in GitHub Actions, with SHA-256 checksums, so every run uses the same copy.
- LLM evaluations run free local models (Qwen2.5 and Llama 3.2 through Ollama) in GitHub Actions and commit their results.

## CI

- `ci.yml`: lint and every project's tests, then the document Q&A pipeline and retrieval eval on the real filings. The build fails if hit@5, hit@10 or MRR drops more than 0.02 below `evals/baseline/financebench.json`.
- `data-engineering.yml`: dbt build and tests, the quality monitor, the CDC replay, the Kafka/Spark lakehouse with a crash test, and an Airflow backfill.
- `free-llm-eval.yml`: answers and grades all 150 questions with the local models, split across 10 parallel jobs, then commits the results. Starts from the Actions tab or when `evals/run-request.txt` changes.

## Layout

```
projects/<area>/<project>/   one folder per project, each with README, code, tests and results
                             areas: core, healthcare, finance, insurance, tech, data-engineering
src/                         shared code: docpipe, docqa, evalkit (document Q&A platform), flightsdata
tools/                       mirrors public datasets into GitHub releases
evals/                       document Q&A baseline and results
tests/                       document Q&A platform tests
```
