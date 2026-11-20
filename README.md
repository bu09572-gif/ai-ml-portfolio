# AI/ML and Data Engineering Portfolio

Applied AI, machine learning and data engineering projects. Every project uses real public data and free, open-source tools, reports numbers I measured, and has tests that run in CI.

**How to browse:** click a project name below to open its folder. Each folder has its own README with the data, method, results and how to run it.

## Contents

- [Core AI platform](#core-ai-platform): 3 projects
- [Healthcare](#healthcare): 1 project
- [Finance](#finance): 4 projects
- [Insurance](#insurance): 2 projects
- [Tech / LLM](#tech--llm): 1 project
- [Data engineering](#data-engineering): 5 projects

## Core AI platform

Folder: [`projects/core`](projects/core)

| Project | What it does | Key result |
|---|---|---|
| [Document Q&A Platform for SEC Filings](projects/core/document-qa-platform) | Filing ingestion pipeline, Q&A with page citations, and an LLM evaluation harness that fails CI when quality drops | 84 real filings, 150 analyst questions; company/year filters double retrieval hit@5 |
| [Agentic RAG Copilot for SEC Filings](projects/core/agentic-rag-copilot) | Agent that searches, reads and calculates, with guardrails enforced in code | 20.7% vs 19.3% for plain Q&A; 34% vs 14% on open-ended questions |
| [PII Redaction Pipeline on Real Court Cases](projects/core/pii-redaction) | Trained NER model that masks personal identifiers in court judgments | 97.7% of direct identifiers fully masked, 90% precision |

## Healthcare

Folder: [`projects/healthcare`](projects/healthcare)

| Project | What it does | Key result |
|---|---|---|
| [ICD-10 Diagnosis Coding Assistant for Clinical Notes](projects/healthcare/icd10-coding-assistant) | Ranks ICD-10 codes for a clinical note, with the sentence that supports each code | MAP 0.326 on 250 expert-coded cases; 53% of top-5 suggestions correct |

In progress: Clinical LLM evaluation (accuracy, robustness and bias checks); Clinical note summarizer with de-identification; Prior authorization assistant over CMS coverage policies; Claim denial predictor and appeal drafter.

## Finance

Folder: [`projects/finance`](projects/finance)

| Project | What it does | Key result |
|---|---|---|
| [Real-Time Card Fraud Scoring with SHAP Explanations](projects/finance/realtime-fraud-scoring) | XGBoost fraud API that returns a score, an alert and the top 3 reasons | 73% of fraud caught at 80% precision; 9 ms median latency |
| [Explainable Credit Default Model with Fairness Checks](projects/finance/explainable-credit-risk) | Monotone XGBoost with plain-language decline reasons and four-fifths rule checks | AUC 0.776; every adverse impact ratio at or above 0.90 |
| [Model Risk Validation Toolkit](projects/finance/model-risk-checker) | SR 11-7 style validation: leakage, challenger, calibration, drift, monotonicity, fairness | Caught 6 of 6 injected faults and 2 real issues in the fraud model |
| [SEC Filing Analyst](projects/finance/sec-filing-analyst) | Finds each input number on its page and does the math in Python, never in the model | 40% of calculator answers within 1%; 18% overall vs 20% for plain Q&A, with the reasons |

In progress: Anti-money-laundering with graph ML.

## Insurance

Folder: [`projects/insurance`](projects/insurance)

| Project | What it does | Key result |
|---|---|---|
| [Motor Insurance Pricing with Discrimination-Free Premiums](projects/insurance/motor-pricing-fairness) | Frequency and severity pricing models, and what removing driver age really changes | 678k real policies; GBM Gini 0.361 vs GLM 0.279 |
| [Motor Claim Severity Model with Extreme-Value Large-Loss Analysis](projects/insurance/claims-severity-large-loss) | Claim size models below a cap, a Pareto tail fit above it, and a large-loss load | 1.8% of claims carry 34.5% of cost; tail shape 1.07 |

In progress: Underwriting assistant; AI governance report (state AI rules); Subrogation detection.

## Tech / LLM

Folder: [`projects/tech`](projects/tech)

| Project | What it does | Key result |
|---|---|---|
| [Text-to-SQL Assistant with Safe Execution and Self-Repair](projects/tech/text-to-sql) | Plain English to read-only SQL, with error-driven repair | 42.1% execution accuracy on all 1,534 BIRD dev questions with a free 3B model |

In progress: Multi-step agent with guardrails; LLM red-team suite; Search and recommendations; Production RAG with a cost dashboard.

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
