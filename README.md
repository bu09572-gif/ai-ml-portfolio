# AI / ML portfolio: shared foundations + sector projects

Every project in this repo runs on **real public data** and is scored by the same evaluation platform, so each one comes with measured numbers, not claims.

The repo is built in layers:

1. **Foundations** (this first release, all three are built and measured):
   - `src/docpipe`: **document data pipeline**. Real SEC filings in, quality-checked, retrieval-ready chunks out.
   - `src/docqa`: **document Q&A (RAG) with citations**. Search plus LLM answers that cite the exact filing page. The default model is a **free open-source model** (Qwen2.5-3B via Ollama, no API key), and Groq (free tier) and Claude are optional.
   - `src/evalkit`: **evaluation platform**. Retrieval metrics, LLM-graded answer accuracy, citation checks, cost and latency, and a CI regression gate.
2. **Sector projects** (healthcare, finance, tech/LLM, data engineering, insurance). Each reuses the foundations and gets its own folder and README as it is built.

## Results so far (measured on real data)

**Data:** [FinanceBench](https://github.com/patronus-ai/financebench) (Patronus AI, 2023): real 10-K/10-Q/8-K filings and 150 questions written by finance analysts, each with a gold answer and the evidence page. Paper: [arXiv 2311.11944](https://arxiv.org/abs/2311.11944).

**Pipeline run** (`evals/results/financebench/pipeline_run.json`): 84 filings from 32 companies (165.5 MB of PDFs), giving 12,013 pages and 37,785 chunks. All 9 data-quality checks pass. 0.77% of pages have no text layer. Extracting the text took 53 s on a 4-core machine.

**Data check:** all 189 of 189 expert evidence snippets appear on the page number our extractor assigned. This rules out an off-by-one page error that would silently corrupt every retrieval score.

**Retrieval ablation** (`evals/results/financebench/retrieval_summary.json`), 150 questions. "hit@k" means a gold evidence page is in the top k results:

| Configuration (each row adds one change) | hit@1 | hit@5 | hit@10 | MRR | right filing in top 5 |
|---|---|---|---|---|---|
| BM25 over all 84 filings | 0.053 | 0.113 | 0.160 | 0.082 | 0.513 |
| + one result per page (dedupe overlapping chunks) | 0.053 | 0.120 | 0.167 | 0.084 | 0.520 |
| + company and fiscal-year filters detected from the question | 0.113 | 0.233 | 0.313 | 0.160 | 0.800 |
| + finance vocabulary expansion ("capex" → "purchases of property, plant and equipment") | 0.107 | **0.307** | **0.413** | 0.194 | 0.760 |
| Upper reference: told which filing to search | 0.167 | 0.433 | 0.620 | 0.289 | 1.000 |

Metadata filtering doubled hit@5, and vocabulary expansion added another 7 points. Search takes about 14 ms per question on CPU.

Caveat: the vocabulary list was written from general accounting terms, but it is evaluated on the same 150 questions, so treat the last row as slightly optimistic until it is checked on the held-out 10k-question set.

**Answer accuracy (free model, all 150 questions):** this runs in GitHub Actions (`free-llm-eval` workflow) with Qwen2.5-3B answering and Llama-3.2-3B grading, both open-source on CPU at $0. The results are committed automatically to `evals/results/financebench/answers_summary.json` and copied into this table when the run finishes.

| Model (answers) | Judge | Accuracy (LLM judge) | Numeric match | Refusal rate | Answers with citations | Citation page precision | p50 latency | Cost |
|---|---|---|---|---|---|---|---|---|
| qwen2.5:3b | llama3.2:3b | *running* | | | | | | $0 |

Dense and hybrid retrieval rows are added by the `ci` workflow (`--dense`).

## How the three foundations fit together

```
FinanceBench PDFs ──► docpipe ─────────────────────────────► docqa ──────────────────────► evalkit
                                                             LLM: Ollama (free, default) |
                                                             Groq (free tier) | Claude
 (GitHub, 368 filings)  fetch (retries, idempotent)          BM25 index (scipy sparse)    retrieval hit@k / MRR
                        bronze: raw PDFs + sha256 manifest   company/year filters         ablation table
                        silver: page text (pypdfium2)        query expansion              Claude answers with
                        gold:   page-aware chunks            optional dense + RRF hybrid   answers vs gold
                        9 data-quality checks + run report   FastAPI /search /ask          LLM judge + numeric match
                        local disk or s3:// (DATA_ROOT)    page citations [S#]            cost, p50/p95 latency
                                                                                           CI regression gate
```

## Run it

In **GitHub Codespaces** (no local disk needed): click *Code → Codespaces → Create codespace*, then:

```bash
make pipeline      # download + process the 84 filings (about 1.5 min)
make eval          # retrieval ablation table (free, no API key)
make test          # unit tests
make api           # http://localhost:8000/docs
ollama pull qwen2.5:3b       # free local model (Ollama is installed when the Codespace is created)
make eval-answers LIMIT=20   # answers + grading with the free model, $0
```

**Choosing the LLM** (all optional; the default needs nothing):

| `LLM_PROVIDER` | Cost | Setup | Notes |
|---|---|---|---|
| `ollama` (default) | free | none | runs locally in Codespaces/Colab; `LLM_MODEL=llama3.2:3b` etc. |
| `groq` | free tier, no credit card | `GROQ_API_KEY` from console.groq.com | much faster; rate-limited |
| `anthropic` | paid | `ANTHROPIC_API_KEY` | native citation API; strongest answers |

`DATA_ROOT=s3://your-bucket/portfolio make pipeline` writes the same bronze/silver/gold layout to S3 (`pip install -e ".[aws]"`), and Athena can query the parquet files from there.

CI (`.github/workflows/ci.yml`) runs lint and tests on every push. It then runs the real-data pipeline and evaluation, and fails the build if hit@5, hit@10 or MRR drops more than 0.02 below `evals/baseline/financebench.json`. `free-llm-eval.yml` answers and grades all 150 questions with free local models across 10 parallel jobs, then commits the results.

## Who this is for

The foundations map onto these employers. Sector projects add their own lists.

| Foundation | Anchor companies | Also doing similar work |
|---|---|---|
| Document Q&A over filings | Bloomberg, S&P Global, Morgan Stanley | JPMorgan Chase, Goldman Sachs, Moody's, BlackRock, Capital One, American Express, Wells Fargo, Intuit, Fidelity, State Street, Accenture, PwC, McKinsey (QuantumBlack), BCG X, IBM |
| Evaluation platform | Scale AI, Databricks, Datadog | Anthropic, OpenAI, DataRobot, Microsoft, Google, Amazon/AWS, Salesforce, ServiceNow, Snowflake |
| Document data pipeline | Snowflake, Databricks, Capital One | S&P Global, Bloomberg, Fidelity, Goldman Sachs, Oracle, SAP, Accenture, Cognizant |

### Skills, tier 1: basics nearly every company expects
| Skill | Where |
|---|---|
| Python packaging, CLI tools, tests, lint | `pyproject.toml`, `tests/`, `Makefile`, ruff |
| SQL-ready columnar data (parquet), medallion layers | `src/docpipe/run.py`, `storage.py` |
| Data-quality checks with clear pass/fail | `src/docpipe/quality.py` |
| Retries, idempotent incremental loads | `sources/financebench.py`, `run.py` |
| REST API with validation (FastAPI, Pydantic) | `src/docqa/api.py` |
| CI/CD (GitHub Actions), cloud dev environment | `.github/workflows/ci.yml`, `.devcontainer/` |
| Cloud storage (S3 via fsspec) | `src/docpipe/storage.py` |
| Evaluation with held-out labels, honest reporting | `src/evalkit/` |

### Skills, tier 2: what LLM / GenAI teams ask for
| Skill | Where |
|---|---|
| RAG: chunking, BM25, metadata filtering, hybrid (RRF) | `src/docpipe/chunk.py`, `src/docqa/retrieve.py` |
| Grounded answers with citations (source tags; Claude citations API) | `src/docqa/answer.py` |
| Provider-agnostic LLM layer (open-source local models, hosted APIs) | `src/docqa/llm.py` |
| LLM-as-judge plus deterministic metrics | `src/evalkit/metrics.py` |
| Ablation studies, retrieval metrics (hit@k, MRR) | `src/evalkit/run.py` |
| Eval regression gates in CI | `src/evalkit/gate.py` |
| Cost and latency tracking per request | `Answer.cost_usd`, `answers_summary.json` |
| Refusal and fallback handling | `answer.py` (refusal stop reason, server-side fallback) |

**Resume keywords:** Python, RAG, BM25, hybrid retrieval, reciprocal rank fusion, Ollama, open-source LLMs (Qwen, Llama), Claude API, citations, LLM evaluation, LLM-as-judge, FastAPI, Pydantic, parquet, medallion architecture, data-quality checks, GitHub Actions, Codespaces, AWS S3, Athena, pytest.

## How to explain this in an interview

1. **Problem:** "Analysts need answers from 100-page SEC filings, and a wrong number is worse than no answer. I built the pipeline, the Q&A, and the evaluation that proves how good it is."
2. **Data:** "I used FinanceBench, real filings with 150 expert questions and the evidence page for each. Before trusting any score, I checked that all 189 evidence snippets land on the page number my extractor assigns."
3. **Ablation:** "Plain BM25 found the evidence page in the top 5 only 11% of the time. Detecting the company and fiscal year and filtering on them doubled that. A small finance glossary got it to 31%. Each change was measured separately."
4. **Honesty:** "The glossary was evaluated on the same questions, so I say that number is optimistic. And the gap to the 43% you get when the right filing is known tells me where the next gain is: picking the right filing."
5. **Production:** "The eval runs in CI and fails the build if retrieval drops more than 2 points, so a 'small refactor' can't quietly make answers worse."

## Repo layout
```
src/docpipe/     pipeline: sources/, extract.py, chunk.py, quality.py, storage.py, run.py
src/docqa/       retrieve.py (BM25, filters, hybrid), llm.py (Ollama / Groq / Claude), answer.py (cited answers), api.py
src/evalkit/     dataset.py (common eval format), metrics.py, run.py, gate.py
evals/           baseline/ (committed gate thresholds), results/ (latest measured numbers)
tests/           unit tests
```

## Data and license notes
FinanceBench files are downloaded at run time from the authors' GitHub repo and are not stored in this repo. Cite: Islam et al., *FinanceBench: A New Benchmark for Financial Question Answering*, 2023.
