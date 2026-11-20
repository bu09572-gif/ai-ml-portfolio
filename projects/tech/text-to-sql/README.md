# Text-to-SQL Assistant with Safe Execution and Self-Repair

This assistant turns a question in plain English into a SQLite query and runs it read-only. If SQLite returns an error, the assistant sends the error back to the model to fix. I scored it on all 1,534 questions of the BIRD dev set, using a free 3B model on CPU.

## Data

The BIRD dev set (Li et al., 2023; CC BY-SA 4.0) has 1,534 questions written by people over 11 real databases: California schools, card games, a Stack Exchange dump, European football, Formula 1, Czech banking, hospital lab tests and more. Each question comes with a reference SQL query, a difficulty label and an "evidence" hint. A hint is domain knowledge a database user would know, for example "eligible free rate = Free Meal Count / Enrollment". The databases are large and messy: column names have spaces and symbols, codes are cryptic, and one table is called `order`. `data.py` downloads the official zip and checks its SHA-256.

## How it works

1. **Schema prompt.** For each table, the prompt gets a `CREATE TABLE` line with types, primary and foreign keys, and up to 3 example values per text column. That way the model can see that a column holds `'M'` and not `'Male'`. Names that need quoting are quoted. If a schema is too large for the 8k context, the example values are dropped.
2. **Generate.** The model writes one query, given the schema, the question and, optionally, the hint.
3. **Execute safely.** The database file is opened read-only (`mode=ro`). An SQLite authorizer allows only reads and functions, so `DELETE`, `DROP`, `ATTACH` and `PRAGMA` writes are refused even if the model writes them. Each query has a 30 s time limit and returns at most 10,000 rows.
4. **Repair.** If SQLite raises an error, the query and the error go back to the model, up to 2 more times.

**Scoring** uses BIRD's execution accuracy: a query is correct if it returns the same set of rows as the reference query. Row order and column names are ignored.

## Results

All 1,534 dev questions, with Qwen2.5-Coder-3B through Ollama on CPU GitHub Actions runners (20 parallel jobs). One reference query hit the 120 s time limit and is left out, so 1,533 are scored. From `results/summary_qwen2.5-coder-3b.json`.

| | With hint | Without hint |
|---|---|---|
| **Execution accuracy** | **42.1%** | 31.7% |
| First try (no repair) | 38.8% | 30.0% |
| simple (925 questions) | 50.9% | 38.8% |
| moderate (464) | 29.5% | 21.6% |
| challenging (144) | 26.4% | 18.8% |
| First query raised an error | 32.3% | 29.9% |
| Final query still raised an error | 25.0% | 24.3% |
| Median time per question | 7.8 s | 5.9 s |

By database, with hint: student_club 63.9%, superhero 57.4%, european_football_2 55.8%, codebase_community 51.9%, toxicology 44.1%, card_games 43.5%, debit_card_specializing 35.9%, formula_1 33.3%, financial 27.4%, thrombosis_prediction 19.0%, california_schools 16.9%.

What the numbers say:

- **The hint is worth 10 points.** Without it, the model has to guess what terms like "eligible free rate" or "SAT excellence rate" mean.
- **Repair adds 3.4 points.** 495 questions needed a second try. 112 of those ended with a query that ran, and 52 of those were correct. Fixing an error is not the same as fixing the answer.
- **Schema linking is the main weakness.** Of the 383 final queries that still failed, 356 failed on "no such column". The model picks the wrong table alias, or a column that doesn't exist. The two worst databases, california_schools and thrombosis_prediction, have the longest and most cryptic column names (`"Free Meal Count (K-12)"`, `"T-CHO"`). Another 504 queries ran but returned the wrong rows.
- **One bug cost a whole database.** The first full run scored 0% on `financial`. Its table is called `order`, and my schema code didn't quote it, so every question there crashed before the model was called. I fixed the quoting, added a test, and re-ran only that database. The numbers above include the re-run, and the first run's summary is in git history.

What I'd try next, in order: show the model only the tables and columns relevant to the question (schema linking with BM25 or embeddings), add a few worked examples from the BIRD train set, and run a larger free model through Groq (`LLM_PROVIDER=groq`).

## Running it

```bash
export PYTHONPATH=../../../src
pytest -q test_sqlagent.py         # small SQLite file and a scripted model: read-only guard, time limit, repair loop, scoring
python data.py                     # downloads and unpacks the BIRD dev zip
ollama pull qwen2.5-coder:3b
LLM_MODEL=qwen2.5-coder:3b python run_eval.py --limit 5
```

The full run is `.github/workflows/text-to-sql-eval.yml`. It starts when `run-request.txt` changes. A line `db=<name>` in that file re-runs only one database and merges the result into the existing results.
