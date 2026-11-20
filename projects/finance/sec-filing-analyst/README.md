# SEC Filing Analyst

Answers calculation questions about 10-K and 10-Q filings ("What is the FY2019 fixed asset turnover ratio for Activision Blizzard?") by finding each input number, checking that it is really on the page, and doing the math in Python. The model never does arithmetic.

It runs on top of the [Document Q&A](../../core/document-qa-platform/README.md) pipeline and retriever, and is scored by the same evaluation code, on the 50 FinanceBench questions of type "metrics-generated". Everything is free: Qwen2.5-3B answers and Llama-3.2-3B grades, both on CPU in GitHub Actions.

## How it works

```
question
   │
   ▼
plan       model returns JSON: line items needed (name, statement, fiscal year) + a formula
   │       e.g. a = revenue FY2019, b = PP&E FY2018, c = PP&E FY2019, formula a / ((b + c) / 2)
   ▼
find       for each item: search the filing (company/year filters), show the model 3 pages,
   │       it returns the value and which page it came from
   ▼
check      the value must be printed on that page (or another of the 3); if not, the item fails
   │       the table's scale ("In millions", "(Millions)", "$ in thousands") is read from the page text
   ▼
compute    formula evaluated with a small AST evaluator (+ - * / only), then converted to the
   │       unit the question asks for (USD millions, %, ratio)
   ▼
answer     result line, formula, every input with its page as [S#]
```

If the plan has no formula, any item can't be found on a page, or the formula fails, it falls back to the plain Q&A answer.

## Results

50 metric questions, same answer model, judge and grading rules as plain Q&A (refusals never count as correct, and a numeric answer has to be within 1% of the reference). From `results/summary_ollama_qwen2.5-3b.json`; earlier versions are in `results/v1/` and `results/v2/`.

| | Plain Q&A | Analyst v1 | Analyst v2 | Analyst v3 |
|---|---|---|---|---|
| Accuracy | **20%** | 18% | 14% | 18% |
| Wrong answers (not refusals) | **18%** | 30% | 30% | 22% |
| Answered by the calculator | | 16 | 10 | 10 |
| Calculator answers within 1% | | 31% | 30% | **40%** |
| Citations pointing at an evidence page | 54% | 49% | 52% | **57%** |

So on these 50 questions it does **not** beat plain Q&A. What I learned from the misses:

- **v1:** about 11 of the 16 calculator answers had the scale wrong. The model was asked whether a table was in thousands or millions and often guessed, and it also wrote "/ 1000" into formulas. v2 reads the scale from the page text, and conversion constants are stripped from formulas.
- **v2:** two answers came out as 0.0 because pages labeled "(Millions)" weren't recognised, so the numbers were treated as dollars. v3 handles that label. Now all 73 evidence pages for these questions get the right scale.
- **v3:** the remaining calculator misses come from picking the wrong number: the wrong line item (net cash from investing instead of capital expenditures) or the wrong year column. The page check catches made-up numbers, but not a real number from the wrong row. 61 of the 114 inputs in v1 were thrown out by the page check.
- The plan step works: in v3 only 2 of 50 questions got no usable plan. The find step is where questions drop out. In v3, 39 of 102 inputs passed the page check, so only 10 questions reached the calculator.

My conclusion: the structure is sound (when the inputs are right, the math and units are right), but a 3B model reading a full financial statement page is the weak link. The next things to try are a stronger free model (Llama 3.3 70B on Groq's free tier works with `LLM_PROVIDER=groq`) and table-aware extraction: parse the statement into rows and columns first, so the model picks a row label rather than reading raw text.

Latency is a median of 175 s per question on a 2-core CI runner, because each question makes about 4 model calls.

## Running it

```bash
export PYTHONPATH=../../../src
pytest -q test_analyst.py          # no model needed: scripted model replies, real retriever
ollama pull qwen2.5:3b && ollama pull llama3.2:3b
python run_eval.py --limit 5       # answer + grade 5 questions locally
```

The full run is `.github/workflows/sec-analyst-eval.yml` (10 parallel jobs). It starts when `run-request.txt` changes, and it commits the results back.
