# Agentic RAG Copilot for SEC Filings

An agent that answers questions about SEC filings by choosing its own next step: search, read a page, calculate, or answer. Code-level guardrails decide whether an answer is allowed out. It uses the same retriever, filings and grading as the [Document Q&A](../../core/document-qa-platform/README.md) project, so the two can be compared directly on all 150 FinanceBench questions. Free models only: Qwen2.5-3B runs the agent, Llama-3.2-3B grades, both on CPU in GitHub Actions.

## How it works

Each turn the model replies with one JSON action and gets the result back:

| Action | What it gets back |
|---|---|
| `search(query)` | 5 page previews labeled P1, P2, ... (BM25 with the question's company/year filters) |
| `read(P#)` | the full text of that page |
| `calculate(expr)` | the result, computed in Python (digits and + - * / ( ) only) |
| `answer(text)` | the final answer, citing pages as [P#] |

Guardrails are enforced in code, not left to the prompt:

- at most 8 steps, then one last call where only an answer is accepted
- every cited [P#] must be a page the agent actually read
- every number in the answer must be on a page it read or come out of `calculate` (rounding and millions/billions changes are allowed)
- a rejected answer goes back to the agent once with the reason. If the second try also fails, it becomes "not enough information"
- the prompt is kept under about 18k characters: older page reads are shortened, and a model error ends only that question

## Results

150 questions, same grading rules as plain Q&A (refusals never count as correct, and numeric answers must be within 1% of the reference). From `results/summary_ollama_qwen2.5-3b.json`; earlier versions are in `results/v1/` and `results/v2/`.

| | Plain Q&A | Agent v1 | Agent v2 | Agent v3 |
|---|---|---|---|---|
| Accuracy (all 150) | 19.3% | 5.3%* | 6.0% | **20.7%** |
| domain-relevant | 14% | | 8% | **34%** |
| metrics-generated (calculations) | **20%** | | 2% | 4% |
| novel-generated | 24% | | 8% | 24% |
| Wrong answers (not refusals) | **13.3%** | 2.7%* | 12.7% | 30.0% |
| "Not enough information" | 67.3% | 92%* | 81.3% | 49.3% |
| Citations pointing at an evidence page | **35.5%** | | 32.1% | 14.8% |
| Median latency | 42 s | | 176 s | 131 s |

\* v1 finished only 75 questions: 5 of the 10 CI jobs crashed when the prompt outgrew the model's context window.

What changed between versions, and why:

- **v1:** the agent almost never answered (64 of 75 ran out of steps), and long transcripts crashed the model server. 
- **v2:** the prompt is now bounded, errors are caught per question, and the agent is nudged to answer near the step limit. It still scored 6%, because 50 answers were rejected for having no [P#] citation. The 3B model often writes a correct answer and forgets the tags.
- **v3:** when an answer has no citations, code adds the pages the agent read that contain the answer's numbers (48 answers). At the step limit there is one forced answer call.

What the v3 numbers say:

- It beats plain Q&A overall (20.7% vs 19.3%). The gain is all in **domain-relevant** questions (34% vs 14%), which need reading around a filing more than one lookup (for example, what drove a change in margins). Most of these answers have no number to check, so the small judge model decides the grade, and that gain is less certain than the numeric results.
- It is much worse at **calculations** (4% vs 20%). The model reads and calculates a lot (234 `calculate` calls) but combines the wrong numbers.
- It answers more and is wrong more: 30% wrong vs 13%. 18 answers were stopped because they used numbers that were on no page it read, so the guardrail does catch made-up numbers. But a real number from the wrong row still gets through.
- Citation quality dropped (14.8%). Adding citations by matching numbers points at pages the agent read, not always the evidence page. It read an evidence page on only 23% of questions.

So neither approach wins everywhere. The obvious next step is a router: the agent for open-ended questions, and the plain pipeline or the [SEC Filing Analyst](../../finance/sec-filing-analyst/README.md) for calculations. I'd also run the same loop on a stronger free model (Groq, `LLM_PROVIDER=groq`). I haven't measured either yet.

## Running it

```bash
export PYTHONPATH=../../../src
pytest -q test_copilot.py          # scripted model: guardrails, retries, step limit, prompt size
ollama pull qwen2.5:3b && ollama pull llama3.2:3b
python run_eval.py --limit 5
```

The full run is `.github/workflows/copilot-eval.yml` (10 parallel CI jobs). It starts when `run-request.txt` changes, and it commits the results back. Each row of `results/agent_*.jsonl` holds the agent's full step trace.
