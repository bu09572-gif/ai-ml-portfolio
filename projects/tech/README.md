# Tech / LLM projects

| Project | What it does | Key result |
|---|---|---|
| [Text-to-SQL Assistant with Safe Execution and Self-Repair](text-to-sql) | Plain English to read-only SQL, with error-driven repair | 42.1% execution accuracy on all 1,534 BIRD dev questions with a free 3B model |
| [Movie Recommendations: Retrieval, Ranking and Honest Offline Evaluation](movie-recommendations) | Popularity, kNN, implicit ALS, EASE and a two-stage ranker on MovieLens 1M, ranking each user's next movie against the full catalogue | EASE on the last 5 movies: 19.1% Hit@10 vs 10.0% for full-history EASE and 17.9% for the ranker |
| [LLM Red-Team Suite: Jailbreak Attacks, Defences and Over-Refusal](llm-red-team) | JailbreakBench requests x 4 jailbreak patterns x 3 defences on two free 3B models, judged by Llama Guard, with over-refusal measured on matched harmless requests | Prefix injection jailbreaks Qwen2.5 3B 79% of the time; self-check cuts attacks 40% to 5% but refuses 42% of harmless requests |
| [Production RAG: Cost, Latency and Quality Dashboard](rag-cost-dashboard) | Every serving knob (3 free models, k = 3/5/10 chunks, context compression, a small-to-large router) measured on FinanceBench for tokens, CPU seconds, p95 latency and accuracy, with a Streamlit cost dashboard | Compression cuts tokens 40% and lifted the best setting to 22.5% accuracy; the router cost more and scored lower because the small model refuses 58-81% of the time |

In progress: Multi-step agent with guardrails.

[Back to all projects](../../README.md)
