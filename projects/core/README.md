# Core AI platform projects

| Project | What it does | Key result |
|---|---|---|
| [Document Q&A Platform for SEC Filings](document-qa-platform) | Filing ingestion pipeline, Q&A with page citations, and an LLM evaluation harness that fails CI when quality drops | 84 real filings, 150 analyst questions; company/year filters double retrieval hit@5 |
| [Agentic RAG Copilot for SEC Filings](agentic-rag-copilot) | Agent that searches, reads and calculates, with guardrails enforced in code | 20.7% vs 19.3% for plain Q&A; 34% vs 14% on open-ended questions |
| [PII Redaction Pipeline on Real Court Cases](pii-redaction) | Trained NER model that masks personal identifiers in court judgments | 97.7% of direct identifiers fully masked, 90% precision |

[Back to all projects](../../README.md)
