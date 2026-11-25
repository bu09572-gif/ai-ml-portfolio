# Healthcare projects

| Project | What it does | Key result |
|---|---|---|
| [ICD-10 Diagnosis Coding Assistant for Clinical Notes](icd10-coding-assistant) | Ranks ICD-10 codes for a clinical note, with the sentence that supports each code | MAP 0.326 on 250 expert-coded cases; 53% of top-5 suggestions correct |
| [Clinical LLM Evaluation: Accuracy, Robustness and Bias Checks](clinical-llm-safety-eval) | Two free 3B models on MedQA and PubMedQA, with option-order, patient-sex-swap and abstention checks | Llama 3.2 3B: 54.9% MedQA; 29% of answers change when options are reordered vs 5% when patient sex is swapped |
| [Clinical Note Summarizer with De-identification](clinical-note-summarizer) | Removes identifiers before a free local model summarizes 300 real clinical notes; checks leaks, invented numbers and an LLM judge | Identifier repeats fall from 22% to 4% of notes; the 3B judge's flags are wrong 81% of the time |
| [Prior Authorization Assistant over Medicare Coverage Policies](prior-auth-assistant) | Finds the right Medicare coverage policy (LCD) for a clinical note, turns it into a checklist of quoted criteria, and checks the note with cited evidence | Right policy in the top 10 for 82% of notes; a free 3B model's met/not-met verdicts fail the evidence check, so they go to a reviewer |
| [Claim Denial Analytics and Grounded Appeal Letters](claim-denial-appeals) | Denial and appeal analytics on the CMS 2024 Transparency in Coverage file (386M claims), plus a free local model drafting appeal letters from quoted chart evidence | Only 1.6 appeals per 1,000 denials, yet 43% win; grounded letters keep 98% of quotes real vs 75% when the model reads the whole chart |
| [Drug Safety Review with Multi-Agent LLMs and a LoRA Fine-Tuned Extractor](https://github.com/bu09572-gif/drug-safety-agents) | LoRA fine-tuned 0.5B model pulls adverse drug events from PubMed case reports; a planner / worker / reviewer workflow (LangGraph) checks each finding against the FDA label, with full tracing | Fine-tuned F1 0.667 vs 0.293 prompting, 0.362 RAG and 0.467 for a 3x larger model with RAG; 99.5% valid JSON |

[Back to all projects](../../README.md)
