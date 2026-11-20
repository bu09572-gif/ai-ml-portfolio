# Healthcare projects

| Project | What it does | Key result |
|---|---|---|
| [ICD-10 Diagnosis Coding Assistant for Clinical Notes](icd10-coding-assistant) | Ranks ICD-10 codes for a clinical note, with the sentence that supports each code | MAP 0.326 on 250 expert-coded cases; 53% of top-5 suggestions correct |
| [Clinical LLM Evaluation: Accuracy, Robustness and Bias Checks](clinical-llm-safety-eval) | Two free 3B models on MedQA and PubMedQA, with option-order, patient-sex-swap and abstention checks | Llama 3.2 3B: 54.9% MedQA; 29% of answers change when options are reordered vs 5% when patient sex is swapped |
| [Clinical Note Summarizer with De-identification](clinical-note-summarizer) | Removes identifiers before a free local model summarizes 300 real clinical notes; checks leaks, invented numbers and an LLM judge | Identifier repeats fall from 22% to 4% of notes; the 3B judge's flags are wrong 81% of the time |

In progress: Prior authorization assistant over CMS coverage policies; Claim denial predictor and appeal drafter.

[Back to all projects](../../README.md)
