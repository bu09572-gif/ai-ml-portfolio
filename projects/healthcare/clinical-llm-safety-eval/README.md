# Clinical LLM Evaluation: Accuracy, Robustness and Bias Checks

Before a hospital lets a language model answer clinical questions, it needs more than one accuracy number. This project runs two free, open 3-billion-parameter models on two medical benchmarks with expert answers. It then asks three more questions:
1. Does the answer change if I only reorder the options?
2. Does it change if I only swap the patient's sex?
3. Can the model's own inconsistency tell me when not to trust it?

## Data

- **MedQA** (Jin et al., 2021): all 1,273 USMLE-style test questions with 4 options. I use the Hugging Face copy `GBaker/MedQA-USMLE-4-options`, mirrored into this repo's `data-v2` release with a SHA-256 check.
- **PubMedQA** (Jin et al., 2019): all 500 expert-labelled test questions. Each is answered yes, no or maybe from a PubMed abstract with its conclusion removed. The files come from the authors' GitHub.

## Models and checks

The models are Qwen2.5 3B and Llama 3.2 3B, run on CPU with Ollama in GitHub Actions, at temperature 0. That's 20 parallel jobs (`.github/workflows/clinical-llm-eval.yml`). Each MedQA question is asked up to three times:

| Check | How | What a flip means |
|---|---|---|
| **Option order** | Same question, options reordered so every option moves | The model answers by position or guesses. This is the noise floor |
| **Patient sex** | Same vignette with sex words swapped (man to woman, he to she, his to her, and so on), only on the 705 questions where sex should not matter. Questions mentioning pregnancy, sex-specific organs, sex hormones or X-linked disease are left out | A possible bias signal, if it happens more than chance |
| **Abstention** | Answer only when the original and reordered runs agree | Trades coverage for accuracy |

`clinical_eval.py` has the swap rules. "Her" is the hard case: it maps to "him" or "his" depending on the next word. The tests check it.

## Results

From `results/summary.json`. Every row is in `results/rows.jsonl`.

| | Llama 3.2 3B | Qwen2.5 3B |
|---|---|---|
| MedQA accuracy (1,273 questions, chance 25%) | **54.9%** | 50.4% |
| same answer after reordering options | 70.7% | 71.5% |
| accuracy when it abstains on disagreement | 63.6% on 70.7% of questions | 56.6% on 71.5% |
| same answer after swapping patient sex (705 questions) | **95.2%** | 91.5% |
| accuracy, original vs sex-swapped | 53.8% vs 53.5% | 49.2% vs 47.5% |
| PubMedQA accuracy (500 questions, always "yes" = 55.2%) | **69.4%** | 56.0% |

**What this shows:**

- **Both models are far from safe for clinical use.** About half of exam questions are right. A passing USMLE score is roughly 60%, and current large models score above 85% on MedQA.
- **Reordering the options changes about 3 in 10 answers.** A model that knows the answer picks the same option text wherever it sits. These models don't, for 29% of questions. That instability is a useful signal, though. Answering only when both orders agree raises Llama's accuracy from 54.9% to 63.6%, at the cost of abstaining on 29% of questions.
- **I found no evidence of patient-sex bias on these questions.** Swapping the patient's sex changes 4.8% of Llama's answers and 8.5% of Qwen's, far below the 29% caused by reordering options. The flips go both ways:
  - Llama lost 14 correct answers and gained 12 (McNemar p = 0.85).
  - Qwen lost 29 and gained 17 (p = 0.10). That leans the wrong way, so it would be worth a larger test, but it is not significant here.
- **On PubMedQA, Qwen barely beats always answering "yes"**: 56.0% against 55.2%. It answers "maybe" 115 times, while the experts chose "maybe" for only 55 of the 500 questions. Llama is clearly better at 69.4%. This is why a majority-class baseline belongs next to every accuracy number.

## Limits

- Exam questions are not patient care. A model that passes MedQA can still be unsafe on real notes.
- The sex swap only changes words. It cannot test bias that comes from how symptoms are described for different groups. And "no evidence" at 705 questions is not proof of no bias.
- Only small CPU models were run. The checks are model-agnostic: set `EVAL_MODELS` to run others.

## Running it

```bash
pip install -e ../../..          # shared code, from the repo root
pytest -q test_clinical_eval.py  # scripted model, no LLM needed
# full run: edit run-request.txt and push; GitHub Actions runs it and commits results/
```
