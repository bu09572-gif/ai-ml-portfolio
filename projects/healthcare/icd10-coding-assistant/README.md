# ICD-10 Diagnosis Coding Assistant for Clinical Notes

This assistant reads a clinical note and suggests a ranked short list of ICD-10 diagnosis codes for a human coder to review. Codes found by text matching come with the sentence from the note that matched. It is trained and scored on real clinical cases that expert coders had already coded.

## Data

**CodiEsp** (Miranda-Escalada et al., CLEF eHealth 2020; CC BY 4.0):

- 1,000 published clinical case reports, each coded by expert coders with ICD-10 diagnosis codes. The codes come from the Spanish edition of ICD-10-CM.
- Splits: 500 train, 250 dev, 250 test. A case has 11.4 codes on average, and 2,557 different codes appear.
- The corpus includes English translations of every case, and those are what I use.
- I also use the official code list, with 71,344 English diagnosis descriptions.

Both files come from Zenodo. My build environment can't reach Zenodo, so `tools/fetch_urls.py` mirrors them in GitHub Actions into this repo's `data-v2` release. `data.py` downloads that copy and checks the SHA-256.

The data is hard: 14% of the codes in test cases never appear in the 750 train and dev cases. 70% of the 2,194 codes in train and dev appear in only one or two cases.

## How it works

Every candidate code gets four scores. There are 44,826 candidates: the codes seen in training plus all codes in the same 3-character ICD-10 categories.

| Ranker | Idea |
|---|---|
| frequency | how common the code is in training: the floor to beat |
| classifier | TF-IDF of the note + one logistic regression per code seen at least twice |
| neighbours | codes of the 20 most similar training notes, weighted by similarity |
| dictionary | best TF-IDF match between any sentence of the note and the code's official description. This can suggest codes never seen in training, and it gives the matching sentence as evidence |
| ensemble | weighted sum of the four. Weights are tuned on dev, then the model is refit on train + dev |

The tuned weights were classifier 0.5 and dictionary 0.25. Frequency and neighbours got 0, since the classifier already carries what they add.

## Results (250 test cases)

The metric is CodiEsp's official one, MAP over the ranked list. I also report how many of the right codes land in a short list. From `results/metrics.json`.

| Ranker | MAP | Precision@5 | Recall@10 | Recall@20 |
|---|---|---|---|---|
| frequency | 0.080 | 0.191 | 0.126 | 0.188 |
| dictionary | 0.072 | 0.173 | 0.115 | 0.150 |
| neighbours | 0.230 | 0.390 | 0.296 | 0.391 |
| classifier | 0.275 | 0.456 | 0.324 | 0.421 |
| **ensemble** | **0.326** | **0.529** | **0.376** | **0.470** |

- **The ensemble is the best on every measure.** About half of its top 5 suggestions are correct, and its top 20 hold 47% of the codes the expert assigned.
- **The dictionary ranker is weak alone but useful in the ensemble.** Adding it to the classifier takes MAP from 0.275 to 0.326. It is also the only ranker that can suggest a code with no training examples.
- **This is a coder's assistant, not an autocoder.** With 11 codes per case and recall@20 of 47%, a person still has to read the note.

Example: the first test case (`results/examples.json`), a bladder cancer case. Of the top 8 suggestions, 4 are correct: hematuria R31.9, type 2 diabetes E11.9, kidney cyst N28.1 and bladder cancer C67.9. Wrong suggestions include hypertension I10 and kidney disorder, unspecified N28.9.

The evidence sentences are a mixed bag. For hematuria it points at "...presents macroscopic post-voiding hematuria...", which is right. For type 2 diabetes it points at a sentence about "Dupuytren's disease ... type II", a word match with the wrong meaning. The evidence helps the coder check a code, but it does not prove the code.

## Limits

- **Translated text.** The English text is a machine translation of Spanish case reports, and case reports are cleaner than real hospital notes.
- **No large model.** I didn't use one here. A natural next step is to let a model choose among the ensemble's top 30 codes, or to fine-tune a clinical BERT model.

## Running it

```bash
pip install scikit-learn pandas pyarrow requests
python run.py            # about a minute on a 4-core CPU
pytest -q test_coder.py
```
