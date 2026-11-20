# PII Redaction Pipeline on Real Court Cases

Finds and masks personal information in long legal documents, and is scored with the official scorer of the [Text Anonymization Benchmark](https://github.com/NorskRegnesentral/text-anonymization-benchmark) (TAB; Pilán et al., 2022). TAB has 1,268 real judgments from the European Court of Human Rights. People annotated every personal identifier by hand and marked whether it has to be masked to protect the person the case is about.

The question that matters for redaction isn't "how many names did you find". It's whether anyone can still be identified. So the main metric is **entity recall**: an entity counts as protected only if *every* mention of it is masked. Missing one mention of a name is enough to re-identify someone.

## Methods

| Method | What it does |
|---|---|
| rules | regular expressions: dates, application/case numbers, money, ages, emails, phone numbers |
| ner | off-the-shelf spaCy `en_core_web_md` NER, masking people, places, organisations, dates, nationalities and numbers |
| trained | spaCy NER trained on the 1,014 TAB training cases to predict which spans need masking (8 types: person, code, place, organisation, date, demographic, quantity, misc) |
| trained+rules | union of the trained model and the rules |
| trained+ner | union of the trained model and off-the-shelf NER |

## Results

127 test cases (about 0.6M characters), scored with TAB's `evaluation.py`, from `results/test_scores.json`.

| Method | Direct identifiers fully masked | Quasi identifiers fully masked | Token recall | Token precision | Token F1 | ms per case |
|---|---|---|---|---|---|---|
| rules | 49.6% | 56.9% | 59.2% | 89.8% | 0.714 | 1 |
| ner (off the shelf) | 78.6% | 87.3% | 90.1% | 52.1% | 0.660 | 106 |
| **trained** | **97.7%** | 80.9% | 82.1% | **90.1%** | **0.859** | 248 |
| trained+rules | 97.7% | 81.8% | 83.1% | 86.8% | 0.849 | 235 |
| trained+ner | **98.9%** | **91.2%** | **93.9%** | 53.7% | 0.683 | 372 |

"Direct" identifiers are things like names and application numbers. "Quasi" identifiers are things like dates, places and jobs, which identify someone only in combination.

What I take from this:

- **Training on the task matters most.** Off-the-shelf NER misses about 1 in 5 directly identifying entities and masks about twice as much as it should: half its masked tokens didn't need masking. The trained model protects 97.7% of direct identifiers, and 90% of what it masks really needed it.
- **There are two sensible operating points.** `trained` keeps the document readable (0.859 F1). `trained+ner` masks almost everything that could identify someone (98.9% direct, 91.2% quasi), but it removes a lot of harmless text. Which one to use depends on the risk: publishing externally or sharing internally.
- **Where the trained model is weak.** Token recall by type: codes 96%, dates 96%, people 91%, places 72%, quantities 61%, organisations 42%, misc 28%, demographics 25%. Demographic details ("a 45-year-old police officer") and organisations are where re-identification risk remains.
- The rules add little on top of the trained model (+0.9 points of quasi recall) and cost precision.

Training: spaCy 3.8 NER with `en_core_web_md` static vectors, on CPU. Best dev F1 was 0.81 at step 1,000. It stayed flat through step 2,000, where the run was stopped, and `model/model-best` is that checkpoint. Per type on dev: codes 0.97, dates 0.93, people 0.90, places 0.55, quantities 0.56, organisations 0.42.

## Running it

```bash
pip install spacy intervaltree tqdm pandas requests
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl

python data.py         # downloads TAB (checksums pinned) and its official scorer
python train.py        # builds the spaCy corpus and trains the model
python run_eval.py     # scores every method on the test set
pytest -q test_redact.py
```

```python
import redact as R
nlp = R.trained()
R.redact(text, nlp(text))   # "... the applicant, [PERSON], was arrested on [DATETIME] in [LOC] ..."
```

## Data

TAB v1.0, MIT license. The text is public ECHR case law. `data.py` downloads it and checks the sha256 of each file. Nothing from the corpus is stored in this repo.
