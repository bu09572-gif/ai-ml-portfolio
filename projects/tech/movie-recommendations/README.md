# Movie Recommendations: Retrieval, Ranking and Honest Offline Evaluation

This project builds and compares recommenders on real movie ratings, from a popularity baseline to a two-stage retrieve-and-rank system. The test is strict: hide each user's most recent movie, then find it among every movie the user hasn't rated. The main finding is not the model I expected.

## Data

**MovieLens 1M** (GroupLens; Harper and Konstan, 2015): 1,000,209 ratings of 3,706 movies by 6,040 users, with timestamps, genres and user age. It is mirrored from GroupLens into this repo's `data-v2` release with a SHA-256 check.

**Protocol:**
- Per user, sorted by time: the last movie is the test target, the second-to-last is for validation, and the rest is training history.
- A rating counts as "watched", whatever the score.
- Every model ranks all ~3,700 movies, minus those already rated. I don't use the common shortcut of ranking the target against 100 random movies, because it inflates scores and can even change which model looks best (Krichene and Rendle, 2020).

## Models

| Model | Idea |
|---|---|
| popularity | the most-watched movies you haven't seen |
| item kNN | movies similar to the ones you watched (cosine similarity on who watched what, 100 neighbours) |
| implicit ALS | matrix factorisation for implicit feedback (Hu, Koren and Volinsky, 2008), 64 factors, in plain NumPy |
| EASE | a closed-form linear item-to-item model (Steck, 2019), a strong and simple baseline |
| EASE on your last 5 movies | the same EASE weights, scored from only the user's most recent movies. The number 5 was picked on validation from 1, 3, 5, 10, 20 and 50 |
| two-stage ranker | EASE retrieves 100 candidates per user. A gradient-boosted model, trained on the validation targets, re-orders them using EASE, kNN, ALS, recent-history EASE, popularity in the last 30 days, movie age and genre match |

## Results (6,040 users, one hidden movie each)

From `results/metrics.json`.

| Model | Hit@10 | NDCG@10 | Hit@20 | NDCG@20 |
|---|---|---|---|---|
| popularity | 3.5% | 0.017 | 6.4% | 0.025 |
| item kNN | 7.0% | 0.037 | 11.8% | 0.049 |
| implicit ALS | 7.4% | 0.036 | 13.2% | 0.050 |
| EASE (all history) | 10.0% | 0.049 | 16.6% | 0.066 |
| two-stage ranker | 17.9% | 0.095 | 25.4% | 0.115 |
| two-stage ranker without the recent-history feature | 9.7% | 0.048 | 16.4% | 0.065 |
| **EASE on your last 5 movies** | **19.1%** | **0.105** | **26.9%** | **0.125** |

- **Order matters more than model complexity.** Scoring EASE from only the last 5 movies nearly doubles Hit@10, from 10.0% to 19.1%. People watch in runs, such as a sequel, a genre binge or a director, and models that treat history as an unordered set miss that.
- **The ranker's gain all came from one feature.** Without recent-history EASE, the ranker is no better than plain EASE (9.7% against 10.0%). With it, the ranker reaches 17.9%, still below just using that feature over the whole catalogue.
- **Retrieval caps the two-stage system.** The 100 EASE candidates contain the hidden movie for only 43.6% of users. The fix is to retrieve with the recent-history model too. Checking feature value by removing it, and measuring the candidate stage's recall, are the two checks I'd run before adding any ranker in production.
- **EASE beats matrix factorisation and kNN** here, matching published results on this dataset.

**"More like this" from the EASE item weights:**
- **Toy Story:** Toy Story 2, A Bug's Life, Aladdin, Babe, Groundhog Day.
- **The Godfather:** The Godfather Part II, Part III, Star Wars IV, One Flew Over the Cuckoo's Nest, Casablanca.

## Limits

- MovieLens timestamps record when someone rated a movie, not when they watched it. Many users rate dozens of movies in one session when they join. Part of the "recent history" effect may be that rating order, not viewing order.
- Offline hit rate is not the same as what people click or enjoy. A real system would confirm it with an A/B test.
- The data is from 2000 to 2003, with 3,706 movies. A modern catalogue needs approximate nearest-neighbour retrieval instead of dense item-item matrices.

## Running it

```bash
pip install scikit-learn scipy pandas requests
python run.py            # about 1-2 minutes on CPU
pytest -q test_recs.py
```
