import data
import numpy as np
import pandas as pd
import recs


def toy():
    return pd.DataFrame({"user": [1, 1, 1, 2, 2, 2, 2], "item": [10, 20, 30, 10, 20, 30, 40], "ts": [1, 2, 3, 1, 2, 3, 4]})


def test_split_last_by_time():
    tr, va, te = recs.split_last(toy())
    assert te.set_index("user").item.to_dict() == {1: 30, 2: 40}
    assert va.set_index("user").item.to_dict() == {1: 20, 2: 30}
    assert len(tr) == 3


def test_hit_ndcg_ranks_against_all_items():
    s = np.array([[0.9, 0.5, 0.1], [0.1, 0.2, 0.3]])
    m = recs.hit_ndcg(s, np.array([0, 0]), ks=(1,))
    assert m["hit_at_1"] == 0.5 and m["mrr"] == round((1 + 1 / 3) / 2, 4)


def test_mask_seen_and_ease_never_recommend_seen():
    r = toy()
    ix = recs.Index(r)
    x = ix.matrix(r[r.user == 1].assign(user=1))
    s = recs.mask_seen(recs.ease(ix.matrix(r), lam=1.0), ix.matrix(r))
    assert np.isneginf(s[0, :3]).all() and np.isfinite(s[0, 3])
    assert x.shape == (2, 4)


def test_candidate_metrics_miss_when_not_in_candidates():
    c = pd.DataFrame({"u": [0, 0, 1, 1], "i": [5, 6, 5, 6]})
    m = recs.candidate_metrics(c, np.array([0.1, 0.9, 0.5, 0.4]), np.array([5, 7]), ks=(1, 2))
    assert m["hit_at_2"] == 0.5 and m["hit_at_1"] == 0.0 and m["target_in_candidates"] == 0.5


def test_real_data():
    r, m, u = data.load()
    assert len(r) == 1000209 and r.user.nunique() == 6040 and r.item.nunique() == 3706 and len(u) == 6040
