import aml
import data
import numpy as np
import pandas as pd


def toy():
    n = pd.DataFrame({"txId": [10, 20, 30], "time_step": [1, 1, 40], "label": [1, 0, -1]})
    for i, c in enumerate(data.LOCAL):
        n[c] = np.float32([1, 2, 3][i % 3] if i < 3 else 0)
    n["local_0"] = np.float32([1, 2, 3])
    return n, pd.DataFrame({"src": [10, 10], "dst": [20, 30]})


def test_adjacency_direction():
    n, e = toy()
    a = aml.adjacency(n, e).toarray()
    assert a[0, 1] == 1 and a[0, 2] == 1 and a[1, 0] == 0


def test_graph_features_payer_payee_means():
    n, e = toy()
    g = aml.graph_features(n, aml.adjacency(n, e))
    assert g.out_deg.tolist() == [2, 0, 0] and g.in_deg.tolist() == [0, 1, 1]
    assert g.payee_mean_local_0[0] == 2.5            # pays into transactions with 2 and 3
    assert g.payer_mean_local_0[1] == 1.0            # paid by the transaction with 1
    assert g.payer_mean_local_0[0] == 0.0            # no payers


def test_split_is_by_time_and_skips_unknown():
    n, _ = toy()
    tr, te = aml.split(n)
    assert tr.tolist() == [True, True, False] and te.tolist() == [False, False, False]


def test_scores():
    s = aml.scores(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.2, 0.1]))
    assert (s["precision"], s["recall"]) == (0.5, 0.5)


def test_real_data_counts():
    nodes, edges = data.load()
    assert len(nodes) == 203769 and len(edges) == 234355
    assert (nodes.label == 1).sum() == 4545 and (nodes.label == 0).sum() == 42019
    assert nodes.time_step.min() == 1 and nodes.time_step.max() == 49
