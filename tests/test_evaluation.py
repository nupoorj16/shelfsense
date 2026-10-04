import numpy as np
import pytest

from shelfsense.evaluation import ndcg_at_k, recall_at_k, top_k_excluding_seen


def test_recall_counts_hits_in_top_k_only():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=2) == 0.5
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=3) == 1.0


def test_recall_with_no_relevant_items_is_zero():
    assert recall_at_k(["a"], set(), k=10) == 0.0


def test_ndcg_rewards_hits_near_the_top():
    first = ndcg_at_k(["a", "x", "y"], {"a"}, k=3)
    last = ndcg_at_k(["x", "y", "a"], {"a"}, k=3)
    assert first == pytest.approx(1.0)
    assert 0 < last < first


def test_top_k_skips_seen_items_and_sorts_by_score():
    scores = np.array([[0.9, 0.8, 0.1, 0.5]])
    assert top_k_excluding_seen(scores, [{0}], k=2) == [[1, 3]]
