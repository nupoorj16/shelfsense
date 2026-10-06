import pandas as pd

from shelfsense.baselines import cooccurrence_recommender, popularity_recommender
from shelfsense.ranker import label_candidates


def _history():
    rows = [
        ("u1", "a", "2023-01-01"),
        ("u1", "b", "2023-01-02"),
        ("u2", "a", "2023-01-03"),
        ("u2", "b", "2023-01-04"),
        ("u3", "a", "2023-01-05"),
        ("u4", "c", "2023-01-06"),
    ]
    df = pd.DataFrame(rows, columns=["user_id", "parent_asin", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_popularity_skips_owned_items():
    recs = popularity_recommender(_history(), ["u3"], k=2)
    assert recs["u3"][0] == "b"  # "a" is the most popular but u3 owns it


def test_cooccurrence_recommends_what_shared_buyers_also_bought():
    recs = cooccurrence_recommender(_history(), ["u3"], k=1)
    assert recs["u3"] == ["b"]


def test_label_candidates_marks_true_purchases():
    cands = pd.DataFrame({"user_id": ["u", "u"], "parent_asin": ["x", "y"]})
    labeled = label_candidates(cands, {"u": {"y"}})
    assert labeled["label"].tolist() == [0, 1]
