import pandas as pd

from shelfsense.config import TEST_START, VAL_START
from shelfsense.data.prepare import assign_split, drop_duplicates, k_core


def test_drop_duplicates_keeps_first_review():
    reviews = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "parent_asin": ["a", "a", "a"],
            "timestamp": pd.to_datetime(["2021-05-01", "2020-01-01", "2021-01-01"]),
        }
    )
    result = drop_duplicates(reviews)
    assert len(result) == 2
    u1 = result[result["user_id"] == "u1"]
    assert u1["timestamp"].iloc[0] == pd.Timestamp("2020-01-01")


def test_k_core_removes_users_and_items_until_stable():
    # u3 has only 1 review -> removed. That drops item "c" to 1 review -> removed too.
    reviews = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u1"],
            "parent_asin": ["a", "b", "a", "b", "c", "c"],
        }
    )
    result = k_core(reviews, k=2)
    assert set(result["user_id"]) == {"u1", "u2"}
    assert set(result["parent_asin"]) == {"a", "b"}


def test_assign_split_by_time():
    timestamps = pd.Series(
        [
            VAL_START - pd.Timedelta(days=1),
            VAL_START,
            TEST_START,
            TEST_START + pd.Timedelta(days=30),
        ]
    )
    assert assign_split(timestamps).tolist() == ["train", "val", "test", "test"]
