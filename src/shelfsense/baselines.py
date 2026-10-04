"""Simple recommenders that the real system has to beat."""

import numpy as np
import pandas as pd
from scipy import sparse

from shelfsense.evaluation import top_k_excluding_seen

RECENT_DAYS = 180
HISTORY_ITEMS = 20  # how many of a user's latest reviews to use as "taste"


def recent_popular(history: pd.DataFrame, days: int = RECENT_DAYS) -> pd.Series:
    """Review counts per item over the last `days` of history, most popular first."""
    cutoff = history["timestamp"].max() - pd.Timedelta(days=days)
    recent = history[history["timestamp"] >= cutoff]
    return recent["parent_asin"].value_counts()


def popularity_recommender(
    history: pd.DataFrame, user_ids: list[str], k: int
) -> dict[str, list[str]]:
    """Everyone gets the same list: the most reviewed products lately (minus ones they own)."""
    ranked = recent_popular(history).index.tolist()
    seen = history.groupby("user_id")["parent_asin"].agg(set).to_dict()
    recs = {}
    for user in user_ids:
        owned = seen.get(user, set())
        recs[user] = [item for item in ranked[: k + len(owned)] if item not in owned][
            :k
        ]
    return recs


class ItemIndex:
    """Maps product IDs <-> matrix column numbers."""

    def __init__(self, items: list[str]):
        self.items = list(items)
        self.position = {item: i for i, item in enumerate(self.items)}

    def __len__(self) -> int:
        return len(self.items)


def user_item_matrix(
    history: pd.DataFrame, index: ItemIndex
) -> tuple[sparse.csr_matrix, dict[str, int]]:
    users = history["user_id"].unique()
    user_pos = {u: i for i, u in enumerate(users)}
    rows = history["user_id"].map(user_pos).to_numpy()
    cols = history["parent_asin"].map(index.position).to_numpy()
    data = np.ones(len(history), dtype=np.float32)
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(users), len(index)))
    return matrix, user_pos


def cooccurrence_similarity(
    history: pd.DataFrame, index: ItemIndex
) -> sparse.csr_matrix:
    """Cosine similarity between items based on which users reviewed both.

    "Customers who bought A also bought B": the more shared customers,
    the higher the score, normalized so popular items don't win everything.
    """
    matrix, _ = user_item_matrix(history, index)
    co = (matrix.T @ matrix).tocsr()
    co.setdiag(0)
    co.eliminate_zeros()
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    norm = np.sqrt(np.maximum(counts, 1))
    inv = sparse.diags(1.0 / norm)
    return (inv @ co @ inv).tocsr().astype(np.float32)


def latest_items(history: pd.DataFrame, n: int = HISTORY_ITEMS) -> pd.DataFrame:
    """Each user's n most recent reviews, with a recency weight (newest = 1.0)."""
    latest = history.sort_values("timestamp").groupby("user_id").tail(n).copy()
    latest["recency_rank"] = latest.groupby("user_id")["timestamp"].rank(
        ascending=False, method="first"
    )
    latest["weight"] = 0.9 ** (latest["recency_rank"] - 1)
    return latest


def user_profiles(
    latest: pd.DataFrame, user_ids: list[str], index: ItemIndex
) -> sparse.csr_matrix:
    """One row per user: recency weights on the items they reviewed lately."""
    latest = latest[latest["user_id"].isin(user_ids)]
    user_pos = {u: i for i, u in enumerate(user_ids)}
    rows = latest["user_id"].map(user_pos).to_numpy()
    cols = latest["parent_asin"].map(index.position).to_numpy()
    profile = sparse.csr_matrix(
        (latest["weight"].to_numpy(np.float32), (rows, cols)),
        shape=(len(user_ids), len(index)),
    )
    return profile


def owned_items(history: pd.DataFrame) -> dict[str, set[str]]:
    return history.groupby("user_id")["parent_asin"].agg(set).to_dict()


def seen_positions(
    owned: dict[str, set[str]], user_ids: list[str], index: ItemIndex
) -> list[set[int]]:
    return [
        {index.position[i] for i in owned.get(u, ()) if i in index.position}
        for u in user_ids
    ]


def cooccurrence_recommender(
    history: pd.DataFrame, user_ids: list[str], k: int, batch_size: int = 2000
) -> dict[str, list[str]]:
    index = ItemIndex(sorted(history["parent_asin"].unique()))
    similarity = cooccurrence_similarity(history, index)
    popular = recent_popular(history)
    # Tiny popularity bonus breaks ties for users whose items have no co-buyers.
    pop_bonus = np.zeros(len(index), dtype=np.float32)
    pop_positions = [index.position[i] for i in popular.index]
    pop_bonus[pop_positions] = 1e-6 * popular.to_numpy() / popular.max()

    latest = latest_items(history)
    owned = owned_items(history)
    recs: dict[str, list[str]] = {}
    for start in range(0, len(user_ids), batch_size):
        batch = user_ids[start : start + batch_size]
        scores = (user_profiles(latest, batch, index) @ similarity).toarray()
        scores += pop_bonus
        top = top_k_excluding_seen(scores, seen_positions(owned, batch, index), k)
        for user, positions in zip(batch, top, strict=True):
            recs[user] = [index.items[p] for p in positions]
    return recs
