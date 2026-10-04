"""Ranking metrics and the evaluation protocol shared by every model."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from shelfsense.config import INTERACTIONS_FILE

# A recommender takes (history, user_ids, k) and returns one ranked item list per user.
Recommender = Callable[[pd.DataFrame, list[str], int], dict[str, list[str]]]


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Share of the relevant items that appear in the top k."""
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Like recall, but a hit at position 1 counts more than a hit at position 10."""
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal


@dataclass
class EvalSet:
    history: pd.DataFrame  # everything the model is allowed to see
    truth: dict[str, set[str]]  # user -> items they reviewed in the target period

    @property
    def users(self) -> list[str]:
        return list(self.truth)


def load_eval_set(split: str, warm_only: bool = True) -> EvalSet:
    """Build the history and ground truth for split 'val' or 'test'.

    For 'val' the model sees train only. For 'test' it sees train + val.
    warm_only=True keeps users who have at least one review in the history.
    """
    interactions = pd.read_parquet(INTERACTIONS_FILE)
    visible = ["train"] if split == "val" else ["train", "val"]
    history = interactions[interactions["split"].isin(visible)]
    target = interactions[interactions["split"] == split]

    known_users = set(history["user_id"])
    if warm_only:
        target = target[target["user_id"].isin(known_users)]
    else:
        target = target[~target["user_id"].isin(known_users)]

    truth = target.groupby("user_id")["parent_asin"].agg(set).to_dict()
    return EvalSet(history=history, truth=truth)


def evaluate(
    recommendations: dict[str, list[str]],
    truth: dict[str, set[str]],
    ks: tuple[int, ...] = (10, 50),
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        recalls = [
            recall_at_k(recommendations.get(u, []), rel, k) for u, rel in truth.items()
        ]
        ndcgs = [
            ndcg_at_k(recommendations.get(u, []), rel, k) for u, rel in truth.items()
        ]
        metrics[f"recall@{k}"] = float(np.mean(recalls))
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs))
    return metrics


def top_k_excluding_seen(
    scores: np.ndarray, seen: list[set[int]], k: int
) -> list[list[int]]:
    """Pick the k highest-scoring item indices per row, skipping items the user already has."""
    scores = scores.copy()
    for row, items in enumerate(seen):
        if items:
            scores[row, list(items)] = -np.inf
    k = min(k, scores.shape[1])
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(top, order, axis=1).tolist()
