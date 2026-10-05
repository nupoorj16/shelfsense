"""Stage 2: a LightGBM learning-to-rank model that orders the candidates.

LambdaRank learns from groups (one group = one user's candidates) and directly
tries to push the products the user actually reviewed to the top of their list.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from shelfsense.candidates import FEATURES
from shelfsense.config import ARTIFACTS_DIR, SEED

RANKER_FILE = ARTIFACTS_DIR / "ranker.txt"

PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [10],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "lambdarank_truncation_level": 20,
    "verbose": -1,
    "seed": SEED,
}


def label_candidates(cands: pd.DataFrame, truth: dict[str, set[str]]) -> pd.DataFrame:
    labels = [
        item in truth.get(user, ())
        for user, item in zip(cands["user_id"], cands["parent_asin"], strict=True)
    ]
    cands = cands.copy()
    cands["label"] = np.array(labels, dtype=np.int8)
    return cands


def _dataset(cands: pd.DataFrame, reference: lgb.Dataset | None = None) -> lgb.Dataset:
    cands = cands.sort_values("user_id", kind="stable")
    groups = cands.groupby("user_id", sort=False).size().to_numpy()
    return lgb.Dataset(
        cands[FEATURES], label=cands["label"], group=groups, reference=reference
    )


def train_ranker(
    labeled: pd.DataFrame, num_rounds: int = 500, valid_share: float = 0.2
) -> lgb.Booster:
    """Train on users that have at least one positive candidate; early-stop on a held-out slice."""
    has_positive = labeled.groupby("user_id")["label"].transform("max") > 0
    labeled = labeled[has_positive]

    users = labeled["user_id"].unique()
    rng = np.random.default_rng(SEED)
    valid_users = set(
        rng.choice(users, size=int(len(users) * valid_share), replace=False)
    )
    is_valid = labeled["user_id"].isin(valid_users)

    train_set = _dataset(labeled[~is_valid])
    valid_set = _dataset(labeled[is_valid], reference=train_set)
    return lgb.train(
        PARAMS,
        train_set,
        num_boost_round=num_rounds,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)],
    )


def rank(model: lgb.Booster, cands: pd.DataFrame, k: int) -> dict[str, list[str]]:
    cands = cands.assign(score=model.predict(cands[FEATURES]))
    cands = cands.sort_values(["user_id", "score"], ascending=[True, False])
    top = cands.groupby("user_id").head(k)
    return top.groupby("user_id")["parent_asin"].agg(list).to_dict()
