import lightgbm  # noqa: F401  # must be imported before faiss (OpenMP clash on macOS)
import numpy as np
import pandas as pd
import pytest

from shelfsense.serving import Recommender


class FakeScorer:
    """Stands in for the LightGBM model: ranks by embedding similarity."""

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return data["emb_sim"].to_numpy()


@pytest.fixture(scope="session")
def recommender() -> Recommender:
    rng = np.random.default_rng(0)
    n_items, n_warm, dim = 30, 24, 16
    ids = [f"P{i:02d}" for i in range(n_items)]
    vectors = rng.normal(size=(n_items, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    products = pd.DataFrame(
        {
            "parent_asin": ids,
            "title": [
                f"Product {i} {'serum' if i % 2 else 'lipstick'}"
                for i in range(n_items)
            ],
            "store": [f"Brand{i % 3}" for i in range(n_items)],
            "subcategory": ["Skin Care" if i % 2 else "Makeup" for i in range(n_items)],
            "price": [None if i % 5 == 0 else 9.99 + i for i in range(n_items)],
        }
    )
    # P24..P29 never appear in the history: they are the "cold" (brand-new) products.
    rows = []
    for u in range(40):
        for item in rng.choice(n_warm, size=5, replace=False):
            rows.append(
                (
                    f"U{u}",
                    ids[item],
                    5,
                    pd.Timestamp("2023-01-01")
                    + pd.Timedelta(days=int(rng.integers(0, 200))),
                )
            )
    history = pd.DataFrame(
        rows, columns=["user_id", "parent_asin", "rating", "timestamp"]
    )
    return Recommender(vectors, ids, products, history, FakeScorer())
