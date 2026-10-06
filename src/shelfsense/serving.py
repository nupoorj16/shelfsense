"""Online recommender: "customers who viewed this also liked".

The viewed products become a one-person history, then the same two stages as offline run:
candidates (FAISS + co-occurrence + popularity) and LightGBM ranking. Everything heavy
(similarity matrix, FAISS indexes, stats) is built once at startup.
"""

import dataclasses
from typing import Protocol

# LightGBM must be imported before FAISS: loaded the other way round, the two OpenMP runtimes
# crash on macOS. (isort keeps third-party imports in this order; don't move it below shelfsense.)
import lightgbm as lgb
import numpy as np
import pandas as pd

from shelfsense.baselines import latest_items, owned_items
from shelfsense.candidates import (
    Context,
    _add_item_and_user_features,
    _candidates_for_batch,
    build_context,
)
from shelfsense.config import (
    INTERACTIONS_FILE,
    PRODUCTS_FILE,
    RANKER_FILE,
)
from shelfsense.embeddings import load_embeddings
from shelfsense.features import FEATURES
from shelfsense.retrieval import EmbeddingRetriever

VISITOR = "__visitor__"
FLAGS = ["in_emb", "in_cooc", "in_pop", "in_cold"]


class Scorer(Protocol):
    def predict(self, data: pd.DataFrame) -> np.ndarray: ...


class Recommender:
    def __init__(
        self,
        vectors: np.ndarray,
        ids: list[str],
        products: pd.DataFrame,
        history: pd.DataFrame,
        ranker: Scorer,
    ):
        self.ctx: Context = build_context(
            history, products, EmbeddingRetriever(vectors, ids)
        )
        self.ranker = ranker
        self.products = products.set_index("parent_asin")
        popularity = history["parent_asin"].value_counts()
        # Only products with reviews are offered in search: those are the ones we can recommend from.
        self.searchable = (
            self.products.loc[popularity.index]
            .assign(n_reviews=popularity)
            .reset_index()
        )
        self.searchable["_lower"] = self.searchable["title"].fillna("").str.lower()
        self.now = history["timestamp"].max()

    @classmethod
    def load(cls) -> "Recommender":
        vectors, ids = load_embeddings()
        vectors = vectors.astype(
            np.float32
        )  # deploy bundles store float16; FAISS needs float32
        products = pd.read_parquet(PRODUCTS_FILE)
        history = pd.read_parquet(INTERACTIONS_FILE)
        ranker = lgb.Booster(model_file=str(RANKER_FILE))
        return cls(vectors, ids, products, history, ranker)

    def search(self, query: str, limit: int = 10) -> list[dict]:
        words = query.lower().split()
        if not words:
            return []
        mask = np.ones(len(self.searchable), dtype=bool)
        for word in words:
            mask &= self.searchable["_lower"].str.contains(word, regex=False).to_numpy()
        hits = self.searchable[mask].nlargest(limit, "n_reviews")
        return [self._describe(a) for a in hits["parent_asin"]]

    def _describe(self, asin: str, score: float | None = None) -> dict:
        row = self.products.loc[asin]
        price = row["price"]
        return {
            "asin": asin,
            "title": row["title"],
            "store": row["store"] if pd.notna(row["store"]) else None,
            "category": row["subcategory"],
            "price": None if pd.isna(price) else float(price),
            "score": score,
        }

    def recommend(self, viewed: list[str], k: int = 10) -> dict:
        """Top-k products for someone who viewed `viewed` (oldest first, newest last)."""
        viewed = [a for a in dict.fromkeys(viewed) if a in self.products.index]
        if not viewed:
            raise KeyError("none of the products are in the catalog")
        warm = [a for a in viewed if a in self.ctx.index.position]
        if not warm:
            # Brand-new products have no co-purchases or history: meaning (embeddings) is all we have.
            return self._embedding_only(viewed[-1], k)

        ctx = self._visitor_context(warm)
        cands = _candidates_for_batch(
            ctx, [VISITOR], n_emb=150, n_cooc=150, n_pop=100, n_cold=50
        )
        cands = _add_item_and_user_features(ctx, cands)
        cands[FLAGS] = cands[FLAGS].astype(np.int8)
        cands["score"] = self.ranker.predict(cands[FEATURES])
        top = cands.sort_values("score", ascending=False).head(k)
        return {
            "source": "two-stage",
            "viewed": viewed,
            "recommendations": [
                self._describe(a, float(s))
                for a, s in zip(top["parent_asin"], top["score"], strict=True)
            ],
        }

    def _visitor_context(self, warm: list[str]) -> Context:
        """The shared context, with the visitor's views swapped in as the only user history."""
        stamps = self.now + pd.to_timedelta(np.arange(len(warm)), unit="s")
        history = pd.DataFrame(
            {
                "user_id": VISITOR,
                "parent_asin": warm,
                "rating": 5,
                "timestamp": stamps,
            }
        )
        return dataclasses.replace(
            self.ctx,
            history=history,
            latest=latest_items(history),
            owned=owned_items(history),
        )

    def _embedding_only(self, asin: str, k: int) -> dict:
        similar = self.ctx.retriever.similar_items(asin, k)
        return {
            "source": "embedding-only (new product, no purchase history)",
            "viewed": [asin],
            "recommendations": [self._describe(a, s) for a, s in similar],
        }
