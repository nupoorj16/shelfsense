"""Stage 1: find a few hundred candidate products per user with FAISS.

A user's "taste vector" is the average embedding of the products they reviewed
recently (newer reviews count more). FAISS then finds the products whose
embeddings point in the most similar direction.
"""

import faiss
import numpy as np
import pandas as pd

from shelfsense.baselines import latest_items, owned_items


class EmbeddingRetriever:
    def __init__(
        self, vectors: np.ndarray, ids: list[str], catalog: set[str] | None = None
    ):
        """vectors/ids cover every embedded product; catalog limits what can be recommended."""
        self.row_of = {item: i for i, item in enumerate(ids)}
        self.vectors = vectors
        keep = [i for i, item in enumerate(ids) if catalog is None or item in catalog]
        self.catalog_ids = [ids[i] for i in keep]
        # Inner product on normalized vectors = cosine similarity.
        # Exact search is fast enough for ~300K items; at millions we'd use HNSW or IVF.
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors[keep])

    def vector_for(self, item: str) -> np.ndarray | None:
        row = self.row_of.get(item)
        return None if row is None else self.vectors[row]

    def user_vectors(self, history: pd.DataFrame, user_ids: list[str]) -> np.ndarray:
        latest = latest_items(history)
        latest = latest[latest["user_id"].isin(user_ids)]
        dim = self.vectors.shape[1]
        out = np.zeros((len(user_ids), dim), dtype=np.float32)
        position = {u: i for i, u in enumerate(user_ids)}
        rows = latest["parent_asin"].map(self.row_of)
        latest = latest[rows.notna()]
        rows = rows.dropna().astype(int).to_numpy()
        weighted = self.vectors[rows] * latest["weight"].to_numpy(np.float32)[:, None]
        np.add.at(out, latest["user_id"].map(position).to_numpy(), weighted)
        faiss.normalize_L2(out)
        return out

    def search(
        self, queries: np.ndarray, k: int, exclude: list[set[str]] | None = None
    ) -> tuple[list[list[str]], list[list[float]]]:
        extra = max((len(e) for e in exclude), default=0) if exclude else 0
        scores, idx = self.index.search(queries, k + extra)
        items_out, scores_out = [], []
        for row in range(len(queries)):
            skip = exclude[row] if exclude else set()
            items, sims = [], []
            for j, s in zip(idx[row], scores[row], strict=True):
                if j < 0:
                    continue
                item = self.catalog_ids[j]
                if item in skip:
                    continue
                items.append(item)
                sims.append(float(s))
                if len(items) == k:
                    break
            items_out.append(items)
            scores_out.append(sims)
        return items_out, scores_out

    def recommend(
        self, history: pd.DataFrame, user_ids: list[str], k: int
    ) -> tuple[list[list[str]], list[list[float]]]:
        owned = owned_items(history)
        queries = self.user_vectors(history, user_ids)
        return self.search(queries, k, [owned.get(u, set()) for u in user_ids])

    def similar_items(self, item: str, k: int) -> list[tuple[str, float]]:
        vector = self.vector_for(item)
        if vector is None:
            return []
        items, scores = self.search(vector[None, :], k, [{item}])
        return list(zip(items[0], scores[0], strict=True))
