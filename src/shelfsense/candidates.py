"""Build candidate (user, product) pairs and describe each pair with features.

Candidates come from three cheap sources (embeddings, co-occurrence, popularity).
The ranker then scores each pair using the features computed here.

Every feature is computed from the *history* only, never from the period we
are predicting. (The product metadata's average_rating / rating_number were
crawled in 2023, after our cutoffs, so we deliberately do not use them.)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from shelfsense.baselines import (
    ItemIndex,
    cooccurrence_similarity,
    latest_items,
    owned_items,
    recent_popular,
    seen_positions,
    user_profiles,
)
from shelfsense.evaluation import top_k_excluding_seen
from shelfsense.retrieval import EmbeddingRetriever


@dataclass
class Context:
    """Everything precomputed from the history, reused for every batch of users."""

    history: pd.DataFrame
    index: ItemIndex
    similarity: sparse.csr_matrix
    latest: pd.DataFrame
    owned: dict[str, set[str]]
    popular: list[str]
    item_stats: pd.DataFrame
    products: pd.DataFrame
    retriever: EmbeddingRetriever  # warm items only (have history)
    cold_retriever: EmbeddingRetriever  # every embedded product with no history


def item_statistics(history: pd.DataFrame) -> pd.DataFrame:
    now = history["timestamp"].max()
    grouped = history.groupby("parent_asin")
    stats = pd.DataFrame(
        {
            "pop_all": grouped.size(),
            "item_mean_rating": grouped["rating"].mean(),
            "item_age_days": (now - grouped["timestamp"].min()).dt.days,
        }
    )
    for days in (30, 180):
        recent = history[history["timestamp"] >= now - pd.Timedelta(days=days)]
        stats[f"pop_{days}d"] = recent["parent_asin"].value_counts()
    return stats.fillna({"pop_30d": 0, "pop_180d": 0})


def build_context(
    history: pd.DataFrame, products: pd.DataFrame, retriever: EmbeddingRetriever
) -> Context:
    hist_items = set(history["parent_asin"])
    index = ItemIndex(sorted(hist_items))
    return Context(
        history=history,
        index=index,
        similarity=cooccurrence_similarity(history, index),
        latest=latest_items(history),
        owned=owned_items(history),
        popular=recent_popular(history).index.tolist(),
        item_stats=item_statistics(history),
        products=products.set_index("parent_asin")[["subcategory", "store", "price"]],
        retriever=retriever.restricted_to(hist_items),
        cold_retriever=retriever.restricted_to(set(retriever.row_of) - hist_items),
    )


def _candidates_for_batch(
    ctx: Context, users: list[str], n_emb: int, n_cooc: int, n_pop: int, n_cold: int
) -> pd.DataFrame:
    # Source 1: co-occurrence ("bought together"). Keep the full score matrix for features.
    cooc = (user_profiles(ctx.latest, users, ctx.index) @ ctx.similarity).toarray()
    cooc_top = top_k_excluding_seen(
        cooc, seen_positions(ctx.owned, users, ctx.index), n_cooc
    )

    # Source 2: embeddings (similar meaning) over products that have history.
    user_vecs = ctx.retriever.user_vectors(ctx.history, users)
    owned = [ctx.owned.get(u, set()) for u in users]
    emb_items, _ = ctx.retriever.search(user_vecs, n_emb, owned)
    # Source 4: brand-new products. Only meaning (embeddings) can reach them.
    cold_items, _ = ctx.cold_retriever.search(user_vecs, n_cold)

    rows = []
    for r, user in enumerate(users):
        items: dict[str, list[bool]] = {}
        for item in emb_items[r]:
            items.setdefault(item, [False] * 4)[0] = True
        for item in cold_items[r]:
            items.setdefault(item, [False] * 4)[3] = True
        for pos in cooc_top[r]:
            if cooc[r, pos] > 0:
                items.setdefault(ctx.index.items[pos], [False] * 4)[1] = True
        added = 0
        for item in ctx.popular:
            if added == n_pop:
                break
            if item not in owned[r]:
                items.setdefault(item, [False] * 4)[2] = True
                added += 1
        for item, (e, c, p, n) in items.items():
            pos = ctx.index.position.get(item)
            rows.append(
                (user, item, e, c, p, n, cooc[r, pos] if pos is not None else 0.0)
            )

    cands = pd.DataFrame(
        rows,
        columns=[
            "user_id",
            "parent_asin",
            "in_emb",
            "in_cooc",
            "in_pop",
            "in_cold",
            "cooc_score",
        ],
    )
    _add_embedding_features(ctx, cands, users, user_vecs)
    return cands


def _add_embedding_features(
    ctx: Context, cands: pd.DataFrame, users: list[str], user_vecs: np.ndarray
) -> None:
    retriever = ctx.retriever
    item_rows = cands["parent_asin"].map(retriever.row_of).to_numpy()
    item_vecs = retriever.vectors[item_rows]
    user_pos = cands["user_id"].map({u: i for i, u in enumerate(users)}).to_numpy()
    cands["emb_sim"] = np.einsum("ij,ij->i", item_vecs, user_vecs[user_pos])

    # Max similarity to any single recent item: "this is very close to ONE thing you bought".
    recent = ctx.latest[ctx.latest["user_id"].isin(users)]
    recent_rows = recent.groupby("user_id")["parent_asin"].agg(
        lambda s: [retriever.row_of[i] for i in s if i in retriever.row_of]
    )
    max_sim = np.zeros(len(cands), dtype=np.float32)
    for user, idx in cands.groupby("user_id", sort=False).indices.items():
        rows = recent_rows.get(user, [])
        if rows:
            max_sim[idx] = (item_vecs[idx] @ retriever.vectors[rows].T).max(axis=1)
    cands["emb_max_sim"] = max_sim


def _add_item_and_user_features(ctx: Context, cands: pd.DataFrame) -> pd.DataFrame:
    cands = cands.join(ctx.item_stats, on="parent_asin")
    cands = cands.join(ctx.products, on="parent_asin")
    cands[["pop_30d", "pop_180d", "pop_all"]] = cands[
        ["pop_30d", "pop_180d", "pop_all"]
    ].fillna(0)
    cands["is_makeup"] = (cands["subcategory"] == "Makeup").astype(np.int8)

    hist = ctx.history[ctx.history["user_id"].isin(cands["user_id"].unique())]
    hist = hist.join(ctx.products[["subcategory", "store"]], on="parent_asin")
    n_hist = hist.groupby("user_id").size().rename("user_n_history")
    subcat = hist.groupby(["user_id", "subcategory"]).size().rename("subcat_n")
    store = hist.groupby(["user_id", "store"]).size().rename("user_store_count")

    cands = cands.join(n_hist, on="user_id")
    cands = cands.join(subcat, on=["user_id", "subcategory"])
    cands = cands.join(store, on=["user_id", "store"])
    cands["user_subcat_share"] = cands["subcat_n"].fillna(0) / cands["user_n_history"]
    cands["user_store_count"] = cands["user_store_count"].fillna(0)
    return cands.drop(columns=["subcat_n", "subcategory", "store"])


def build_candidates(
    ctx: Context,
    users: list[str],
    n_emb: int = 100,
    n_cooc: int = 100,
    n_pop: int = 50,
    n_cold: int = 50,
    batch_size: int = 2000,
) -> pd.DataFrame:
    parts = []
    for start in range(0, len(users), batch_size):
        batch = users[start : start + batch_size]
        parts.append(_candidates_for_batch(ctx, batch, n_emb, n_cooc, n_pop, n_cold))
    cands = pd.concat(parts, ignore_index=True)
    cands = _add_item_and_user_features(ctx, cands)
    flags = ["in_emb", "in_cooc", "in_pop", "in_cold"]
    cands[flags] = cands[flags].astype(np.int8)
    return cands
