"""Column names the ranker is trained on (kept FAISS-free so training can import it)."""

FEATURES = [
    "emb_sim",
    "emb_max_sim",
    "cooc_score",
    "pop_30d",
    "pop_180d",
    "pop_all",
    "item_mean_rating",
    "item_age_days",
    "is_makeup",
    "price",
    "user_n_history",
    "user_subcat_share",
    "user_store_count",
    "in_emb",
    "in_cooc",
    "in_pop",
]
