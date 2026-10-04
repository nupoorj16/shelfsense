"""Clean the interim reviews and split them by time into train / val / test."""

import json

import pandas as pd

from shelfsense.config import (
    INTERACTIONS_FILE,
    ITEMS_FILE,
    MIN_INTERACTIONS,
    PROCESSED_DIR,
    PRODUCTS_FILE,
    REVIEWS_FILE,
    TEST_START,
    VAL_START,
)


def drop_duplicates(reviews: pd.DataFrame) -> pd.DataFrame:
    """Keep only the first time each user reviewed each product."""
    reviews = reviews.sort_values("timestamp", kind="stable")
    return reviews.drop_duplicates(["user_id", "parent_asin"], keep="first")


def k_core(reviews: pd.DataFrame, k: int) -> pd.DataFrame:
    """Repeatedly drop users and products with fewer than k reviews.

    Dropping a user can push a product below k (and vice versa), so we loop
    until nothing changes.
    """
    while True:
        user_counts = reviews["user_id"].map(reviews["user_id"].value_counts())
        item_counts = reviews["parent_asin"].map(reviews["parent_asin"].value_counts())
        keep = (user_counts >= k) & (item_counts >= k)
        if keep.all():
            return reviews
        reviews = reviews[keep]


def assign_split(timestamps: pd.Series) -> pd.Series:
    split = pd.Series("train", index=timestamps.index)
    split[timestamps >= VAL_START] = "val"
    split[timestamps >= TEST_START] = "test"
    return split


def main() -> None:
    reviews = pd.read_parquet(REVIEWS_FILE)
    print(f"Raw reviews: {len(reviews):,}")

    reviews = drop_duplicates(reviews)
    print(f"After removing duplicates: {len(reviews):,}")

    reviews = k_core(reviews, MIN_INTERACTIONS)
    reviews = reviews.reset_index(drop=True)
    reviews["split"] = assign_split(reviews["timestamp"])
    print(f"After {MIN_INTERACTIONS}-core: {len(reviews):,}")

    products = pd.read_parquet(PRODUCTS_FILE)
    items = products[products["parent_asin"].isin(reviews["parent_asin"])].reset_index(
        drop=True
    )
    items = items.drop(columns=["categories"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    reviews.to_parquet(INTERACTIONS_FILE, index=False)
    items.to_parquet(ITEMS_FILE, index=False)

    stats = {
        "interactions": len(reviews),
        "users": int(reviews["user_id"].nunique()),
        "items": int(reviews["parent_asin"].nunique()),
        "split_sizes": reviews["split"].value_counts().to_dict(),
        "first_date": str(reviews["timestamp"].min().date()),
        "last_date": str(reviews["timestamp"].max().date()),
    }
    (PROCESSED_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
