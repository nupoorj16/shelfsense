"""Keep only the reviews of our makeup + skincare products, with just the columns we need."""

from pathlib import Path

import pandas as pd

RAW_REVIEWS = Path("data/raw/Beauty_and_Personal_Care.jsonl.gz")
PRODUCTS = Path("data/interim/products.parquet")
OUT = Path("data/interim/reviews.parquet")
COLUMNS = ["user_id", "parent_asin", "rating", "timestamp"]
CHUNK_SIZE = 500_000


def main() -> None:
    products = pd.read_parquet(PRODUCTS, columns=["parent_asin"])
    product_ids = set(products["parent_asin"])
    print(f"Loaded {len(product_ids)} product IDs")

    kept = []
    reader = pd.read_json(RAW_REVIEWS, lines=True, chunksize=CHUNK_SIZE)
    for i, chunk in enumerate(reader):
        chunk = chunk[COLUMNS]
        good = chunk[chunk["parent_asin"].isin(product_ids)]
        kept.append(good)
        print(f"chunk {i}: kept {len(good)} of {len(chunk)}")

    reviews = pd.concat(kept, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    reviews.to_parquet(OUT, index=False)
    print(f"Saved {len(reviews)} reviews to {OUT}")
    print(f"Unique users: {reviews['user_id'].nunique()}")


if __name__ == "__main__":
    main()
