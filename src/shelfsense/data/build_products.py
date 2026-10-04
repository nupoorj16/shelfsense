"""Build a table of makeup + skincare products from the raw metadata file."""

from pathlib import Path

import pandas as pd

RAW_META = Path("data/raw/meta_Beauty_and_Personal_Care.jsonl.gz")
OUT = Path("data/interim/products.parquet")
KEEP_SUBCATEGORIES = ["Skin Care", "Makeup"]
COLUMNS = [
    "parent_asin",
    "title",
    "categories",
    "price",
    "average_rating",
    "rating_number",
    "store",
]
CHUNK_SIZE = 100_000


def main() -> None:
    kept = []
    reader = pd.read_json(RAW_META, lines=True, chunksize=CHUNK_SIZE)
    for i, chunk in enumerate(reader):
        chunk = chunk[COLUMNS]
        subcategory = chunk["categories"].str[1]
        good = chunk[subcategory.isin(KEEP_SUBCATEGORIES)]
        kept.append(good)
        print(f"chunk {i}: kept {len(good)} of {len(chunk)}")

    products = pd.concat(kept, ignore_index=True)
    products["subcategory"] = products["categories"].str[1]
    products["price"] = pd.to_numeric(products["price"], errors="coerce")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    products.to_parquet(OUT, index=False)
    print(f"Saved {len(products)} products to {OUT}")


if __name__ == "__main__":
    main()
