"""Paths and settings shared across the pipeline."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = Path("artifacts")

PRODUCTS_FILE = INTERIM_DIR / "products.parquet"
REVIEWS_FILE = INTERIM_DIR / "reviews.parquet"
INTERACTIONS_FILE = PROCESSED_DIR / "interactions.parquet"
ITEMS_FILE = PROCESSED_DIR / "items.parquet"

# Every user and every product must have at least this many reviews.
MIN_INTERACTIONS = 5

# Time-based split: train < VAL_START <= val < TEST_START <= test.
VAL_START = pd.Timestamp("2022-07-01")
TEST_START = pd.Timestamp("2023-01-01")

SEED = 42
