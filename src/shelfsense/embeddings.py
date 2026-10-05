"""Turn every product's text into an embedding (a list of 384 numbers).

Products with similar meaning ("vitamin C serum" and "brightening face serum")
end up with similar numbers, even if they share no words.
"""

import numpy as np
import pandas as pd

from shelfsense.config import ARTIFACTS_DIR, PRODUCTS_FILE

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_FILE = ARTIFACTS_DIR / "item_embeddings.npy"
EMBEDDING_IDS_FILE = ARTIFACTS_DIR / "item_embedding_ids.parquet"


def product_text(products: pd.DataFrame) -> list[str]:
    title = products["title"].fillna("")
    store = products["store"].fillna("")
    return (title + " | " + products["subcategory"] + " | " + store).tolist()


def load_embeddings() -> tuple[np.ndarray, list[str]]:
    vectors = np.load(EMBEDDINGS_FILE)
    ids = pd.read_parquet(EMBEDDING_IDS_FILE)["parent_asin"].tolist()
    return vectors, ids


def main() -> None:
    from sentence_transformers import SentenceTransformer

    products = pd.read_parquet(PRODUCTS_FILE)
    texts = product_text(products)
    print(f"Embedding {len(texts):,} products with {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_FILE, vectors)
    products[["parent_asin"]].to_parquet(EMBEDDING_IDS_FILE, index=False)
    print(f"Saved {vectors.shape} to {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
