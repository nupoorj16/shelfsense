"""Build a small, self-contained folder (`space/`) to deploy the demo, e.g. to a Hugging Face Space.

The full system needs ~1 GB of files. The demo only recommends products that have purchase history
(~47K), so we keep just those: embeddings in float16 and no new-product pool (it scored 0 anyway).
Run: `uv run python -m shelfsense.deploy`
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from shelfsense.config import INTERACTIONS_FILE, PRODUCTS_FILE, RANKER_FILE
from shelfsense.embeddings import EMBEDDING_IDS_FILE, EMBEDDINGS_FILE, load_embeddings

OUT = Path("space")


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    interactions = pd.read_parquet(INTERACTIONS_FILE)
    keep = set(interactions["parent_asin"])

    products = pd.read_parquet(PRODUCTS_FILE)
    products = products[products["parent_asin"].isin(keep)]
    vectors, ids = load_embeddings()
    rows = [i for i, asin in enumerate(ids) if asin in keep]

    _write(interactions, OUT / INTERACTIONS_FILE)
    _write(products, OUT / PRODUCTS_FILE)
    _write(
        pd.DataFrame({"parent_asin": [ids[i] for i in rows]}),
        OUT / EMBEDDING_IDS_FILE,
    )
    np.save(OUT / EMBEDDINGS_FILE, vectors[rows].astype(np.float16))
    shutil.copy(RANKER_FILE, OUT / RANKER_FILE)

    shutil.copytree("src", OUT / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy(name, OUT / name)
    shutil.copy("deploy/Dockerfile", OUT / "Dockerfile")
    shutil.copy("deploy/SPACE_README.md", OUT / "README.md")
    # The Docker build needs a README.md next to pyproject.toml, and Spaces reads its settings from it.

    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(
        f"{len(keep):,} products, {len(interactions):,} interactions -> {OUT}/ ({size / 1e6:.0f} MB)"
    )


if __name__ == "__main__":
    main()
