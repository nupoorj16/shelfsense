"""Full recommender: train ranker, evaluate on val/test, compare with baselines."""

import json
import subprocess
import sys

import pandas as pd

from shelfsense.config import ARTIFACTS_DIR, PRODUCTS_FILE, RANKER_FILE
from shelfsense.evaluation import evaluate, load_eval_set
from shelfsense.ranker import label_candidates, rank, train_ranker


def _prepare(split, items, retriever):
    from shelfsense.candidates import build_candidates, build_context

    es = load_eval_set(split, warm_only=True)
    hist_items = set(es.history["parent_asin"])
    truth_items = sum(len(v) for v in es.truth.values())
    reach = sum(len(v & hist_items) for v in es.truth.values())
    print(
        f"users: {len(es.users):,}, reachable: {reach}/{truth_items} ({reach / truth_items:.1%})"
    )
    ctx = build_context(es.history, items, retriever)
    cands = build_candidates(ctx, es.users, n_emb=150, n_cooc=150, n_pop=100, n_cold=50)
    print(f"  {len(cands):,} candidate pairs")
    return es, cands


def build() -> None:
    """Retrieval + candidate generation (FAISS). Saves labeled candidates and baseline scores."""
    import faiss  # noqa: F401  # must load before pyarrow/LightGBM; see train_and_eval

    from shelfsense.baselines import cooccurrence_recommender, popularity_recommender
    from shelfsense.embeddings import load_embeddings
    from shelfsense.retrieval import EmbeddingRetriever

    print("Loading data...")
    vectors, ids = load_embeddings()
    items = pd.read_parquet(PRODUCTS_FILE)
    retriever = EmbeddingRetriever(vectors, ids)
    baselines = {}
    for split in ("val", "test"):
        print(f"\n=== {split.upper()} ===")
        es, cands = _prepare(split, items, retriever)
        label_candidates(cands, es.truth).to_parquet(
            ARTIFACTS_DIR / f"cands_{split}.parquet"
        )
        if split == "test":
            baselines = {
                "popularity": evaluate(
                    popularity_recommender(es.history, es.users, 50), es.truth
                ),
                "cooccurrence": evaluate(
                    cooccurrence_recommender(es.history, es.users, 50), es.truth
                ),
            }
    (ARTIFACTS_DIR / "baselines.json").write_text(json.dumps(baselines, indent=2))


def train_and_eval() -> None:
    """Train on val candidates, score on test. Runs without FAISS loaded (OpenMP clash)."""
    val = pd.read_parquet(ARTIFACTS_DIR / "cands_val.parquet")
    ranker = train_ranker(val, num_rounds=200)
    ranker.save_model(str(RANKER_FILE))
    test = pd.read_parquet(ARTIFACTS_DIR / "cands_test.parquet")
    truth = load_eval_set("test", warm_only=True).truth
    results = json.loads((ARTIFACTS_DIR / "baselines.json").read_text())
    results["ranker"] = evaluate(rank(ranker, test, 50), truth)
    for name, scores in results.items():
        print(name, {k: round(v, 4) for k, v in scores.items()})
    (ARTIFACTS_DIR / "results.json").write_text(json.dumps(results, indent=2))


def main() -> None:
    subprocess.run(
        [sys.executable, "-u", "-m", "shelfsense.pipeline", "build"], check=True
    )
    subprocess.run(
        [sys.executable, "-u", "-m", "shelfsense.pipeline", "train"], check=True
    )


if __name__ == "__main__":
    {"build": build, "train": train_and_eval}.get(
        sys.argv[1] if len(sys.argv) > 1 else "", main
    )()
