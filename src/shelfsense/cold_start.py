"""Step 8: how does the system do on brand-new products and brand-new users?

Cold item: a test purchase of a product with no reviews in the history. Only content
(embeddings) can reach it. Cold user: a test user with no history. Only popularity can serve them.
Run after the pipeline: `uv run python -m shelfsense.cold_start`.
"""

import json

import lightgbm as lgb
import pandas as pd

from shelfsense.baselines import popularity_recommender
from shelfsense.config import ARTIFACTS_DIR, RANKER_FILE
from shelfsense.evaluation import evaluate, load_eval_set
from shelfsense.ranker import rank


def _only(truth: dict[str, set[str]], keep: set[str], want_in: bool) -> dict:
    """Keep each user's truth items that are (want_in) or are not in `keep`; drop users left empty."""
    out = {
        u: {i for i in items if (i in keep) == want_in} for u, items in truth.items()
    }
    return {u: items for u, items in out.items() if items}


def cold_items() -> dict:
    es = load_eval_set("test", warm_only=True)
    hist_items = set(es.history["parent_asin"])
    cands = pd.read_parquet(ARTIFACTS_DIR / "cands_test.parquet")
    recs = rank(lgb.Booster(model_file=str(RANKER_FILE)), cands, 50)

    warm_truth = _only(es.truth, hist_items, True)
    cold_truth = _only(es.truth, hist_items, False)
    n_cold = sum(map(len, cold_truth.values()))
    n_all = sum(map(len, es.truth.values()))

    positives = cands[cands["label"] == 1]
    in_cands = positives["parent_asin"].map(lambda i: i not in hist_items).sum()
    top10 = [i for items in recs.values() for i in items[:10]]
    return {
        "cold_share_of_test_purchases": n_cold / n_all,
        "cold_purchases_in_candidates": int(in_cands) / n_cold,
        "cold_share_of_top10_recs": sum(i not in hist_items for i in top10)
        / len(top10),
        "ranker_on_warm_items": evaluate(recs, warm_truth),
        "ranker_on_cold_items": evaluate(recs, cold_truth),
    }


def cold_users() -> dict:
    es = load_eval_set("test", warm_only=False)
    recs = popularity_recommender(es.history, es.users, 50)
    return {"n_users": len(es.users), "popularity": evaluate(recs, es.truth)}


def main() -> None:
    results = {"cold_items": cold_items(), "cold_users": cold_users()}
    print(json.dumps(results, indent=2))
    (ARTIFACTS_DIR / "cold_start.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
