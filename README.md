# ShelfSense

A two-stage product recommender for makeup and skincare ("customers who viewed this also liked"), built on the
Amazon Reviews 2023 dataset (McAuley Lab, UCSD; Hou et al. 2024, arXiv:2403.03952).

1. **Retrieval:** sentence-transformer embeddings + FAISS find ~400 candidates per user.
2. **Ranking:** a LightGBM LambdaRank model orders the candidates.

## Results (test split, 2023, 18,382 warm users)

| Model | Recall@10 | NDCG@10 | Recall@50 |
|---|---|---|---|
| Co-occurrence | 0.0075 | 0.0053 | 0.0202 |
| Popularity | 0.0190 | 0.0137 | 0.0564 |
| **Two-stage** | **0.0304** | **0.0202** | **0.0799** |

Time-based split (train < 2022-07, val to 2022-12, test 2023). The ranker is trained on val only. Only 61.8% of test
purchases are of previously seen products, which caps absolute scores.

## Run

```
uv run python -m shelfsense.data.download && uv run python -m shelfsense.data.build_products && uv run python -m shelfsense.data.build_reviews
uv run python -m shelfsense.pipeline
```

## How I used AI
Steps 1–2 (setup, data download/exploration) I wrote myself with mentoring. Steps 3 onward were written by Claude and reviewed by me.
