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
| **Two-stage** | **0.0302** | **0.0205** | **0.0789** |

Time-based split (train < 2022-07, val to 2022-12, test 2023). The ranker is trained on val only. Only 61.8% of test
purchases are of previously seen products, which caps absolute scores.

## Cold start (test split)
- **Cold products** (38% of test purchases have no prior reviews): a dedicated embedding-only candidate source over all ~270K
  unreviewed products found only 0.14% of them in the top 50, so Recall@10 on cold products is 0. Content embeddings alone cannot
  pick the few products that take off from a pool this large. Warm-product Recall@10 is 0.033.
- **Cold users** (1,338 users with no history): popularity fallback gets Recall@10 0.026 (small sample).

## Run

```
uv run python -m shelfsense.data.download && uv run python -m shelfsense.data.build_products && uv run python -m shelfsense.data.build_reviews
uv run --group train python -m shelfsense.pipeline   # trains the ranker, logs to MLflow
```

## Serve

```
uv run uvicorn shelfsense.api:app          # http://127.0.0.1:8000 (demo page), /docs (API)
```
Endpoints: `GET /search?q=...`, `GET /recommend?asin=...&k=10` (repeat `asin` for several viewed products), `GET /health`.
A one-item history is far less information than the ranker saw in training, so item-to-item results are looser than the offline numbers.

Docker (model and data are mounted, not baked into the image):
```
docker build -t shelfsense .
docker run -p 8000:8000 -v $PWD/data/interim:/app/data/interim:ro -v $PWD/data/processed:/app/data/processed:ro -v $PWD/artifacts:/app/artifacts:ro shelfsense
```
Tests and lint: `uv run pytest` and `uv run ruff check .` (also run by GitHub Actions on every push).
Experiment history: `uv run --group train mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## How I used AI
Steps 1–2 (setup, data download/exploration) I wrote myself with mentoring. Steps 3 onward were written by Claude and reviewed by me.
