# ShelfSense

A two-stage recommender for makeup and skincare: "customers who viewed this also liked". It retrieves a few hundred candidate
products with sentence embeddings and FAISS, then re-ranks them with a LightGBM LambdaRank model. It is served by a FastAPI
app with a small demo page.

Data: Amazon Reviews 2023, Beauty & Personal Care, filtered to Skin Care and Makeup (McAuley Lab, UCSD; Hou et al. 2024,
[arXiv:2403.03952](https://arxiv.org/abs/2403.03952)). Raw data is not included; the download script fetches it.

## How it works

```
viewed product(s) ─► candidates (~350 per user) ─► LightGBM ranker ─► top-k
                     ├─ embedding similarity (FAISS, all-MiniLM-L6-v2)
                     ├─ co-purchase similarity ("bought together")
                     ├─ recent popularity
                     └─ new products (embedding-only)
```

The ranker scores each candidate using 17 features: embedding similarities, co-purchase score, 30/180-day and all-time
popularity, item age and mean rating, price, category, and how well the product matches the user's category and brand history.
All features are computed from history only, never from the period being predicted.

## Results

Time-based split: train before 2022-07, validation to 2022-12, test 2023. Test is 18,382 users with at least one earlier review.
The ranker is trained on the validation period only, so test is never seen in training.

| Model | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 |
|---|---|---|---|---|
| Co-occurrence | 0.0075 | 0.0053 | 0.0202 | 0.0084 |
| Popularity | 0.0190 | 0.0137 | 0.0564 | 0.0232 |
| **Two-stage (FAISS + LightGBM)** | **0.0302** | **0.0205** | **0.0789** | **0.0327** |

The two-stage model improves Recall@10 over the popularity baseline by 59% (relative). Absolute numbers are low by design:
only 61.8% of test purchases are of products that have history, and the task is to pick about 10 of ~47K products.

Scale: 7.4M raw reviews and 320K products embedded; after de-duplication and a 5-core filter, 1.2M interactions, 136K users
and 47K products. About 10.7M (validation) and 7.2M (test) candidate pairs are ranked.

### What did not work: cold start
- **New products** (38% of test purchases have no earlier reviews): a dedicated embedding-only candidate source over ~270K
  unreviewed products reached only 0.14% of them, so Recall@10 on new products is 0. Content embeddings alone cannot find the few
  products that take off in a pool this large. Next ideas: brand and metadata signals, or a launch-recency prior.
- **New users** (1,338, no history): a popularity fallback gets Recall@10 0.026. Small sample.

### Limitations
- One run, one seed, no confidence intervals.
- The demo recommends from a single viewed product. The ranker was trained on users with several past reviews, so item-to-item
  results are looser than the offline numbers suggest.
- Offline metrics on reviews are a proxy: a review is not a click or a purchase.

## Run it

```
uv sync
# data (several GB) and pipeline
uv run python -m shelfsense.data.download && uv run python -m shelfsense.data.build_products && uv run python -m shelfsense.data.build_reviews
uv run python -m shelfsense.data.prepare                # dedupe, 5-core filter, time-based split
uv run --group train python -m shelfsense.embeddings       # embeds all products (slow on CPU)
uv run --group train python -m shelfsense.pipeline         # builds candidates, trains the ranker, logs to MLflow
```

Serve (needs the files the pipeline writes to `data/` and `artifacts/`):

```
uv run uvicorn shelfsense.api:app      # demo page at http://127.0.0.1:8000, API docs at /docs
```

Endpoints: `GET /search?q=...`, `GET /recommend?asin=...&k=10` (repeat `asin` for several viewed products, newest last), `GET /health`.
Startup takes about 8 s and a recommendation about 0.35 s on an M2 Pro.

Docker (model and data are mounted, not baked into the image):

```
docker build -t shelfsense .
docker run -p 8000:8000 \
  -v $PWD/data/interim:/app/data/interim:ro -v $PWD/data/processed:/app/data/processed:ro \
  -v $PWD/artifacts:/app/artifacts:ro shelfsense
```

Deploy (Hugging Face Space, Docker SDK): `uv run python -m shelfsense.deploy` builds `space/` (75 MB: embeddings for the 47K products
with history in float16, no new-product pool, no raw data). On the 10 test products tried, its top-10 lists matched the full model's
exactly. Push that folder to a new Space. The demo needs about 1 GB of RAM.

Tests and lint: `uv run pytest` and `uv run ruff check .` (also run by GitHub Actions on every push).
Experiment history: `uv run --group train mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Engineering notes
- **FAISS and LightGBM clash on macOS.** They ship different OpenMP runtimes, and loading both crashes. The pipeline runs retrieval
  and training as separate processes, and the server imports LightGBM first and pins FAISS to one thread per search.
- **Exact search.** FAISS `IndexFlatIP` is exact and fast enough at 320K vectors; at millions of products I would switch to HNSW or IVF.
- **Reproducibility.** `uv.lock` pins dependencies, a fixed seed is used, and every training run is logged to MLflow.

## How I used AI
Steps 1–2 (project setup, data download and exploration) I wrote myself with mentoring. From step 3 on (cleaning and split,
baselines, embeddings, ranker, evaluation, cold start, MLflow, API, tests, CI, Docker) the code was written by Claude (Anthropic's
coding assistant) and I reviewed it, ran it, and own the results and the decisions described here.
