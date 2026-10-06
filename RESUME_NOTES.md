# Resume notes (real measured numbers only)

## Results — test split (2023), warm users only, 18,382 users
Ranker trained on the val split (2022-07 → 2022-12) only; test never seen in training.

| Model | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 |
|---|---|---|---|---|
| Co-occurrence | 0.0075 | 0.0053 | 0.0202 | 0.0084 |
| Popularity | 0.0190 | 0.0137 | 0.0564 | 0.0232 |
| Two-stage (FAISS + LightGBM) | 0.0302 | 0.0205 | 0.0789 | 0.0327 |

- Ranker vs. popularity: Recall@10 +59% relative, NDCG@10 +50%.
- Ceiling: only 61.8% of test purchases are of products seen in history, so absolute scores are low by design.

## Scale
- 320K products embedded; 7.4M reviews; ~10.7M (val) and ~7.2M (test) candidate pairs ranked by LightGBM LambdaRank.
- Candidates per user: 150 embedding (FAISS) + 150 co-occurrence + 100 popularity + 50 cold-product (embedding-only).

## Cold start (test)
- 38.2% of test purchases are of products with no history. Cold-product candidate source (50/user over ~270K unreviewed products) reached only 0.14% of them; cold-product Recall@10 = 0.
- Cold users (1,338): popularity fallback Recall@10 0.026, NDCG@10 0.026. Small sample.
- Honest framing: measured the problem and showed content-only retrieval is not enough at this pool size; next idea would be metadata/brand signals or a launch-recency prior.

## Draft bullet
Built a two-stage recommender (sentence-transformer embeddings + FAISS retrieval, LightGBM LambdaRank) on 7.4M Amazon
beauty reviews; on a time-based test split it beat a popularity baseline by 59% Recall@10 (0.030 vs 0.019).

## Caveat to state honestly
Single run, one seed, no confidence intervals yet. Steps 3+ AI-written, user-reviewed.

## MLflow
- Each pipeline training run is logged to MLflow (params, metrics for ranker + baselines, feature importance, model artifact). Re-run reproduced test Recall@10 0.0302 (seed fixed, early stop at iteration 96).

## Serving / engineering
- FastAPI service + demo page; recommendation latency ~0.35 s per request on an M2 (single viewed product, ~1.2M interactions, 320K products); startup ~8 s.
- 21 pytest tests, ruff, GitHub Actions CI, Dockerfile (image 830 MB; verified end to end in a Linux container).
