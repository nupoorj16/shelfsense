# ShelfSense — project memory for Claude

## What this is
A resume project: a two-stage product recommender for makeup + skincare ("customers who viewed this also liked").
Stage 1: retrieval (sentence-transformers embeddings + FAISS). Stage 2: ranking (LightGBM learning-to-rank).
Target roles: SWE/SDE I and AI Engineer. User is a beginner (Python rusty).

## Working mode (CHANGED 2026-10-04)
- Steps 1–2: mentor mode, user wrote the code (with heavy hints; download.py code was given by Claude).
- From step 3 on: user asked to finish fast ("done by tonight") → **Claude builds, user reviews**.
  Claude writes and runs the code, then gives a SHORT plain-language walkthrough + interview talking points per step.
- Keep explanations short and simple; define terms; no quizzes.
- Be honest in README "How I used AI": steps 1–2 user-written with mentoring; steps 3+ AI-written, user-reviewed.
- Commit at the end of each step (attribution line per system reminder).
- Keep updated: CLAUDE.md, LEARNING_LOG.md, RESUME_NOTES.md (real measured numbers only), README.md.

## Environment
macOS, Apple M2 Pro, 16 GB RAM, uv 0.12, git 2.50. Project Python 3.12 via uv.
System python3 (3.13) fails SSL downloads — always use `uv run`.

## Tools & decisions (the "why")
- Python 3.12 (not 3.13): ML libs lag on newest Python.
- uv (not pip/conda/poetry): fast, lock file → reproducible installs.
- src/ layout (`uv init --package`): code importable as `shelfsense`.
- ruff (not flake8+black+isort), pytest, pre-commit.
- stdlib urllib for download (no `requests` needed).
- Parquet via pyarrow (not CSV): smaller, faster, keeps dtypes.
- .gitignore uses root-anchored `/data/` (plain `data/` once hid src/shelfsense/data/ — bug, fixed).

## Data
Amazon Reviews 2023, McAuley Lab UCSD (Hou et al. 2024, arXiv:2403.03952). No formal license; research use →
cite, never commit raw data. Category Beauty_and_Personal_Care, filtered to categories[1] in {Skin Care, Makeup}.
- data/raw/: reviews 2.96 GB gz (23.9M reviews), meta 0.71 GB gz (1,028,914 products).
- data/interim/products.parquet: 320,042 products (Skin Care 199,381; Makeup 120,661). price: to_numeric coerce
  ('—' strings), 198,105 missing. Columns: parent_asin, title, categories, price, average_rating, rating_number, store, subcategory.
- data/interim/reviews.parquet: 7,426,398 reviews, 4,407,355 users, 319,981 products, 2000-04 → 2023-09.
  Reviews/user median 1. Users ≥5 reviews: 184,414. Duplicate (user, product) pairs: 100,171.
- Pipeline: `uv run python -m shelfsense.data.download` → `.build_products` → `.build_reviews`.

## Plan / progress
1. Project setup — DONE (41f89b3)
2. Download + explore — DONE (cbfc9e7, 51326a3)
3. Clean + time-based split — DONE (457f7e5)
4. Baselines (popularity, item co-occurrence) — DONE (d732aac)
5. Embeddings + FAISS retrieval — DONE (82ddc41)
6. LightGBM ranker — DONE (82ddc41)
7. Evaluate vs. baselines (Recall@K, NDCG@K) — DONE (ranker R@10 0.0302 vs popularity 0.0190)
8. Cold start — DONE (cold products Recall@10 = 0; cold users via popularity 0.026; see RESUME_NOTES)
9. MLflow tracking — DONE (SQLite store mlflow.db; view: `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`)
10. FastAPI service + demo page — DONE (a5a307b)
11. Tests, CI, Docker — DONE (21 tests, GitHub Actions, Dockerfile verified locally)
12. Deploy + README — README DONE; deploy pending (user accounts)  <- CURRENT (deploy needs user's own GitHub/host accounts)

## Gotchas
- macOS: FAISS + LightGBM in one process hang/segfault (two OpenMP copies). Pipeline runs `build` (FAISS, 1 thread) and `train` (LightGBM) as separate processes.
- Serving (`serving.py`): `import lightgbm` must come BEFORE faiss (else segfault in lgb.Booster). OpenMP thread count is per-thread, so `EmbeddingRetriever.search` sets faiss threads=1 on every call (FastAPI handlers run in worker threads).
- mlflow lives in the `train` dependency group (`uv run --group train ...`); the Docker image excludes it.
- Docker image has no data: mount data/interim, data/processed, artifacts at /app/... (README).
