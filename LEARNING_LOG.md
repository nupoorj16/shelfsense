# Learning Log

Key ideas from each step, in plain language, plus "why we chose X over Y".

---

## Step 1: Project setup

### Concepts
- **Recommender system (two stages):** like a bookstore clerk. First grab a rough pile of maybe-good items fast (**retrieval**), then sort that pile carefully (**ranking**). We split it because carefully scoring *every* item is too slow.
- **Package:** code someone else wrote that you install and use (e.g., pandas).
- **Dependency:** a package your project needs. A **dev dependency** is only needed while building (ruff, pytest), not while running — like a carpenter's saw that doesn't ship with the chair.
- **Virtual environment (`.venv`):** a private box of Python + packages just for this project, so projects don't fight over versions.
- **`pyproject.toml`:** the project's ID card (name, Python version, dependencies).
- **Lock file (`uv.lock`):** the *exact* version of every package, so everyone gets an identical setup.
- **`.gitignore`:** a list of files git must never save (`.venv`, data, secrets).
- **Staging area:** the "waiting room" for your next commit. `git add` puts files there; `git commit` saves the snapshot.
- **Test / smoke test:** a tiny program that checks your code automatically. A smoke test is the simplest one: "does it even start?"
- **Linter / formatter:** a code spell-checker (finds mistakes) and a code tidier (fixes spacing/quotes).
- **pre-commit hook:** a checklist that runs automatically on every `git commit`.

### Debugging lessons
- **Reading a pytest error:** `file.py:7` = file and line. The line starting with `E` is the actual error. "collected 0 items" = pytest found no functions starting with `test_`.
- **Reading a ruff error:** `I001` is the rule code (searchable). In the `help:` box, `-` lines get removed and `+` lines get added. `[*]` = ruff can fix it with `--fix`.
- **"Skipped (no files to check)" in pre-commit:** pre-commit only checks files git tracks. Run `git add` first.
- **Where am I?** `.` means "the current folder". Running `git add .` inside `tests/` only adds `tests/`. Check with `pwd`; run commands from the project root.
- **`git commit --amend`:** adds changes to the last commit. Only safe *before* pushing.

### Why X over Y
| Chose | Over | Why |
|---|---|---|
| uv | pip, conda, poetry | Much faster; lock file gives reproducible installs (matters for Docker/CI later); manages Python versions too. |
| Python 3.12 | 3.13 | ML libraries (FAISS, LightGBM, PyTorch) support the newest Python last. |
| `src/` layout | code in project root | Tests import the *installed* package, catching packaging mistakes early. |
| ruff | flake8 + black + isort | One fast tool replaces three. |
| pytest | unittest | Less boilerplate (plain `assert`), the industry default. |
| pre-commit | "remember to run ruff" | Automatic — humans forget, hooks don't. |

---

## Step 2: Data (download, explore, filter)

### Concepts
- **JSONL:** one JSON record per line, so huge files can be read a piece at a time.
- **Chunked reading:** read 100K–500K rows at a time instead of the whole 24M-row file (pizza slice by slice).
- **Biased sample:** the review file is sorted by user, so "first 100K rows" over-represents heavy reviewers. Always ask how a sample was taken.
- **Outliers:** one "user" had 2,973 reviews in the sample. The median user had 1–3. Averages lie when outliers exist; look at the median.
- **Sparsity:** most users touch a tiny fraction of products (median user in the full data: 1 review). The core challenge of recommenders.
- **Boolean filtering:** `df[mask]` keeps rows where `mask` is True (like a stencil).
- **Data cleaning:** the price column mixed numbers, None and the text "—". `pd.to_numeric(errors="coerce")` turns junk into NaN (missing).

### Debugging lessons
- `KeyError: 'reviewerID'` → that column doesn't exist (it was the 2018 dataset's name). Check `df.columns`.
- `NameError` → you used a variable that was never created in *this* file.
- A program that prints nothing: check the file is saved and non-empty.
- Silent git bug: `.gitignore` line `data/` ignored *every* folder named data, including our code. `/data/` anchors it to the project root. Always read `git status` before committing.
- pre-commit "Failed – files were modified" means it fixed things. `git add -u` and commit again.

### Why X over Y
| Chose | Over | Why |
|---|---|---|
| Amazon Reviews 2023 | older 2018 version | Newer, bigger, cleaner metadata, documented. |
| Makeup + Skin Care | whole Beauty category / other categories | Domain I know (can sanity-check results), rich product text, constant new launches (cold start). |
| Parquet | CSV | ~10x smaller/faster, keeps data types (dates stay dates, lists stay lists). |
| stdlib `urllib` | `requests` | No new dependency needed for a simple download. |

---

## Step 3: Clean + time split

### Concepts
- **Duplicates:** ~100K times the same person reviewed the same product twice. We keep the first one.
- **k-core (k=5):** repeatedly drop users and products with fewer than 5 reviews, until stable. You can't learn taste from 1 review. Result: 1.2M reviews, 136K users, 47K products.
- **Time-based split:** train = before 2022-07-01, validation = 2022-07-01 to 2022-12-31, test = 2023. Like real life: learn from the past, predict the future.
- **Data leakage:** letting future information into training (e.g., random splits). Makes scores look great but fake.
- **Warm vs. cold:** a "warm" user has history in training. A "cold" user/product is brand new. In our test period, about half of reviews are for products that did not exist in training. Beauty launches constantly.
- **Validation set:** a middle period used to tune/train the ranker without touching the test set. The test set is only for the final score.

### Known limitation (be honest in interviews)
- The 5-core filter is computed over all years, so it slightly uses future info to decide which users/products to keep. This is the standard benchmark practice (same as the dataset authors' 5-core files), but a stricter setup would filter on training data only.

---

## Step 7: Evaluate ranker vs. baselines

### Concepts
- **Leakage bug caught:** the first draft trained the ranker on a split's answers and scored it on the same answers. Fix: train on val, score only on test.
- **Recall@K / NDCG@K:** did the real purchases show up in the top K, and how high. Test result: ranker 0.030 Recall@10 vs. popularity 0.019.
- **OpenMP clash (macOS):** FAISS and LightGBM each bundle their own thread library. Used together in one process, LightGBM hung forever and FAISS segfaulted.
  Fix: FAISS pinned to 1 thread, and the pipeline runs as two processes (`build` with FAISS, `train` with LightGBM). Candidates pass between them via parquet.
- Debugging method worth mentioning in interviews: `ps` showed 0% CPU, `sample` showed the exact stuck thread barrier.

---

## Step 8: Cold start

### Concepts
- **Cold item / cold user:** a product or person with no history. Co-occurrence and popularity cannot see them; only content (embeddings) can.
- **Measured first:** 38% of test purchases are cold products, but the original system put only ~1% of them in its candidates and recommended almost none.
- **Survivorship trap avoided:** the 5-core catalog has only ~2K products without history, and they are there *because* they later got 5+ reviews. Searching only those almost gives away the answer. Fair pool: every embedded product with no history (~270K).
  Product features come from the full product table, so missing metadata cannot reveal which products will be reviewed. This also made the warm retrieval honest (warm retriever only sees products with history).
- **Result (negative, and that is fine):** cold-product Recall@10 = 0 (0.14% of cold purchases reach the candidates). Cold users via popularity: Recall@10 0.026.

## Step 9: MLflow tracking

### Concepts
- **Experiment tracking:** every training run records its settings (params), scores (metrics) and files (artifacts), so you can compare runs later instead of trusting memory.
- **What is logged per run:** LightGBM params, seed, best iteration (early stopping), train/test candidate counts, Recall/NDCG@10/50 for ranker + both baselines, feature importance (gain), saved model and results.json.
- **Storage:** local SQLite file `mlflow.db` (git-ignored). View with `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`.
- **Gotchas hit:** `mlflow-skinny` has no SQLite/UI (use full `mlflow`); metric names can't contain `@` (we log `recall_at_10`).
- Logging lives in the `train` process only, so it never shares a process with FAISS (OpenMP clash).

## Steps 10–11: API, tests, CI, Docker

### Concepts
- **Serving = same two stages, online.** Viewed product(s) become a one-person history; candidates + LightGBM score them (~0.35 s). Heavy objects (FAISS indexes, co-purchase matrix) are built once at startup.
- **New products** (no purchase history) fall back to embedding similarity only.
- **Bug worth telling in interviews:** server died silently on the first `/recommend`. Cause: OpenMP's thread setting is per thread and FastAPI runs handlers in worker threads. Fix: set FAISS to 1 thread inside every search; import LightGBM before FAISS.
- **Tests:** synthetic mini-catalog + fake scorer, so CI needs no dataset. FastAPI `TestClient` tests status codes (200/404/422).
- **CI:** GitHub Actions runs ruff + pytest on each push. **Docker:** deps layer cached; data mounted rather than baked in (image 830 MB, verified serving on Linux).
