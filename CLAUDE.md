# ShelfSense — project memory for Claude

## What this is
A beginner's resume project: a two-stage product recommender ("customers who viewed this also liked").
Stage 1: retrieval (sentence-transformers embeddings + FAISS). Stage 2: ranking (LightGBM learning-to-rank).
Target roles: SWE/SDE I and AI Engineer.

## How Claude must behave here (MENTOR MODE)
- The user writes the code. Do NOT write solutions unless they say "just write it".
- Help ladder, one level at a time: (1) simple concept + analogy, (2) what to build + inputs/outputs,
  (3) hint, (4) bigger hint / tiny example that is NOT the solution, (5) only on "just write it": code + line-by-line walkthrough.
- Claude MAY write boring setup (config files, scaffolding, Dockerfile, CI) but must explain each part.
- Review code like a kind senior engineer and ask guiding questions instead of fixing.
- Simple language, define every term, short explanations (user gets overwhelmed by walls of text).
- NO check-question quizzes (user said "forget the mini quizzes", 2026-10-01). Just give clear next actions.
- Point out interview talking points. "simpler" = re-explain more simply.
- Never move to the next step until the user says they're ready. Treat errors as debugging lessons.
- Commit at the end of each step with a clear message.
- Keep updated: CLAUDE.md (this file), LEARNING_LOG.md, RESUME_NOTES.md, README.md (incl. "How I used AI").

## Environment
macOS (Apple Silicon), 16 GB RAM, uv 0.12, git 2.50. Project Python 3.12 via uv (system python3 is 3.13 — don't use it).

## Tools & decisions
- Python 3.12 (not 3.13): ML libs (FAISS, LightGBM, torch) lag on newest Python.
- uv (not pip/conda/poetry): fast, lock file → reproducible installs (Docker later).
- src/ layout via `uv init --package`: code importable as `shelfsense`, tests import the installed package.
- ruff (not flake8+black+isort): one fast tool for linting + formatting. No custom ruff config yet (defaults).
- pytest: standard Python test runner. pre-commit: runs ruff + hygiene hooks before every commit.

## Plan
1. Project setup (uv, folders, git, ruff, pre-commit, pytest)  DONE
2. Find, download, explore dataset  <- CURRENT
3. Clean + time-based split
4. Baselines (popularity, simple similarity)
5. Embeddings + FAISS retrieval
6. LightGBM ranker
7. Evaluate vs. baselines (Recall@K, NDCG@K)
8. Cold start
9. MLflow tracking
10. FastAPI service + demo page
11. Tests, CI, Docker
12. Deploy + README

## Current progress
- 2026-09-30: Kickoff. Asked user about their background. Explained recommenders. Introduced step 1 (not started coding yet).
- User's background: skipped the questions; wants plain step-by-step instructions. Assume beginner, keep chunks small.
  Python is rusty ("been a while") — include quick syntax refreshers with analogous (not exact) examples.
- 2026-10-01: STEP 1 DONE. Commit 41f89b3 "set up project: uv, ruff, pytest, pre-commit".
  Smoke test passes, pre-commit installed and all hooks pass. Claude wrote LEARNING_LOG.md (step 1) — user to commit it.
  Lessons hit: running commands from tests/ instead of root (twice), pre-commit skips untracked files, --amend.
- Step 2 dataset research done by Claude:
  Amazon Reviews 2023 (McAuley Lab, UCSD; Hou et al. 2024, arXiv:2403.03952). No formal license stated;
  intended for research → cite it, don't commit/redistribute raw data (download via script), non-commercial portfolio use.
  UPDATE 2026-10-01: user is a programmer, watches lots of TV, loves shopping esp. makeup & skincare; asked Claude to pick.
  NEW CHOICE: Beauty_and_Personal_Care, narrowed to Makeup + Skin Care subcategories (via meta `categories` field).
  Why: user's domain knowledge = can sanity-check recs by eye; rich text (ingredients, skin type) for embeddings;
  constant new launches = real cold-start story. Cost: big files (reviews 2.96 GB gz, meta 0.71 GB gz; full 5-core
  730K users / 208K items / 6.6M ratings) → stream in chunks, keep few columns, filter subcategories, maybe a time window.
  Rejected: Movies_and_TV (mostly physical DVD sales, dated), Software (less relatable; the project already shows coding).
  Previous candidate notes (kept for reference): Video_Games.
  User then worried a male interviewer may not relate to beauty; Claude said domain barely matters but offered neutral
  options: Video_Games (fits easily), Cell_Phones_and_Accessories (5-core 381K users/112K items/2.8M ratings;
  2.5 GB + 0.86 GB gz). FINAL DECISION (2026-10-01): BEAUTY (Beauty_and_Personal_Care → makeup + skincare).
- Step 2 STARTED. 2a: user writes download script src/shelfsense/data/download.py using stdlib urllib
  (no new dependency), saving to data/raw/, skipping files that already exist.
  User asked for the full code → Claude gave it (tested in scratchpad, uses .part temp file + rename).
  Gotcha: system python3 (python.org 3.13) fails with SSL CERTIFICATE_VERIFY_FAILED; uv's 3.12 works → always `uv run`.
  User's pattern: prefers being given exact commands + code with explanations over hints.
- 2a DONE: data/raw/ has Beauty reviews (2.96 GB gz, verified with gzip -t) + meta (0.71 GB gz). Script not yet committed.
  Claude's peek: meta `categories` is a list; categories[1] is subcategory. In first 50K products: Hair Care 15.3K,
  Skin Care 9.0K, Makeup 6.4K, Nail 6.3K, Tools & Accessories 5.6K (incl. 'Makeup Brushes & Tools'), Fragrance 2.3K.
  → filter: categories[1] in {"Skin Care", "Makeup"} (decide later whether to add Makeup Brushes & Tools).
  Review fields: rating, title, text, images, asin, parent_asin, user_id, timestamp (ms epoch), helpful_vote, verified_purchase.
  Meta fields: main_category, title, average_rating, rating_number, features, description, price, images, videos, store,
  categories, details, parent_asin, bought_together. Use parent_asin as the product ID everywhere.
  First 100K review rows: 10K users, mean 9.8 reviews/user → file looks GROUPED BY USER, so head-of-file is a biased sample
  (teaching moment for user in exploration). Dates 2000–2023. Ratings skew to 5 (61%).
- Download script committed (05af48e). pandas (dep) + ipykernel (dev) added, not yet committed.
- 2b IN PROGRESS: notebooks/01_explore.ipynb created (empty). User got lost at "explore a sample" →
  re-explained as a click-by-click VS Code notebook walkthrough, one cell at a time.
  Notebook cells done: head(), basic stats (hit KeyError 'reviewerID' — old 2018 column name; confused .columns),
  reviews-per-user describe (max 2973 → power reviewers/outliers). User understands file is sorted by user (biased sample).
- 2c NEXT: user writes src/shelfsense/data/build_products.py (skeleton with TODOs given): chunked read of meta,
  keep categories[1] in {Skin Care, Makeup}, save data/interim/products.parquet. Needs `uv add pyarrow`.
  Claude's test: 1,028,914 products total → 320,042 kept (Skin Care 199,381; Makeup 120,661); ~40 s.
  No duplicate parent_asin, no missing titles. price: 198K None, 122K float, 15 strings '—' → to_parquet raises
  ArrowInvalid "Could not convert '—'". DELIBERATELY not in skeleton: let user hit it, fix via pd.to_numeric(errors="coerce").
- 2026-10-04: 2c WORKS. data/interim/products.parquet (25 MB, 320,042 rows; 198,105 missing prices). User added to_numeric.
  BUG FOUND (Claude's fault): .gitignore `data/` also ignored src/shelfsense/data/ → download.py & build_products.py
  were never committed (commits 32ac8dd/05af48e only had pyproject/uv.lock) and ruff/pre-commit skipped them.
  Fixed to `/data/` (root-anchored). Now ruff shows F841 unused `subcategory` on build_products.py line 27–28 → user to fix.
  Then commit: .gitignore, src/shelfsense/data/, notebooks/, pyproject.toml, uv.lock, CLAUDE.md.
  F841 fixed by user, ruff passes. All files STAGED (git add) but user had not yet run `git commit` as of 2026-10-04.
- 2d: Claude wrote skeleton src/shelfsense/data/build_reviews.py with 2 TODOs (isin product_ids; nunique users).
  Claude's full-run measurement (expected results for user's build_reviews.py): 23,911,390 total reviews →
  7,426,398 kept; 4,407,355 unique users; 319,981 products with ≥1 review. Dates 2000-04-01 → 2023-09-13.
  ~520 MB in memory, ~4m20s runtime. Reviews/user: median 1, mean 1.69, max 1,531 → extreme sparsity.
  Users with ≥5 reviews: 184,414; items with ≥5: 125,082. Duplicate (user, product) pairs: 100,171 → handle in step 3.
- 2d original plan: chunked read of reviews (4 cols: user_id, parent_asin, rating, timestamp), keep only reviews of kept products,
  save data/interim/reviews.parquet. Claude timing: ~8.7 s per 1M rows → full file ~3.5 min. Then full-data stats in notebook.
- Reference: Rejected All_Beauty (5-core only 253 users). Backup category: Musical_Instruments (0.6 GB).
  URL pattern: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/<Category>.jsonl.gz
               https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_<Category>.jsonl.gz
  Plan: user does own cleaning/5-core filter/time split (learning goal), not the pre-made benchmark CSVs.
