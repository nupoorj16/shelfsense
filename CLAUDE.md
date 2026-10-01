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
macOS (Apple Silicon), 16 GB RAM, uv 0.12, Python 3.13, git 2.50.

## Tools & decisions
- Python 3.12 (not 3.13): ML libs (FAISS, LightGBM, torch) lag on newest Python.
- uv (not pip/conda/poetry): fast, lock file → reproducible installs (Docker later).
- src/ layout via `uv init --package`: code importable as `shelfsense`, tests import the installed package.
- ruff (not flake8+black+isort): one fast tool for linting + formatting. No custom ruff config yet (defaults).
- pytest: standard Python test runner. pre-commit: runs ruff + hygiene hooks before every commit.

## Plan
1. Project setup (uv, folders, git, ruff, pre-commit, pytest)  <- CURRENT
2. Find, download, explore dataset
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
- Step 1: 1a–1c done (uv init --package --python 3.12, uv run works). Claude wrote .gitignore.
  1d done (ruff, pytest, pre-commit installed as dev deps). 1e: smoke test passes (user asked for the code, given).
  Ruff flagged I001 in test_smoke.py (trailing space + 1 blank line); user to fix with ruff --fix/format.
  Claude wrote .pre-commit-config.yaml (pre-commit-hooks v6.0.0 + ruff-pre-commit v0.16.9, validated).
  Remaining in step 1: user runs pre-commit install + run --all-files, then first commit.
