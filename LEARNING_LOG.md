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
