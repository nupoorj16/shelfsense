FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Hugging Face Spaces runs containers as user 1000.
RUN useradd -m -u 1000 app
WORKDIR /app
RUN chown app:app /app
USER app
ENV PATH="/app/.venv/bin:$PATH" UV_LINK_MODE=copy

COPY --chown=app pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY --chown=app src ./src
COPY --chown=app data ./data
COPY --chown=app artifacts ./artifacts
RUN uv sync --frozen --no-dev

EXPOSE 7860
CMD ["uvicorn", "shelfsense.api:app", "--host", "0.0.0.0", "--port", "7860"]
