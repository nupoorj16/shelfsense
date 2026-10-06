"""HTTP service. Run: `uv run uvicorn shelfsense.api:app` then open http://127.0.0.1:8000"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from shelfsense.serving import Recommender

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup (takes a while: builds FAISS indexes and the co-purchase matrix).
    if getattr(app.state, "recommender", None) is None:
        app.state.recommender = Recommender.load()
    yield


app = FastAPI(title="ShelfSense", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def demo_page() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/search")
def search(
    q: Annotated[str, Query(min_length=2)],
    limit: Annotated[int, Query(ge=1, le=25)] = 10,
) -> list:
    """Find products by title words, most reviewed first."""
    return app.state.recommender.search(q, limit)


@app.get("/recommend")
def recommend(
    asin: Annotated[list[str], Query(min_length=1, max_length=20)],
    k: Annotated[int, Query(ge=1, le=50)] = 10,
) -> dict:
    """Products to show someone who viewed `asin` (repeat the parameter for several, newest last)."""
    try:
        return app.state.recommender.recommend(asin, k)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown product") from None
