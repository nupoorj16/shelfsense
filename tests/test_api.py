import pytest
from fastapi.testclient import TestClient

from shelfsense.api import app


@pytest.fixture()
def client(recommender):
    app.state.recommender = recommender  # skip loading the real model
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_demo_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "ShelfSense" in r.text


def test_search_then_recommend(client):
    hit = client.get("/search", params={"q": "serum"}).json()[0]
    r = client.get("/recommend", params={"asin": hit["asin"], "k": 4})
    assert r.status_code == 200
    assert len(r.json()["recommendations"]) == 4


def test_unknown_product_is_404(client):
    assert client.get("/recommend", params={"asin": "NOPE"}).status_code == 404


def test_bad_input_is_422(client):
    assert client.get("/search", params={"q": "a"}).status_code == 422
    assert client.get("/recommend").status_code == 422
    assert client.get("/recommend", params={"asin": "P01", "k": 0}).status_code == 422
