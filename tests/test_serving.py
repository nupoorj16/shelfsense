import pytest


def test_recommend_returns_k_unique_products_not_already_viewed(recommender):
    out = recommender.recommend(["P01"], k=5)
    asins = [r["asin"] for r in out["recommendations"]]
    assert out["source"] == "two-stage"
    assert len(asins) == 5 and len(set(asins)) == 5
    assert "P01" not in asins


def test_recommendations_are_sorted_by_score(recommender):
    scores = [
        r["score"] for r in recommender.recommend(["P02", "P03"], 8)["recommendations"]
    ]
    assert scores == sorted(scores, reverse=True)


def test_new_product_falls_back_to_embeddings(recommender):
    out = recommender.recommend(["P27"], k=3)
    assert out["source"].startswith("embedding-only")
    assert len(out["recommendations"]) == 3


def test_unknown_product_raises(recommender):
    with pytest.raises(KeyError):
        recommender.recommend(["NOPE"])


def test_search_matches_all_words_and_handles_missing_price(recommender):
    hits = recommender.search("serum product")
    assert hits and all("serum" in h["title"] for h in hits)
    assert recommender.search("   ") == []
    assert any(h["price"] is None for h in recommender.search("lipstick", 25))
