---
title: ShelfSense
emoji: 💄
colorFrom: pink
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# ShelfSense demo

Two-stage product recommender for makeup and skincare ("customers who viewed this also liked"):
sentence-embedding + FAISS retrieval, then a LightGBM LambdaRank re-ranker, trained on Amazon Reviews 2023
(McAuley Lab, UCSD; Hou et al. 2024, arXiv:2403.03952).

Search for a product, click it, and see recommendations. API docs at `/docs`.
This demo covers the ~47K products with purchase history. Source code and evaluation: see the project repository.
