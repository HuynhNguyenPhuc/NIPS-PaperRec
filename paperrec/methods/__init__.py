"""Recommenders: each takes an indexed Catalogue and a query title, and returns K catalogue positions."""

from __future__ import annotations

from paperrec.methods.bm25 import recommend_bm25
from paperrec.methods.louvain import recommend_graph
from paperrec.methods.random_baseline import recommend_random
from paperrec.methods.tfidf import recommend_tfidf

__all__ = ["recommend_bm25", "recommend_graph", "recommend_random", "recommend_tfidf"]
