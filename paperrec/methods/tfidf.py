"""TF-IDF title recommender: rank the catalogue by cosine similarity to the query title."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from paperrec.catalogue import Catalogue
from paperrec.config import K


def recommend_tfidf(catalogue: Catalogue, title: str) -> list[int]:
    """
    Rank the catalogue by TF-IDF cosine similarity to the query title.

    Args:
        catalogue: Indexed catalogue whose TF-IDF matrix is used.
        title: Query title.

    Returns:
        positions: Top K catalogue positions.
    """
    # The query is vectorised with the vocabulary fitted on the catalogue.
    q = catalogue.vectorizer.transform([title])
    sims = cosine_similarity(q, catalogue.matrix).ravel()

    # argsort on the negated similarities gives the most similar papers first.
    return list(np.argsort(-sims)[:K])
