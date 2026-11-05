"""BM25 title recommender: each title is a document of its distinct terms."""

from __future__ import annotations

import math

from paperrec.catalogue import Catalogue
from paperrec.config import BM25_B, BM25_K1, K
from paperrec.dataset import title_terms
from paperrec.methods.tfidf import recommend_tfidf


def recommend_bm25(catalogue: Catalogue, title: str, k1: float = BM25_K1, b: float = BM25_B) -> list[int]:
    """
    Rank the catalogue by BM25 score of the query's title terms.

    Each title is a document of its distinct terms (term frequency 1), with the
    Lucene IDF and the usual length normalisation. Papers with no shared term
    are not candidates; if none shares a term, the TF-IDF ranking is returned.

    Args:
        catalogue: Indexed catalogue whose postings and lengths are used.
        title: Query title.
        k1: Term-frequency saturation.
        b: Length normalisation.

    Returns:
        positions: Top K catalogue positions.
    """
    n = len(catalogue.ids)
    scores: dict[int, float] = {}

    # Each query term adds its IDF-weighted contribution to every paper that contains it.
    for t in sorted(title_terms(title)):
        postings = catalogue.postings.get(t)
        if not postings:
            continue

        # Lucene IDF, which stays positive even for a term that appears in most papers.
        df = len(postings)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))

        # Length normalisation scales this term's contribution for each paper.
        for pos in postings:
            norm = k1 * (1 - b + b * catalogue.doc_len[pos] / catalogue.avg_len)
            scores[pos] = scores.get(pos, 0.0) + idf * (k1 + 1) / (1 + norm)

    # With no shared term there is nothing to score, so the TF-IDF ranking is used instead.
    if not scores:
        return recommend_tfidf(catalogue, title)

    # Ties on score go to the lower catalogue position, so the ranking is deterministic.
    ranked = sorted(scores, key=lambda p: (-scores[p], p))

    return ranked[:K]
