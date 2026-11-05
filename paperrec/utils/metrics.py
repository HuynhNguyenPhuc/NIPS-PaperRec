"""Ranking metrics for one query: precision, recall, reciprocal rank and hit at K."""

from __future__ import annotations

from paperrec.config import K


def score_list(top: list[int], relevant: set[int]) -> dict[str, float]:
    """
    Score one ranked list of catalogue positions against its relevant set.

    Args:
        top: Recommended catalogue positions, best first, length K.
        relevant: Catalogue positions that count as relevant to the query.

    Returns:
        scores: precision@K, recall@K, reciprocal rank of the first hit (0 if none), hit@K.
    """
    # hits[r] is True when the paper at rank r + 1 is relevant.
    hits = [p in relevant for p in top]

    # The reciprocal rank uses only the first relevant paper in the list.
    first = next((r for r, h in enumerate(hits, 1) if h), None)

    return {
        "precision": sum(hits) / K,
        "recall": sum(hits) / len(relevant),
        "mrr": 1 / first if first else 0.0,
        "hit": float(any(hits)),
    }
