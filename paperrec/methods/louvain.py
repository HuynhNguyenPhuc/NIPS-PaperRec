"""Term-graph recommender: rank papers by shared title terms, optionally inside one Louvain community."""

from __future__ import annotations

from sklearn.metrics.pairwise import cosine_similarity

from paperrec.catalogue import Catalogue
from paperrec.config import K
from paperrec.dataset import title_terms
from paperrec.methods.tfidf import recommend_tfidf


def shared_term_scores(catalogue: Catalogue, terms: set[str]) -> dict[int, int]:
    """
    Count the kept title terms each catalogue paper shares with a query.

    Args:
        catalogue: Indexed catalogue whose kept-term index is used.
        terms: Title terms of the query, kept or not.

    Returns:
        scores: Catalogue position to number of shared kept terms.
    """
    # Only kept terms are looked up, because the others were dropped from the index.
    scores: dict[int, int] = {}
    for t in sorted(terms & catalogue.keep):
        for pos in catalogue.index.get(t, []):
            scores[pos] = scores.get(pos, 0) + 1

    return scores


def recommend_graph(catalogue: Catalogue, title: str, use_communities: bool = True) -> list[int]:
    """
    Rank papers by shared title terms, optionally inside one Louvain community.

    With use_communities, candidates are restricted to the community that
    shares the most terms with the query. Ties are broken by TF-IDF cosine.

    Args:
        catalogue: Indexed catalogue with its term graph and communities.
        title: Query title.
        use_communities: Whether to restrict candidates to one community.

    Returns:
        positions: Top K catalogue positions, or TF-IDF ranking when no term is shared.
    """
    scores = shared_term_scores(catalogue, title_terms(title))

    # Without a shared kept term the graph has no evidence, so TF-IDF decides instead.
    if not scores:
        return recommend_tfidf(catalogue, title)

    candidates = sorted(scores)

    # Without communities every paper that shares a term stays a candidate.
    if use_communities:
        # The community with the most shared-term weight is the one the query is routed to.
        community_score: dict[int, int] = {}
        for pos, s in scores.items():
            c = catalogue.community_of[pos]
            community_score[c] = community_score.get(c, 0) + s
        best = max(sorted(community_score), key=community_score.get)
        candidates = [p for p in candidates if catalogue.community_of[p] == best]

    # Remaining ties on shared terms are broken by TF-IDF cosine, then by position.
    q = catalogue.vectorizer.transform([title])
    sims = cosine_similarity(q, catalogue.matrix).ravel()
    candidates.sort(key=lambda p: (-scores[p], -sims[p], p))

    return candidates[:K]
