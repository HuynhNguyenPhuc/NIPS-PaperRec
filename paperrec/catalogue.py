"""Catalogue of Papers."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from paperrec.config import MAX_DF
from paperrec.dataset import title_terms
from paperrec.graph import build_term_graph, louvain_communities


class Catalogue:
    """Catalogue of Papers."""

    def __init__(self, titles: list[str], ids: list[int]) -> None:
        self.ids = ids
        self.terms = [title_terms(t) for t in titles]
        n = len(ids)

        # Document frequency of each term across the catalogue titles.
        df: dict[str, int] = {}
        for ts in self.terms:
            for t in ts:
                df[t] = df.get(t, 0) + 1

        # Terms in more than MAX_DF of the catalogue are too common to link papers.
        self.keep = {t for t, c in df.items() if c <= MAX_DF * n}

        # Sorted iteration keeps the index, the graph and the communities identical across processes.
        self.index: dict[str, list[int]] = {}
        self.postings: dict[str, list[int]] = {}
        for pos, ts in enumerate(self.terms):
            for t in sorted(ts):
                self.postings.setdefault(t, []).append(pos)
                if t in self.keep:
                    self.index.setdefault(t, []).append(pos)

        # TF-IDF is fitted on the catalogue titles only, so query terms outside it get no weight.
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(titles)

        # Length statistics of the distinct-term documents, used by BM25.
        self.doc_len = [len(ts) for ts in self.terms]
        self.avg_len = sum(self.doc_len) / n

        # The term graph and its communities back the term-graph recommenders.
        self.graph = build_term_graph(self.index, n)
        self.communities = louvain_communities(self.graph)
        # Maps each paper position to the index of its community.
        self.community_of: dict[int, int] = {}
        for c, members in enumerate(self.communities):
            for node in members:
                self.community_of[node] = c
