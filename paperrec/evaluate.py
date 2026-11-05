"""Evaluation."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from paperrec.catalogue import Catalogue
from paperrec.dataset import make_split
from paperrec.methods import (
    recommend_bm25,
    recommend_graph,
    recommend_random,
    recommend_tfidf,
)
from paperrec.utils.metrics import score_list

METRICS = ("precision", "recall", "mrr", "hit")


def prepare_split(papers: pd.DataFrame, seed: int) -> tuple[Catalogue, list[set[int]], pd.DataFrame, np.random.Generator]:
    """
    Draw the split for a seed and index its catalogue.

    The same generator is returned so that the random baseline continues the
    stream that made the split, as the committed benchmark does.

    Args:
        papers: Papers with an ``authors`` column of author-id sets.
        seed: Seed of the split and of the random baseline.

    Returns:
        catalogue: Indexed catalogue of the non-query papers.
        cat_authors: Author set of each catalogue paper, by position.
        queries: The query papers, with a fresh index.
        rng: Generator positioned after the split draw.
    """
    # The seed fixes the split and every draw that follows it from the same stream.
    rng = np.random.default_rng(seed)
    cat, qry = make_split(papers, rng)

    # The catalogue is built from the non-query papers only.
    catalogue = Catalogue(cat["title"].tolist(), cat["id"].tolist())

    return catalogue, cat["authors"].tolist(), qry, rng


def relevant_positions(cat_authors: list[set[int]], authors: set[int]) -> set[int]:
    """
    Catalogue positions of the papers that share an author with a query.

    Args:
        cat_authors: Author set of each catalogue paper, by position.
        authors: Author set of the query paper.

    Returns:
        positions: Catalogue positions that count as relevant.
    """
    return {p for p, au in enumerate(cat_authors) if au & authors}


def evaluate_split(papers: pd.DataFrame, seed: int) -> dict:
    """
    Run every method on one split and average the metrics over the queries.

    A query is scored only if at least one catalogue paper shares an author with it.

    Args:
        papers: Papers with an ``authors`` column of author-id sets.
        seed: Seed of the split and of the random baseline.

    Returns:
        run: Seed, per-method mean metrics, and term-graph statistics.
    """
    catalogue, cat_authors, qry, rng = prepare_split(papers, seed)

    # Each method maps a query title to a list of catalogue positions.
    methods = {
        "Random": lambda title, i: recommend_random(catalogue, rng),
        "TF-IDF title": lambda title, i: recommend_tfidf(catalogue, title),
        "BM25 title": lambda title, i: recommend_bm25(catalogue, title),
        "Term graph + Louvain": lambda title, i: recommend_graph(catalogue, title),
        "Term graph, no communities": lambda title, i: recommend_graph(catalogue, title, use_communities=False),
    }
    totals = {m: {k: [] for k in METRICS} for m in methods}
    evaluated = 0

    # Each query is scored by every method, so all means share one query set.
    for i, row in qry.iterrows():
        relevant = relevant_positions(cat_authors, row["authors"])

        # Queries with no relevant catalogue paper have nothing to score, so they are skipped.
        if not relevant:
            continue
        evaluated += 1

        # Every method is scored against the same relevant set.
        for name, rec in methods.items():
            top = [int(p) for p in rec(row["title"], i)]
            scores = score_list(top, relevant)
            for k in METRICS:
                totals[name][k].append(scores[k])

    # Means are over the evaluated queries only.
    results = {
        name: {k: float(np.mean(v)) for k, v in vals.items()} for name, vals in totals.items()
    }

    # Graph statistics describe the catalogue this split produced, for the results file.
    graph_stats = {
        "catalogue_papers": len(cat_authors),
        "queries_evaluated": evaluated,
        "graph_edges": catalogue.graph.number_of_edges(),
        "communities": len(catalogue.communities),
        "modularity": float(nx.community.modularity(catalogue.graph, catalogue.communities, weight="weight")),
    }

    return {"seed": seed, "methods": results, "graph": graph_stats}


def summarise(runs: list[dict]) -> dict:
    """
    Mean and population standard deviation of each metric over seeds.

    Args:
        runs: Output of evaluate_split, one per seed.

    Returns:
        summary: Method to metric to (mean, std).
    """
    out = {}

    # Each method and metric is summarised over the seeds independently.
    for name in runs[0]["methods"]:
        out[name] = {}
        for metric in METRICS:
            vals = [r["methods"][name][metric] for r in runs]

            # np.std uses ddof=0, the population form, which the committed tables use.
            out[name][metric] = (float(np.mean(vals)), float(np.std(vals)))

    return out
