"""Dataset."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from paperrec.config import QUERY_FRACTION


def load(data_dir: Path) -> tuple[pd.DataFrame, dict[int, set[int]]]:
    """
    Load papers and the author set of each paper.

    Args:
        data_dir: Folder holding papers.csv and paper_authors.csv.

    Returns:
        papers: Papers with an ``authors`` column of author-id sets.
        authors_of: Paper id to its set of author ids.
    """
    # The author links live in their own table, so they are grouped per paper here.
    papers = pd.read_csv(data_dir / "papers.csv")
    links = pd.read_csv(data_dir / "paper_authors.csv")
    authors_of = links.groupby("paper_id")["author_id"].agg(set).to_dict()

    # Papers without any author link get an empty set, so they can be filtered later.
    papers["authors"] = papers["id"].map(lambda i: authors_of.get(i, set()))

    return papers, authors_of


def title_terms(title: str) -> set[str]:
    """
    Tokenise a title into lower-cased words without English stop words.

    Args:
        title: Paper title.

    Returns:
        terms: Words of length at least two that are not stop words.
    """
    # Lower-case alphanumeric runs are the tokens, so punctuation and hyphens split words.
    words = re.findall(r"[a-z0-9]+", str(title).lower())

    return {w for w in words if len(w) > 1 and w not in ENGLISH_STOP_WORDS}


def make_split(papers: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out a random share of author-bearing papers as queries.

    Args:
        papers: Papers with an ``authors`` column.
        rng: Random generator; its state is advanced by the draw.

    Returns:
        catalogue: Papers not used as queries, with a fresh index.
        queries: The held-out papers, with a fresh index.
    """
    # Papers with no author cannot have a relevant paper, so they never become queries.
    with_authors = papers[papers["authors"].map(len) > 0]
    query_ids = set(
        rng.choice(with_authors["id"].values, size=int(QUERY_FRACTION * len(with_authors)), replace=False)
    )

    # Everything that is not a query stays in the catalogue.
    catalogue = papers[~papers["id"].isin(query_ids)].reset_index(drop=True)
    queries = papers[papers["id"].isin(query_ids)].reset_index(drop=True)

    return catalogue, queries
