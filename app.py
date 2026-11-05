"""Gradio demo: recommend NIPS papers for a title query."""

from __future__ import annotations

import ast
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from paperrec.catalogue import Catalogue
from paperrec.methods import (
    recommend_bm25,
    recommend_graph,
    recommend_random,
    recommend_tfidf,
)

# Paths are taken from this file, so the app runs from any working directory.
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
K = 5
DEFAULT_METHOD = "Term graph + Louvain"
NOTE = (
    "Enter a NIPS paper title (1987-2017) to get five similar papers from the catalogue. "
    "On the benchmark (see README), the Louvain community step lowers Precision@5 from "
    "0.048 to 0.032 on the share-an-author proxy, so BM25, TF-IDF or the term graph without "
    "communities score higher than the app's default. For demonstration only."
)

# The catalogue is built once at startup, so each query only ranks it.
papers = pd.read_csv(DATA / "papers.csv")
catalogue = Catalogue(papers["title"].tolist(), papers["id"].tolist())

# Each entry maps a query title to catalogue positions, as the benchmark's recommenders do.
METHODS = {
    "Random": lambda title: recommend_random(catalogue, np.random.default_rng(0)),
    "TF-IDF title": lambda title: recommend_tfidf(catalogue, title),
    "BM25 title": lambda title: recommend_bm25(catalogue, title),
    "Term graph + Louvain": lambda title: recommend_graph(catalogue, title),
    "Term graph, no communities": lambda title: recommend_graph(catalogue, title, use_communities=False),
}


def parse_authors(raw: str) -> str:
    """Turn the list literal stored in papers.csv into a comma-separated string."""
    try:
        return ", ".join(ast.literal_eval(raw))
    except (ValueError, SyntaxError, TypeError):
        return str(raw)


def recommend(title: str, method: str) -> pd.DataFrame:
    """
    Return up to five catalogue papers similar to the query title.

    Args:
        title: Query paper title.
        method: Key of METHODS.

    Returns:
        table: Rank, title, year and authors of each recommended paper.
    """
    if not title.strip():
        return pd.DataFrame(columns=["Rank", "Title", "Year", "Authors"])

    # The chosen method ranks the catalogue for this title.
    positions = METHODS[method](title)

    # Each position is looked up in the paper table to build one row per recommendation.
    rows = []
    for rank, pos in enumerate(positions, 1):
        row = papers.iloc[int(pos)]
        rows.append({
            "Rank": rank,
            "Title": row["title"],
            "Year": int(row["year"]),
            "Authors": parse_authors(row["authors"]),
        })

    return pd.DataFrame(rows)


with gr.Blocks(title="NIPS paper recommender") as demo:
    gr.Markdown(NOTE)

    # Inputs are the title and the method; the table shows the ranked result.
    query = gr.Textbox(label="Paper title", placeholder="e.g. Support Vector Machines for Classification")
    method = gr.Dropdown(choices=list(METHODS), value=DEFAULT_METHOD, label="Method")
    run = gr.Button("Recommend")
    table = gr.Dataframe(label=f"Top {K} recommendations")

    # Pressing Enter in the title box and clicking the button both run the same query.
    run.click(recommend, inputs=[query, method], outputs=table)
    query.submit(recommend, inputs=[query, method], outputs=table)


if __name__ == "__main__":
    demo.launch()
