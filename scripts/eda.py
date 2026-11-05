"""Exploratory analysis of the NIPS papers, their titles and authorship links.

Writes the numbers to results/eda.json and the figures to results/figures/.
Usage (from the repository root):
    python -m scripts.eda --data data --out results
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from paperrec.config import MAX_DF  # noqa: E402
from paperrec.dataset import load, title_terms  # noqa: E402


def describe(papers, authors_of: dict[int, set[int]]) -> dict:
    """Counts for the inventory and structure layers."""
    n_authors = papers["authors"].map(len)
    titles = papers["title"].astype(str)

    return {
        "papers": int(len(papers)),
        "year_min": int(papers["year"].min()),
        "year_max": int(papers["year"].max()),
        "unique_authors": int(len({a for s in authors_of.values() for a in s})),
        "paper_author_links": int(sum(len(s) for s in authors_of.values())),
        "papers_without_authors": int((n_authors == 0).sum()),
        "authors_per_paper_mean": float(n_authors.mean()),
        "authors_per_paper_median": float(n_authors.median()),
        "authors_per_paper_max": int(n_authors.max()),
        "duplicated_titles": int(titles.duplicated(keep=False).sum()),
        "title_tokens_median": float(np.median([len(title_terms(t)) for t in titles])),
    }


def term_stats(titles: list[str]) -> dict:
    """Document frequency of title terms, against the MAX_DF cut-off used by the benchmark."""
    n = len(titles)
    df = Counter(t for title in titles for t in title_terms(title))
    dropped = sum(1 for c in df.values() if c > MAX_DF * n)

    return {
        "distinct_terms": len(df),
        "terms_dropped_by_max_df": dropped,
        "max_df_threshold_papers": MAX_DF * n,
        "top_terms": df.most_common(20),
    }


def related_counts(papers, authors_of: dict[int, set[int]]) -> np.ndarray:
    """
    For each paper, how many other papers share at least one author with it.

    This is the relevance proxy of the benchmark, computed on the full dataset
    rather than on a catalogue, so it is an upper bound on what a split sees.
    """
    # Author to papers index, so each paper's neighbours come from a lookup rather than a scan.
    papers_of_author: dict[int, set[int]] = {}
    for pid, auths in authors_of.items():
        for a in auths:
            papers_of_author.setdefault(a, set()).add(pid)

    # Distinct neighbour count per paper, in the order of the papers table.
    counts = []
    for pid in papers["id"]:
        related: set[int] = set()
        for a in authors_of.get(pid, set()):
            related |= papers_of_author[a]
        related.discard(pid)
        counts.append(len(related))

    return np.array(counts)


def plot_all(papers, related: np.ndarray, top_terms: list, figdir: Path) -> None:
    figdir.mkdir(parents=True, exist_ok=True)

    # Yearly counts show how the catalogue grows over time.
    per_year = papers.groupby("year").size()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(per_year.index, per_year.values, color="#4C72B0")
    ax.set(title="Papers per year", xlabel="Year", ylabel="Papers")
    fig.tight_layout()
    fig.savefig(figdir / "papers_per_year.png", dpi=150)
    plt.close(fig)

    # Log scale, because most papers have a few authors and a few have many.
    n_authors = papers["authors"].map(len)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(n_authors, bins=range(0, int(n_authors.max()) + 2), color="#55A868")
    ax.set(title="Authors per paper", xlabel="Authors", ylabel="Papers", yscale="log")
    fig.tight_layout()
    fig.savefig(figdir / "authors_per_paper.png", dpi=150)
    plt.close(fig)

    # Papers with no related paper are left out, since they would dominate the log scale.
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(related[related > 0], bins=50, color="#C44E52")
    ax.set(
        title="Papers sharing an author with a paper (full dataset)",
        xlabel="Related papers", ylabel="Papers", yscale="log",
    )
    fig.tight_layout()
    fig.savefig(figdir / "related_papers.png", dpi=150)
    plt.close(fig)

    # Only the 15 most frequent terms are drawn, so the bars stay readable.
    terms, counts = zip(*top_terms[:15])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(list(reversed(terms)), list(reversed(counts)), color="#8172B2")
    ax.set(title="Most frequent title terms", xlabel="Papers")
    fig.tight_layout()
    fig.savefig(figdir / "top_title_terms.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    papers, authors_of = load(args.data)
    related = related_counts(papers, authors_of)
    terms = term_stats(papers["title"].astype(str).tolist())
    stats = {
        "describe": describe(papers, authors_of),
        "title_terms": terms,
        "related_papers": {
            "median": float(np.median(related)),
            "mean": float(np.mean(related)),
            "share_with_zero": float(np.mean(related == 0)),
            "share_with_at_least_10": float(np.mean(related >= 10)),
        },
    }

    # The JSON is written before the figures, so the numbers survive a plotting failure.
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eda.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    plot_all(papers, related, terms["top_terms"], args.out / "figures")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
