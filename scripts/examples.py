"""Top recommendations for example queries, with every method side by side.

Writes results/examples.md. In each title, the words shared with the query are bold.
Usage (from the repository root):
    python -m scripts.examples --data data --out results --top 3
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from paperrec.catalogue import Catalogue
from paperrec.config import K
from paperrec.dataset import load, title_terms
from paperrec.methods import recommend_bm25, recommend_graph, recommend_tfidf

QUERIES = [
    "Support vector machines for classification",
    "Gaussian processes for regression",
    "Reinforcement learning with function approximation",
    "Convolutional networks for image recognition",
]

# Words are matched the same way title_terms tokenises titles.
WORD = re.compile(r"[A-Za-z0-9]+")


def highlight(title: str, query_terms: set[str]) -> str:
    """
    Wrap each title word that is also a query term in bold.

    Args:
        title: Catalogue paper title, with its original casing.
        query_terms: Title terms of the query, as returned by title_terms.

    Returns:
        marked: The title with the shared words in bold.
    """
    # Words and the text between them are kept as separate pieces, so runs can be merged.
    pieces = re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9]+", title)

    def is_shared(piece: str) -> bool:
        # title_terms lower-cases its tokens, so the comparison is done in lower case too.
        return bool(WORD.fullmatch(piece)) and piece.lower() in query_terms

    # A single space between two shared words stays inside the same bold run.
    out, open_run = [], False
    for i, piece in enumerate(pieces):
        if WORD.fullmatch(piece):
            bold = is_shared(piece)
        else:
            bold = (
                piece == " " and 0 < i < len(pieces) - 1
                and is_shared(pieces[i - 1]) and is_shared(pieces[i + 1])
            )
        if bold and not open_run:
            out.append("**")
            open_run = True
        elif not bold and open_run:
            out.append("**")
            open_run = False
        out.append(piece)
    if open_run:
        out.append("**")
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--top", type=int, default=K)
    args = parser.parse_args()

    papers, _ = load(args.data)
    catalogue = Catalogue(papers["title"].tolist(), papers["id"].tolist())
    methods = {
        "TF-IDF title": lambda t: recommend_tfidf(catalogue, t),
        "BM25 title": lambda t: recommend_bm25(catalogue, t),
        "Term graph + Louvain": lambda t: recommend_graph(catalogue, t),
        "Term graph, no communities": lambda t: recommend_graph(catalogue, t, use_communities=False),
    }

    # Markdown is built in memory and written once at the end.
    lines = [
        "# Example queries",
        "",
        "Bold words are title terms shared with the query, after stop words and one-letter tokens are removed.",
        "",
    ]
    for query in QUERIES:
        query_terms = title_terms(query)

        # Every method ranks the same catalogue, so the columns can be read across.
        ranked = {name: [int(p) for p in rec(query)[: args.top]] for name, rec in methods.items()}
        lines += [
            f"## {query}",
            "",
            "| Rank | " + " | ".join(methods) + " |",
            "|---:|" + "---|" * len(methods),
        ]
        for rank in range(args.top):
            cells = []
            for name in methods:
                row = papers.iloc[ranked[name][rank]]
                cells.append(f"{highlight(row['title'], query_terms)} ({int(row['year'])})")
            lines.append(f"| {rank + 1} | " + " | ".join(cells) + " |")
        lines.append("")

    # The output folder may not exist yet on a fresh clone.
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "examples.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.out / "examples.md")


if __name__ == "__main__":
    main()
