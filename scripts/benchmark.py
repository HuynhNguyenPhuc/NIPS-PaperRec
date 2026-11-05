"""Benchmark paper recommenders on held-out NIPS papers.

Each split holds out 20% of the papers (with at least one author) as queries.
The remaining papers form the catalogue. A query is relevant to a catalogue
paper when they share at least one author. The graph uses title terms only,
so the relevance signal (authors) is not used to build any ranking.

Usage (from the repository root):
    python -m scripts.benchmark --data data --out results --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paperrec.config import K
from paperrec.dataset import load
from paperrec.evaluate import METRICS, evaluate_split, summarise


def write_markdown(summary: dict, runs: list[dict], path: Path) -> None:
    lines = [
        f"| Method | Precision@{K} | Recall@{K} | MRR | Hit@{K} |",
        "|---|---:|---:|---:|---:|",
    ]

    # One row per method; each cell is mean ± std over the seeds.
    for name, m in summary.items():
        cells = [f"{m[k][0]:.4f} ± {m[k][1]:.4f}" for k in METRICS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    g = runs[0]["graph"]

    # Seed and graph lines describe the splits behind the table above.
    lines += [
        "",
        f"Seeds: {', '.join(str(r['seed']) for r in runs)}. Queries per seed: "
        + ", ".join(str(r["graph"]["queries_evaluated"]) for r in runs) + ".",
        f"Term graph (seed {runs[0]['seed']}): {g['catalogue_papers']} catalogue papers, "
        f"{g['graph_edges']} edges, {g['communities']} Louvain communities, modularity {g['modularity']:.3f}.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    papers, _ = load(args.data)
    # Each seed is an independent split, and the summary averages over them.
    runs = [evaluate_split(papers, s) for s in args.seeds]
    summary = summarise(runs)

    (args.out / "results.json").write_text(
        json.dumps({"runs": runs, "summary": summary}, indent=2), encoding="utf-8"
    )
    write_markdown(summary, runs, args.out / "results.md")
    print((args.out / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
