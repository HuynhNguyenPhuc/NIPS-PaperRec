"""
Plot the benchmark summary as one radar chart with every method on the same axes.

Each axis is min-max scaled across the methods: the worst method sits on the inner ring
and the best on the outer ring, so the chart shows relative position only. The raw
range of every axis is written in the footnote.
Reads results/results.json and writes results/figures/benchmark_radar.png.
Usage (from the repository root):
    python -m scripts.plot_radar --results results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from paperrec.config import K  # noqa: E402

AXES = [
    ("precision", f"Precision@{K}"),
    ("recall", f"Recall@{K}"),
    ("mrr", "MRR"),
    ("hit", f"Hit@{K}"),
]
INK, MUTE = "#1e293b", "#64748b"
GRID = "#e2e8f0"
FLOOR = 0.15

# Baselines are drawn as grey dashed lines; the two term-graph methods are blue, the app's design solid.
STYLE = {
    "Random": {"color": "#94a3b8", "ls": ":", "lw": 1.6, "fill": False},
    "TF-IDF title": {"color": "#475569", "ls": "--", "lw": 1.8, "fill": False},
    "BM25 title": {"color": "#1e293b", "ls": "-.", "lw": 1.8, "fill": False},
    "Term graph, no communities": {"color": "#60a5fa", "ls": "-", "lw": 2.0, "fill": True},
    "Term graph + Louvain": {"color": "#2563eb", "ls": "-", "lw": 2.6, "fill": True},
}
ORDER = ["Random", "TF-IDF title", "BM25 title", "Term graph, no communities", "Term graph + Louvain"]


def plot(summary: dict, out: Path) -> None:
    """
    Draw every method on one radar, with the baselines as lines and the term-graph methods filled.

    Args:
        summary: Method to metric to (mean, std), as written by scripts.benchmark.
        out: Path of the image to write.
    """
    means = pd.DataFrame({m: {k: summary[m][k][0] for k, _ in AXES} for m in ORDER}).T
    scaled = FLOOR + (1 - FLOOR) * (means - means.min()) / (means.max() - means.min())

    # Axes are spaced evenly around the circle; the first point is repeated to close each shape.
    n = len(AXES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed = np.append(angles, angles[0])

    fig = plt.figure(figsize=(8, 6.4))
    ax = fig.add_subplot(111, polar=True)
    for method in ORDER:
        style = STYLE[method]
        ring = np.append(scaled.loc[method].values, scaled.loc[method].values[0])
        ax.plot(closed, ring, color=style["color"], lw=style["lw"], ls=style["ls"], label=method, zorder=3)
        if style["fill"]:
            ax.fill(closed, ring, color=style["color"], alpha=0.12, zorder=2)

    ax.set_ylim(0, 1)
    ax.set_yticks([FLOOR, 0.5, 1.0])
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels([label for _, label in AXES], fontsize=11, color=INK)
    ax.tick_params(axis="x", pad=28)
    ax.grid(color=GRID, lw=0.8)
    ax.spines["polar"].set_color(GRID)

    # The legend sits to the right, in the same order as the methods are listed.
    legend = ax.legend(loc="center left", bbox_to_anchor=(1.2, 0.5), frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color(INK)

    # Raw values are in results.md; the footnote only says how to read the rings.
    fig.text(0.02, 0.02, "Outer ring = best method on that axis, inner ring = worst. Values in results.md.",
             ha="left", va="bottom", fontsize=8, color=MUTE)
    fig.subplots_adjust(left=0.1, right=0.7, top=0.92, bottom=0.12)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    summary = json.loads((args.results / "results.json").read_text(encoding="utf-8"))["summary"]

    # The radar goes to the same figures folder as the bar chart.
    figdir = args.results / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    out = figdir / "benchmark_radar.png"
    plot(summary, out)
    print(out)


if __name__ == "__main__":
    main()
