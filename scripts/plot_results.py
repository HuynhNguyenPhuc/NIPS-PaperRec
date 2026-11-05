"""Bar chart of the benchmark metrics, mean with standard deviation over seeds.

Reads results/results.json and writes results/figures/benchmark_metrics.png.
Usage (from the repository root):
    python -m scripts.plot_results --results results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from paperrec.config import K  # noqa: E402

METRIC_LABELS = {"precision": f"Precision@{K}", "recall": f"Recall@{K}", "mrr": "MRR", "hit": f"Hit@{K}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    summary = json.loads((args.results / "results.json").read_text(encoding="utf-8"))["summary"]
    methods = list(summary)
    metrics = list(METRIC_LABELS)
    x = np.arange(len(metrics))
    width = 0.8 / len(methods)

    # Each method is one bar per metric, with its std as an error bar.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, name in enumerate(methods):
        means = [summary[name][m][0] for m in metrics]
        stds = [summary[name][m][1] for m in metrics]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=name)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics])
    ax.set(ylabel="Score (higher is better)", title="Benchmark: mean ± std over seeds")
    ax.legend(fontsize=8)
    fig.tight_layout()

    # The figure is saved next to the other benchmark outputs.
    figdir = args.results / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "benchmark_metrics.png", dpi=150)
    print(figdir / "benchmark_metrics.png")


if __name__ == "__main__":
    main()
