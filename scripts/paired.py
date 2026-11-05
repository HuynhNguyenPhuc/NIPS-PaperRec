"""Paired per-query comparison of BM25 and TF-IDF title rankings, and a sweep over the BM25 length parameter b.

Writes results/paired.md, results/paired.json and results/paired_per_query.json.
Usage (from the repository root):
    python -m scripts.paired --data data --out results --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paperrec.config import BM25_B, BM25_K1, K
from paperrec.dataset import load, title_terms
from paperrec.evaluate import prepare_split, relevant_positions
from paperrec.methods import recommend_bm25, recommend_tfidf
from paperrec.utils.metrics import score_list
from paperrec.utils.stats import paired_summary

# k1 is fixed at BM25_K1; the effective length weight is lambda = k1 * b / (k1 + 1).
B_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
METRICS = ("precision", "hit", "mrr")


def run_seed(papers, seed: int) -> dict:
    """
    Score every evaluated query with TF-IDF and with BM25 for each b in B_VALUES.

    The split is drawn exactly as in paperrec.evaluate.evaluate_split, so the
    queries and catalogue match the committed benchmark.

    Args:
        papers: Papers with an ``authors`` column of author-id sets.
        seed: Seed of the split.

    Returns:
        per_query: Method to metric to a list with one value per evaluated query.
        no_overlap: Evaluated queries whose title shares no term with any catalogue title.
    """
    catalogue, cat_authors, qry, _ = prepare_split(papers, seed)

    # The TF-IDF baseline is keyed "tfidf"; each BM25 setting is keyed by its b value.
    methods = {"tfidf": None, **{f"bm25_b{b:g}": b for b in B_VALUES}}
    per_query = {name: {m: [] for m in METRICS} for name in methods}
    no_overlap = 0

    # Each query is scored by TF-IDF and by every BM25 setting on the same relevant set.
    for _, row in qry.iterrows():
        relevant = relevant_positions(cat_authors, row["authors"])

        # Queries with no relevant catalogue paper are skipped, as in the benchmark.
        if not relevant:
            continue

        # The title is read once and shared by every method for this query.
        title = row["title"]

        # A query whose terms appear in no catalogue title falls back to TF-IDF under BM25.
        if not any(catalogue.postings.get(t) for t in title_terms(title)):
            no_overlap += 1

        # Each method ranks the same catalogue for this query.
        for name, b in methods.items():
            if b is None:
                top = recommend_tfidf(catalogue, title)
            else:
                top = recommend_bm25(catalogue, title, k1=BM25_K1, b=b)
            scores = score_list([int(p) for p in top], relevant)
            for m in METRICS:
                per_query[name][m].append(scores[m])

    return {"per_query": per_query, "no_overlap": no_overlap}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--boot", type=int, default=10000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    papers, _ = load(args.data)

    # A fixed generator for the resampling keeps the intervals and p-values reproducible.
    stats_rng = np.random.default_rng(12345)
    report = {"k": K, "k1": BM25_K1, "b_default": BM25_B, "boot": args.boot, "seeds": {}}
    per_query_out: dict[str, dict] = {}

    # Each split gets its own paired statistics; the splits are reported separately.
    for seed in args.seeds:
        run = run_seed(papers, seed)
        pq = run["per_query"]
        base = pq["tfidf"]

        # Each split reports its own paired statistics; the splits are not pooled.
        seed_out = {
            "n_queries": len(base["precision"]),
            "no_overlap_queries": run["no_overlap"],
            "means": {n: {m: float(np.mean(v)) for m, v in pq[n].items()} for n in pq},
            "bm25_vs_tfidf": {
                m: paired_summary(pq["bm25_b0.75"][m], base[m], args.boot, stats_rng) for m in METRICS
            },
            "sweep_vs_tfidf": {
                f"b={b:g}": {
                    "lambda": BM25_K1 * b / (BM25_K1 + 1),
                    "precision": paired_summary(pq[f"bm25_b{b:g}"]["precision"], base["precision"], args.boot, stats_rng),
                }
                for b in B_VALUES
            },
        }
        report["seeds"][str(seed)] = seed_out
        per_query_out[str(seed)] = pq

        # Progress goes to stdout, so a long run shows each split as it finishes.
        print(f"seed {seed}: n={seed_out['n_queries']} no_overlap={run['no_overlap']} "
              f"P@5 tfidf={seed_out['means']['tfidf']['precision']:.4f} "
              f"bm25={seed_out['means']['bm25_b0.75']['precision']:.4f}", flush=True)

    (args.out / "paired.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "paired_per_query.json").write_text(json.dumps(per_query_out), encoding="utf-8")
    (args.out / "paired.md").write_text(render(report), encoding="utf-8")
    print(f"wrote {args.out / 'paired.json'} and {args.out / 'paired.md'}")


def render(report: dict) -> str:
    """
    Markdown tables for the paired analysis.

    Args:
        report: Output of main, as a dict.

    Returns:
        text: Markdown document.
    """
    seeds = list(report["seeds"])
    lines = [
        "# Paired comparison: BM25 title vs TF-IDF title",
        "",
        f"Metric: Precision@{report['k']} per query. Paired differences: method minus TF-IDF, same queries. "
        f"BM25 uses k1 = {report['k1']} and b = {report['b_default']}. Bootstrap resamples: {report['boot']}. "
        "Interval: 95% percentile bootstrap over queries within one split. p-value: two-sided sign-flip permutation.",
        "",
        "## Per split: BM25 minus TF-IDF",
        "",
        "| Seed | Queries | Wins | Ties | Losses | Mean diff | 95% CI | p (sign-flip) | Mean gain on wins | Mean loss on losses | Queries with no shared term |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]

    # One row per split; the splits are reported separately, not pooled.
    for s in seeds:
        r = report["seeds"][s]
        d = r["bm25_vs_tfidf"]["precision"]
        lines.append(
            f"| {s} | {d['n']} | {d['wins']} | {d['ties']} | {d['losses']} | {d['mean']:+.4f} | "
            f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] | {d['p_signflip']:.4f} | "
            f"{d['mean_gain_on_wins']:+.3f} | {d['mean_loss_on_losses']:+.3f} | {r['no_overlap_queries']} |"
        )

    # Hit and MRR get their own intervals, next to the precision table.
    lines += ["", "## Hit@5 and MRR@5 differences per split", "",
              "| Seed | Hit@5 diff | 95% CI | MRR@5 diff | 95% CI |", "|---:|---:|---|---:|---|"]

    # One row per split, with the Hit and MRR differences and their intervals.
    for s in seeds:
        h = report["seeds"][s]["bm25_vs_tfidf"]["hit"]
        m = report["seeds"][s]["bm25_vs_tfidf"]["mrr"]
        lines.append(
            f"| {s} | {h['mean']:+.4f} | [{h['ci_low']:+.4f}, {h['ci_high']:+.4f}] | "
            f"{m['mean']:+.4f} | [{m['ci_low']:+.4f}, {m['ci_high']:+.4f}] |"
        )

    # The sweep is reported per b with its lambda, so the length effect can be read directly.
    lines += ["", "## Sensitivity to b (k1 fixed at 1.5): P@5 difference from TF-IDF per split", "",
              "| b | lambda | " + " | ".join(f"Seed {s} diff [95% CI]" for s in seeds) + " |",
              "|---:|---:|" + "---|" * len(seeds)]

    # Each row is one b value; each seed column shows its mean difference and interval.
    for key in report["seeds"][seeds[0]]["sweep_vs_tfidf"]:
        cells = []
        lam = report["seeds"][seeds[0]]["sweep_vs_tfidf"][key]["lambda"]
        for s in seeds:
            p = report["seeds"][s]["sweep_vs_tfidf"][key]["precision"]
            cells.append(f"{p['mean']:+.4f} [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}]")
        lines.append(f"| {key[2:]} | {lam:.2f} | " + " | ".join(cells) + " |")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
