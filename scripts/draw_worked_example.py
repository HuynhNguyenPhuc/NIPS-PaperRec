"""Worked example of the TF-IDF and BM25 scores for one query and two catalogue papers.

Writes docs/figures/methods/tfidf_worked.svg and bm25_worked.svg. The query and the
two papers are fixed; every number drawn is computed from the Catalogue that the
recommenders use, and the script checks each hand calculation against them.
Usage (from the repository root):
    python -m scripts.draw_worked_example --data data --out docs/figures/methods
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import ConnectionPatch, Rectangle  # noqa: E402

from paperrec.catalogue import Catalogue  # noqa: E402
from paperrec.config import BM25_B, BM25_K1  # noqa: E402
from paperrec.dataset import load, title_terms  # noqa: E402
from paperrec.methods import recommend_bm25, recommend_tfidf  # noqa: E402

QUERY = "Support vector machines for classification"
TITLE_A = "Support Vector Regression Machines"
TITLE_B = "Adaptive Nearest Neighbor Classification Using Support Vector Machines"

INK, MUTE = "#1e293b", "#475569"
PAPER, PAPER_EDGE = "#e2e8f0", "#94a3b8"
ACCENT = "#2563eb"
SEGMENT_COLOURS = ["#2563eb", "#0d9488", "#d97706", "#7c3aed"]
CMAP = LinearSegmentedColormap.from_list("paper", ["#ffffff", ACCENT])
FOOTNOTE = f"A: {TITLE_A}\nB: {TITLE_B}"


def find_position(papers, title: str) -> int:
    """
    Catalogue position of the paper with the given title.

    Args:
        papers: Papers in the order the catalogue was built from.
        title: Exact title of one paper.

    Returns:
        position: Row of the paper, which is its catalogue position.
    """
    rows = papers.index[papers["title"] == title].tolist()
    assert len(rows) == 1, f"expected one paper titled {title!r}, found {len(rows)}"
    return rows[0]


def title_words(vec, title: str) -> list[str]:
    """
    Words of a title as the TF-IDF vectorizer reads them, in order, without repeats.

    Args:
        vec: Fitted TfidfVectorizer.
        title: Title or query text.

    Returns:
        words: Vocabulary words of the title.
    """
    return list(dict.fromkeys(t for t in vec.build_analyzer()(title) if t in vec.vocabulary_))


def tfidf_figure(cat: Catalogue, pa: int, pb: int, out: Path) -> None:
    """
    Draw the TF-IDF matrix: idf per term, normalised weights per title, and the products that give each cosine.

    Args:
        cat: Catalogue whose fitted vectorizer and matrix are used.
        pa: Catalogue position of paper A.
        pb: Catalogue position of paper B.
        out: Folder for the SVG.
    """
    vec = cat.vectorizer
    voc = vec.vocabulary_
    q = vec.transform([QUERY]).toarray().ravel()
    doc_a = cat.matrix[pa].toarray().ravel()
    doc_b = cat.matrix[pb].toarray().ravel()
    cos_a = float(q @ doc_a)
    cos_b = float(q @ doc_b)
    assert cos_a > cos_b, "the worked example must rank A above B"
    top = [int(p) for p in recommend_tfidf(cat, QUERY)]
    assert top[0] == pa and pb not in top, "TF-IDF ranking does not match the figure"

    words_q = title_words(vec, QUERY)
    words_a = title_words(vec, TITLE_A)
    words_b = title_words(vec, TITLE_B)
    idf = {t: float(vec.idf_[voc[t]]) for t in voc}
    length_q = math.sqrt(sum(idf[t] ** 2 for t in words_q))
    length_a = math.sqrt(sum(idf[t] ** 2 for t in words_a))
    length_b = math.sqrt(sum(idf[t] ** 2 for t in words_b))

    # Rows are the terms of the query and both titles, rarest first.
    terms = sorted(set(words_q) | set(words_a) | set(words_b), key=lambda t: -idf[t])
    n = len(terms)
    y = np.arange(n)
    idx = [voc[t] for t in terms]
    products_a = (q * doc_a)[idx]
    products_b = (q * doc_b)[idx]
    assert math.isclose(products_a.sum(), cos_a, rel_tol=1e-9)
    assert math.isclose(products_b.sum(), cos_b, rel_tol=1e-9)
    vmax = max(q.max(), doc_a.max(), doc_b.max())
    pmax = max(products_a.max(), products_b.max())

    fig, (ax_idf, ax) = plt.subplots(
        1, 2, figsize=(7.6, 0.42 * n + 3.0),
        gridspec_kw={"width_ratios": [1, 2.3], "wspace": 0.05}, sharey=True,
    )

    # Left: each term's idf; query terms in blue, the others grey.
    idf_vals = [idf[t] for t in terms]
    colours = [ACCENT if t in words_q else PAPER_EDGE for t in terms]
    ax_idf.barh(y, idf_vals, color=colours, height=0.6)
    for yi, v in zip(y, idf_vals):
        ax_idf.text(v + 0.08, yi, f"{v:.2f}", va="center", fontsize=8, color=MUTE)
    ax_idf.set_yticks(y, terms, fontsize=9)
    ax_idf.invert_yaxis()
    ax_idf.set_title("Term idf, rarest first", fontsize=10, color=INK)
    ax_idf.spines[["top", "right"]].set_visible(False)

    # Right: weight columns hold normalised tf-idf; product columns hold query weight times title weight.
    columns = [
        ("q", q[idx], 0.0, vmax), ("A", doc_a[idx], 1.0, vmax), ("B", doc_b[idx], 2.0, vmax),
        ("q·A", products_a, 3.6, pmax), ("q·B", products_b, 4.6, pmax),
    ]
    for header, values, x, scale in columns:
        for k, v in enumerate(values):
            ax.add_patch(Rectangle((x - 0.45, k - 0.45), 0.9, 0.9,
                                   facecolor=CMAP(v / scale), edgecolor="#e2e8f0", lw=0.8))
            ax.text(x, k, f"{v:.2f}" if v > 0 else "–", ha="center", va="center", fontsize=8,
                    color="#ffffff" if v / scale > 0.55 else INK)
        ax.text(x, -0.9, header, ha="center", va="center", fontsize=10, color=INK, weight="bold")

    # Under the weight columns: each title's length before normalisation. Under the product columns: the cosine.
    for x, length in [(0.0, length_q), (1.0, length_a), (2.0, length_b)]:
        ax.text(x, n + 0.2, f"length {length:.2f}", ha="center", va="center", fontsize=8, color=MUTE)
    ax.text(3.6, n + 0.2, f"cos {cos_a:.2f}", ha="center", va="center", fontsize=10, color=ACCENT, weight="bold")
    ax.text(4.6, n + 0.2, f"cos {cos_b:.2f}", ha="center", va="center", fontsize=10, color=ACCENT, weight="bold")
    ax.set_xlim(-0.6, 5.2)
    ax.set_ylim(n + 0.9, -1.4)
    ax.axis("off")

    fig.subplots_adjust(top=0.9, bottom=0.12, left=0.16, right=0.98)
    fig.text(0.02, 0.01, FOOTNOTE, ha="left", va="bottom", fontsize=8, color=MUTE)
    fig.savefig(out / "tfidf_worked.svg", bbox_inches="tight")
    plt.close(fig)


def bm25_figure(cat: Catalogue, pa: int, pb: int, out: Path) -> None:
    """
    Draw BM25: the idf carried over from TF-IDF, the length penalty, and the per-word sums.

    Args:
        cat: Catalogue whose vectorizer, postings, lengths and average length are used.
        pa: Catalogue position of paper A.
        pb: Catalogue position of paper B.
        out: Folder for the SVG.
    """
    vec = cat.vectorizer
    voc = vec.vocabulary_
    n = len(cat.ids)
    avg = cat.avg_len
    q_terms = sorted(title_terms(QUERY))
    df = {t: len(cat.postings.get(t, [])) for t in q_terms}
    idf_bm25 = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in q_terms}
    idf_tfidf = {t: float(vec.idf_[voc[t]]) for t in q_terms}

    def penalty(ratio, b: float):
        # Multiplies each word's idf; it is exactly 1 for a title of average length.
        return (BM25_K1 + 1) / (1 + BM25_K1 * (1 - b + b * ratio))

    def shared(pos: int) -> list[str]:
        # Query words the paper contains, largest BM25 idf first.
        return sorted((t for t in q_terms if pos in cat.postings.get(t, [])), key=lambda t: -idf_bm25[t])

    ratio_a = cat.doc_len[pa] / avg
    ratio_b = cat.doc_len[pb] / avg
    pen_a = float(penalty(ratio_a, BM25_B))
    pen_b = float(penalty(ratio_b, BM25_B))
    words_a, words_b = shared(pa), shared(pb)
    score_a = sum(idf_bm25[t] * pen_a for t in words_a)
    score_b = sum(idf_bm25[t] * pen_b for t in words_b)
    assert score_a > score_b, "the worked example must rank A above B"
    top = [int(p) for p in recommend_bm25(cat, QUERY)]
    assert top[0] == pa and pb in top, "BM25 ranking does not match the figure"

    order = sorted(q_terms, key=lambda t: -idf_tfidf[t])
    colour = {t: SEGMENT_COLOURS[i % len(SEGMENT_COLOURS)] for i, t in enumerate(q_terms)}
    # Two rows: the idf link and the length on top; the penalty and the score below.
    fig = plt.figure(figsize=(7.6, 6.6))

    # Top left: the same query words, with TF-IDF idf on the left and BM25 idf on the right, linked.
    axl = fig.add_axes([0.08, 0.56, 0.17, 0.34])
    axr = fig.add_axes([0.36, 0.56, 0.17, 0.34])
    y = np.arange(len(order))
    axl.barh(y, [idf_tfidf[t] for t in order], color=ACCENT, height=0.6)
    axr.barh(y, [idf_bm25[t] for t in order], color=colour[order[0]], height=0.6)
    for yi, t in zip(y, order):
        # Values sit inside the bar ends so the connector arrows start clear of them.
        axl.text(idf_tfidf[t] - 0.15, yi, f"{idf_tfidf[t]:.2f}", ha="right", va="center", fontsize=9, color="#ffffff")
        axr.text(idf_bm25[t] - 0.15, yi, f"{idf_bm25[t]:.2f}", ha="right", va="center", fontsize=9, color="#ffffff")
        con = ConnectionPatch(xyA=(idf_tfidf[t] + 0.2, yi), coordsA=axl.transData,
                              xyB=(0, yi), coordsB=axr.transData,
                              arrowstyle="-|>", color=MUTE, lw=1.0, mutation_scale=10)
        fig.add_artist(con)
    axl.set_yticks(y, order, fontsize=10)
    axl.set_title("idf from TF-IDF", fontsize=10, color=INK)
    axr.set_yticks([])
    axr.set_title("idf in BM25", fontsize=10, color=INK)
    for ax in (axl, axr):
        ax.set_xlim(0, 7.2)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.spines[["top", "right", "bottom"]].set_visible(False)

    # Top right: each title's length against the average title.
    ax1 = fig.add_axes([0.62, 0.56, 0.34, 0.34])
    ax1.set_xlim(0, 1.6)
    ax1.set_ylim(-0.6, 1.6)
    ax1.axis("off")
    ax1.plot([1, 1], [-0.45, 1.45], color=PAPER_EDGE, lw=1.2, ls=":")
    ax1.text(1, 1.5, "average", ha="center", va="bottom", fontsize=9, color=MUTE)
    for ratio, yy, label in [(ratio_a, 1, "A"), (ratio_b, 0, "B")]:
        ax1.add_patch(Rectangle((0, yy - 0.22), ratio, 0.44, facecolor=PAPER_EDGE, edgecolor="none"))
        ax1.text(-0.03, yy, label, ha="right", va="center", fontsize=12, color=INK, weight="bold")
        ax1.text(ratio + 0.03, yy, f"{ratio:.2f}", ha="left", va="center", fontsize=10, color=INK)
    ax1.set_title("title length / average", fontsize=10, color=INK)

    # Bottom left: the penalty for each length, read off the curve.
    ax2 = fig.add_axes([0.08, 0.1, 0.40, 0.34])
    ratios = np.linspace(0, 2.5, 200)
    ax2.plot(ratios, penalty(ratios, BM25_B), color=ACCENT, lw=2)
    ax2.plot(ratios, penalty(ratios, 0.0), color=MUTE, lw=1.2, ls="--")
    for ratio, pen, label in [(ratio_a, pen_a, "A"), (ratio_b, pen_b, "B")]:
        ax2.plot([ratio, ratio, 0], [0.5, pen, pen], color=INK, lw=0.9, ls=":")
        ax2.scatter([ratio], [pen], color=INK, s=28, zorder=3)
        ax2.text(0.04, pen + 0.03, f"{label} {pen:.2f}", ha="left", va="bottom", fontsize=10, color=INK, weight="bold")
    ax2.set_xlim(0, 2.5)
    ax2.set_ylim(0.5, 2.0)
    ax2.set_xlabel("length / average", fontsize=9)
    ax2.set_ylabel("penalty per word", fontsize=9)
    ax2.set_title("penalty", fontsize=10, color=INK)
    ax2.spines[["top", "right"]].set_visible(False)

    # Bottom right: each shared word's bar is its BM25 idf times the penalty; the dashed box is the sum without it.
    ax3 = fig.add_axes([0.62, 0.1, 0.34, 0.34])
    for x, words, pen, score, label in [(0, words_a, pen_a, score_a, "A"), (1, words_b, pen_b, score_b, "B")]:
        raw = sum(idf_bm25[t] for t in words)
        ax3.add_patch(Rectangle((x - 0.33, 0), 0.66, raw, facecolor="none", edgecolor=MUTE, lw=1.2, ls="--"))
        bottom = 0.0
        for t in words:
            v = idf_bm25[t] * pen
            ax3.add_patch(Rectangle((x - 0.4, bottom), 0.8, v, facecolor=colour[t], edgecolor="#ffffff", lw=1))
            ax3.text(x, bottom + v / 2, t, ha="center", va="center", fontsize=7, color="#ffffff")
            bottom += v
        # The label sits above the taller of the bar and its dashed box, so neither covers it.
        ax3.text(x, max(score, raw) + 0.6, f"{label}  {score:.2f}", ha="center", va="bottom", fontsize=11,
                 color=INK, weight="bold")
    ax3.set_xlim(-0.5, 1.5)
    ax3.set_ylim(0, 21)
    ax3.set_xticks([0, 1], ["A", "B"], fontsize=11)
    ax3.set_title("score = sum of idf × penalty", fontsize=10, color=INK)
    ax3.spines[["top", "right"]].set_visible(False)

    fig.text(0.02, 0.0, FOOTNOTE, ha="left", va="bottom", fontsize=8, color=MUTE)
    fig.savefig(out / "bm25_worked.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures/methods"))
    args = parser.parse_args()

    papers, _ = load(args.data)
    cat = Catalogue(papers["title"].tolist(), papers["id"].tolist())
    pa = find_position(papers, TITLE_A)
    pb = find_position(papers, TITLE_B)

    # The figures go to the folder the methods document reads them from.
    args.out.mkdir(parents=True, exist_ok=True)
    tfidf_figure(cat, pa, pb, args.out)
    bm25_figure(cat, pa, pb, args.out)
    print(args.out / "tfidf_worked.svg")
    print(args.out / "bm25_worked.svg")


if __name__ == "__main__":
    main()
