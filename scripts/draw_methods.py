"""Schematic diagrams of the random, term-graph and Louvain rankings, one SVG per method.

The diagrams are drawn for explanation, not fitted to the data: the node layout
illustrates the rule, not the actual scores. The TF-IDF and BM25 figures are
worked examples on real data, written by scripts.draw_worked_example.
Writes docs/figures/methods/{random,term_graph,louvain}.svg.
Usage (from the repository root):
    python -m scripts.draw_methods --out docs/figures/methods
"""

from __future__ import annotations

import argparse
from pathlib import Path

W, H = 720, 200
INK, MUTE = "#1e293b", "#475569"
PAPER, PAPER_EDGE = "#e2e8f0", "#94a3b8"
ACCENT, ACCENT_SOFT = "#2563eb", "#bfdbfe"
FONT = "font-family=\"Helvetica, Arial, sans-serif\""


def svg(body: str) -> str:
    return (
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {W} {H}\" width=\"{W}\" height=\"{H}\">\n"
        f"<rect width=\"{W}\" height=\"{H}\" fill=\"#ffffff\"/>\n{body}</svg>\n"
    )


def text(x, y, s, size=12, color=INK, anchor="middle", weight="normal") -> str:
    return (f"<text x=\"{x}\" y=\"{y}\" {FONT} font-size=\"{size}\" fill=\"{color}\" "
            f"text-anchor=\"{anchor}\" font-weight=\"{weight}\">{s}</text>\n")


def box(x, y, w, h, fill, stroke, rx=6) -> str:
    return (f"<rect x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" rx=\"{rx}\" "
            f"fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"1.2\"/>\n")


def random_method() -> str:
    body = text(30, 26, "query title", 12, MUTE, "start")
    body += box(30, 34, 150, 26, "#f8fafc", PAPER_EDGE)
    body += text(105, 52, "any query", 11, MUTE)
    chosen = {1, 4, 6, 10, 13}
    for i in range(15):
        x = 30 + i * 45
        fill, stroke = (ACCENT_SOFT, ACCENT) if i in chosen else (PAPER, PAPER_EDGE)
        body += box(x, 96, 34, 52, fill, stroke)
    body += text(30, 88, "catalogue", 11, MUTE, "start")
    body += text(30, 176, "Five papers are drawn uniformly at random; the query is not used.", 12, MUTE, "start")

    return svg(body)


def term_graph_method() -> str:
    body = ""
    qx, qy = 110, 96
    body += f"<circle cx=\"{qx}\" cy=\"{qy}\" r=\"30\" fill=\"{INK}\"/>\n"
    body += text(qx, qy + 4, "query", 11, "#ffffff")
    body += text(qx, 150, "support · vector · machine", 10, MUTE)

    # Node positions are fixed by hand to illustrate the rule, not taken from data.
    nodes = [  # (x, y, shared kept terms with the query)
        (330, 62, 3), (330, 110, 2), (330, 158, 1),
        (500, 62, 2), (500, 158, 1),
        (640, 110, 0),
    ]
    for x, y, shared in nodes:
        if shared:
            body += (f"<line x1=\"{qx + 30}\" y1=\"{qy}\" x2=\"{x - 20}\" y2=\"{y}\" stroke=\"{ACCENT}\" "
                     f"stroke-width=\"{0.8 + shared * 1.1:.1f}\" opacity=\"0.7\"/>\n")
        fill = ACCENT if shared else PAPER
        stroke = ACCENT if shared else PAPER_EDGE
        body += f"<circle cx=\"{x}\" cy=\"{y}\" r=\"16\" fill=\"{fill}\" stroke=\"{stroke}\"/>\n"
        body += text(x, y + 4, str(shared), 11, "#ffffff" if shared else MUTE)

    # The caption explains the number printed on each circle.
    body += text(500, 194, "number on each paper = shared kept terms", 11, MUTE)
    body += text(30, 26, "Candidates are ranked by shared kept terms (ties by TF-IDF).", 12, MUTE, "start")

    return svg(body)


def louvain_method() -> str:
    body = ""
    body += ("<ellipse cx=\"250\" cy=\"100\" rx=\"160\" ry=\"80\" fill=\"#f8fafc\" stroke=\"#cbd5e1\" "
             "stroke-dasharray=\"5 4\"/>\n")
    body += ("<ellipse cx=\"540\" cy=\"100\" rx=\"140\" ry=\"80\" fill=\"#eff6ff\" stroke=\"#93c5fd\" "
             "stroke-dasharray=\"5 4\"/>\n")
    body += text(250, 178, "other communities: not ranked", 11, MUTE)
    body += text(540, 178, "chosen community: most shared-term score", 11, ACCENT)

    # The query sits left of the chosen community, and its links fan out into it.
    qx, qy = 80, 100
    body += f"<circle cx=\"{qx}\" cy=\"{qy}\" r=\"28\" fill=\"{INK}\"/>\n"
    body += text(qx, qy + 4, "query", 11, "#ffffff")

    # Papers outside the chosen community are drawn grey, with no links.
    for x, y in [(190, 60), (200, 120), (250, 90), (300, 140)]:
        body += f"<circle cx=\"{x}\" cy=\"{y}\" r=\"12\" fill=\"{PAPER}\" stroke=\"{PAPER_EDGE}\"/>\n"
    for x, y in [(470, 70), (520, 110), (580, 75), (500, 140), (600, 130)]:
        body += (f"<line x1=\"{qx + 28}\" y1=\"{qy}\" x2=\"{x - 12}\" y2=\"{y}\" stroke=\"{ACCENT}\" "
                 "stroke-width=\"1.4\" opacity=\"0.6\"/>\n")
        body += f"<circle cx=\"{x}\" cy=\"{y}\" r=\"13\" fill=\"{ACCENT}\"/>\n"
    body += text(30, 26, "Louvain restricts the candidates to one community first.", 12, MUTE, "start")

    return svg(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/figures/methods"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = {
        "random.svg": random_method(),
        "term_graph.svg": term_graph_method(),
        "louvain.svg": louvain_method(),
    }

    # Each diagram is written to its own file in the output folder.
    for name, content in files.items():
        (args.out / name).write_text(content, encoding="utf-8")
        print(args.out / name)


if __name__ == "__main__":
    main()
