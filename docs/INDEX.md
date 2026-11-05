# Index

## 1. Concept to code

| Concept | Code |
|---|---|
| Benchmark run | `scripts/benchmark.py` (`main`, `write_markdown`) |
| Query/catalogue split | `paperrec/dataset.py:make_split`, `paperrec/evaluate.py:prepare_split` |
| Title tokens | `paperrec/dataset.py:title_terms` |
| Loading papers and authors | `paperrec/dataset.py:load` |
| Catalogue index, TF-IDF, BM25 | `paperrec/catalogue.py:Catalogue` |
| Term graph and Louvain communities (graph build) | `paperrec/graph.py` (`build_term_graph`, `louvain_communities`) |
| Random | `paperrec/methods/random_baseline.py:recommend_random` |
| BM25 title | `paperrec/methods/bm25.py:recommend_bm25` (constants `BM25_K1`, `BM25_B` in `paperrec/config.py`) |
| TF-IDF title | `paperrec/methods/tfidf.py:recommend_tfidf` |
| Term graph + Louvain | `paperrec/methods/louvain.py:recommend_graph` (default) |
| Term graph, no communities | `paperrec/methods/louvain.py:recommend_graph(use_communities=False)` |
| Per-query metrics | `paperrec/utils/metrics.py:score_list` (precision, recall, MRR, hit) |
| Per-split evaluation and seed summary | `paperrec/evaluate.py:evaluate_split`, `summarise` |
| Paired per-query statistics (win/tie/loss, bootstrap, sign-flip) | `paperrec/utils/stats.py:paired_summary`; run by `scripts/paired.py` |
| Constants (K, query fraction, MAX_DF, Louvain seed) | `paperrec/config.py` |
| Exploratory analysis | `scripts/eda.py` |
| Metric bar chart | `scripts/plot_results.py` |
| Radar chart | `scripts/plot_radar.py` |
| Method diagrams (random, term graph, Louvain) | `scripts/draw_methods.py` |
| Worked example of TF-IDF and BM25 (one query, two papers) | `scripts/draw_worked_example.py` |
| Example queries | `scripts/examples.py` |
| Demo | `app.py` (uses `paperrec.catalogue.Catalogue`) |

## 2. Data

- `data/papers.csv`: `id`, `year`, `title`, `event_type`, `authors` (a Python list literal).
- `data/authors.csv`: `id`, `name`.
- `data/paper_authors.csv`: `id`, `paper_id`, `author_id`. The benchmark uses this file for relevance.

## 3. Results files (`results/`)

| File | Contents |
|---|---|
| `results.md` | The benchmark table (mean ± std over seeds) |
| `results.json` | Per-seed scores, the summary, and term-graph statistics |
| `eda.json` | Numbers from `scripts/eda.py` |
| `examples.md` | Top five papers per example query, for all three recommenders (`scripts/examples.py`) |
| `figures/` | Figures written by `scripts/eda.py` (data), `scripts/plot_results.py` and `scripts/plot_radar.py` (benchmark) |
| `docs/figures/methods/` | Method diagrams (`random`, `term_graph`, `louvain`) written by `scripts/draw_methods.py`; TF-IDF and BM25 worked examples (`tfidf_worked`, `bm25_worked`) written by `scripts/draw_worked_example.py` |

## 4. Run

```bash
pip install -r requirements.txt
python -m scripts.benchmark --data data --out results --seeds 0 1 2   # about 4 minutes on a laptop CPU
python -m scripts.eda --data data --out results
python -m scripts.plot_results --results results
python -m scripts.plot_radar --results results
python -m scripts.examples --data data --out results
python -m scripts.draw_methods --out docs/figures/methods
python app.py                                                          # demo
```

Options of `scripts/benchmark.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `data` | Folder with `papers.csv` and `paper_authors.csv` |
| `--out` | `results` | Output folder |
| `--seeds` | `0 1 2` | Split and random-baseline seeds |
