"""Benchmark settings shared by the library, the benchmark and the demo."""

# Split and ranking.
K = 5  # Papers recommended per query.
QUERY_FRACTION = 0.2  # Share of author-bearing papers held out as queries.

# Term graph. A term in more than this share of the catalogue links too many papers to be informative.
MAX_DF = 0.05

# Community detection on the term graph. Fixed so the partition is reproducible.
LOUVAIN_SEED = 0

# BM25 parameters. Lucene's defaults are k1 = 1.2 and b = 0.75; this benchmark uses k1 = 1.5.
BM25_K1 = 1.5  # Term-frequency saturation.
BM25_B = 0.75  # Length normalisation.
