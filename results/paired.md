# Paired comparison: BM25 title vs TF-IDF title

Metric: Precision@5 per query. Paired differences: method minus TF-IDF, same queries. BM25 uses k1 = 1.5 and b = 0.75. Bootstrap resamples: 10000. Interval: 95% percentile bootstrap over queries within one split. p-value: two-sided sign-flip permutation.

## Per split: BM25 minus TF-IDF

| Seed | Queries | Wins | Ties | Losses | Mean diff | 95% CI | p (sign-flip) | Mean gain on wins | Mean loss on losses | Queries with no shared term |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 1315 | 50 | 1231 | 34 | +0.0024 | [-0.0005, +0.0053] | 0.1147 | +0.204 | -0.206 | 1 |
| 1 | 1286 | 43 | 1216 | 27 | +0.0028 | [+0.0000, +0.0056] | 0.0593 | +0.214 | -0.207 | 1 |
| 2 | 1268 | 44 | 1193 | 31 | +0.0017 | [-0.0011, +0.0046] | 0.2724 | +0.200 | -0.213 | 2 |

## Hit@5 and MRR@5 differences per split

| Seed | Hit@5 diff | 95% CI | MRR@5 diff | 95% CI |
|---:|---:|---|---:|---|
| 0 | +0.0084 | [-0.0030, +0.0198] | +0.0074 | [+0.0007, +0.0141] |
| 1 | +0.0047 | [-0.0062, +0.0148] | +0.0102 | [+0.0034, +0.0173] |
| 2 | +0.0039 | [-0.0079, +0.0158] | +0.0068 | [+0.0001, +0.0136] |

## Sensitivity to b (k1 fixed at 1.5): P@5 difference from TF-IDF per split

| b | lambda | Seed 0 diff [95% CI] | Seed 1 diff [95% CI] | Seed 2 diff [95% CI] |
|---:|---:|---|---|---|
| 0 | 0.00 | -0.0005 [-0.0043, +0.0033] | -0.0008 [-0.0042, +0.0026] | +0.0011 [-0.0027, +0.0049] |
| 0.25 | 0.15 | +0.0029 [-0.0002, +0.0059] | +0.0014 [-0.0016, +0.0044] | +0.0022 [-0.0008, +0.0050] |
| 0.5 | 0.30 | +0.0029 [+0.0000, +0.0058] | +0.0017 [-0.0011, +0.0045] | +0.0021 [-0.0008, +0.0050] |
| 0.75 | 0.45 | +0.0024 [-0.0005, +0.0053] | +0.0028 [+0.0002, +0.0056] | +0.0017 [-0.0011, +0.0046] |
| 1 | 0.60 | +0.0026 [-0.0002, +0.0053] | +0.0026 [-0.0000, +0.0054] | +0.0003 [-0.0025, +0.0032] |
