| Method | Precision@5 | Recall@5 | MRR | Hit@5 |
|---|---:|---:|---:|---:|
| Random | 0.0026 ± 0.0005 | 0.0007 ± 0.0002 | 0.0068 ± 0.0013 | 0.0129 ± 0.0025 |
| TF-IDF title | 0.0464 ± 0.0021 | 0.0344 ± 0.0025 | 0.1216 ± 0.0094 | 0.1922 ± 0.0102 |
| BM25 title | 0.0487 ± 0.0017 | 0.0354 ± 0.0017 | 0.1297 ± 0.0086 | 0.1978 ± 0.0091 |
| Term graph + Louvain | 0.0322 ± 0.0019 | 0.0205 ± 0.0021 | 0.0869 ± 0.0059 | 0.1365 ± 0.0079 |
| Term graph, no communities | 0.0475 ± 0.0027 | 0.0342 ± 0.0021 | 0.1252 ± 0.0099 | 0.1958 ± 0.0116 |

Seeds: 0, 1, 2. Queries per seed: 1315, 1286, 1268.
Term graph (seed 0): 5791 catalogue papers, 743316 edges, 32 Louvain communities, modularity 0.342.
