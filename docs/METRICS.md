# Metrics

## 1. Notation

A query is a held-out paper with title $q$ and authors $A_q$. The catalogue $\mathcal{C}$ is the set of the other papers. A recommender returns an ordered list $L = (p_1, \dots, p_K)$ with $K = 5$. The set of relevant catalogue papers is $R_q = \{p \in \mathcal{C} : A_p \cap A_q \neq \emptyset\}$. Let $\mathbf{1}[p_i \in R_q]$ be the hit indicator at rank $i$.

## 2. Per-query metrics

| Metric | Definition | Range |
|---|---|---|
| Precision@K | $\frac{1}{K}\sum_{i=1}^{K} \mathbf{1}[p_i \in R_q]$ | 0 to 1 |
| Recall@K | $\frac{1}{\lvert R_q \rvert}\sum_{i=1}^{K} \mathbf{1}[p_i \in R_q]$ | 0 to 1 |
| MRR | $1 / r$, where $r$ is the rank of the first hit; 0 if no hit in the top K | 0 to 1 |
| Hit@K | 1 if any of the top K is relevant, else 0 | 0 or 1 |

Queries with $R_q = \emptyset$ are skipped. The number of skipped queries is not a score; it is reported as "queries evaluated" in the results.

## 3. Aggregation

- **Queries.** Each metric is averaged over the evaluated queries of one split.
- **Seeds.** Each seed gives one split, one random baseline draw and one Louvain run (seed 0). The table reports the mean and the population standard deviation (`numpy.std`) over seeds.

## 4. Interpretation

- Precision and recall are close to each other here because $\lvert R_q \rvert$ is often large (median about 10 related papers on the full dataset, see `results/eda.json`), so recall@5 stays small even for good rankings.
- The standard deviation over seeds reflects the choice of queries and the random baseline. It does not give a significance test. Differences smaller than about one standard deviation should not be read as real.
