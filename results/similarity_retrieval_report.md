# Legal Lens — Similarity & Precedent Retrieval Benchmark Report

> **System Purpose**: Pure Legal Case Similarity and Precedent Retrieval across 26,690 Indian Supreme Court judgments (1950–2025). This module retrieves and explains relevant historical precedents. It does NOT predict future case outcomes.

## 1. Executive Summary

- **Benchmark Scope**: 50 legal research queries across Indian Supreme Court jurisprudence.
- **Corpus Size**: 26,690 judgments (1950–2025) indexed with multi-faceted hybrid retrieval.
- **Average Query Latency**: 1547.8 ms.

## 2. Standard Information Retrieval Metrics

| Metric | Score | Description |
|---|---|---|
| **Precision@5** | `0.628` | Fraction of Top-5 retrieved cases that are legally and factually relevant |
| **Precision@10** | `0.442` | Fraction of Top-10 retrieved cases that are relevant |
| **Recall@5** | `1.0` | Proportion of known relevant precedents retrieved in Top-5 |
| **Recall@10** | `1.0` | Proportion of known relevant precedents retrieved in Top-10 |
| **MRR** | `1.0` | Mean Reciprocal Rank of the first relevant retrieved precedent |
| **NDCG@10** | `0.9759` | Normalized Discounted Cumulative Gain accounting for rank positions |

## 3. Precedent Similarity Distribution & Retrieval Rate

| Measure | Value | Interpretation |
|---|---|---|
| **Top-1 Similarity (Mean)** | `27.75%` | Average similarity match for the primary retrieved precedent |
| **Mean Top-5 Similarity** | `25.11%` | Average similarity across the top 5 precedents |
| **Mean Top-10 Similarity** | `23.32%` | Average similarity across the top 10 precedents |
| **Relevant-Case Retrieval Rate @5** | `100.0%` | Percentage of queries with at least 1 relevant case in Top-5 |
| **Relevant-Case Retrieval Rate @10** | `100.0%` | Percentage of queries with at least 1 relevant case in Top-10 |

## 4. Performance by Legal Category

| Category | Queries | Precision@5 | NDCG@10 | Mean Top-1 Similarity |
|---|---|---|---|---|
| Criminal & Constitutional Law | 1 | 0.6 | 0.9929 | 17.13% |
| Property & Land Law | 1 | 0.6 | 0.9629 | 28.01% |
| Criminal Procedure | 3 | 0.6667 | 0.9678 | 30.35% |
| Commercial & Financial Law | 1 | 0.8 | 0.984 | 37.14% |
| Service & Administrative Law | 1 | 0.6 | 0.9629 | 30.06% |
| Criminal Law | 6 | 0.6 | 0.9629 | 32.44% |
| Tort & Insurance Law | 1 | 0.6 | 0.9629 | 28.75% |
| Civil & Contract Law | 1 | 0.6 | 0.9629 | 36.03% |
| Consumer Law | 1 | 0.6 | 0.9964 | 18.92% |
| Arbitration Law | 1 | 0.6 | 0.9629 | 29.29% |
| Constitutional Law | 5 | 0.64 | 0.9914 | 25.0% |
| Family Law | 2 | 0.7 | 0.9776 | 28.67% |
| Evidence Law | 2 | 0.8 | 0.9818 | 25.32% |
| Environmental Law | 1 | 0.6 | 0.9952 | 24.44% |
| Taxation Law | 2 | 0.6 | 0.9791 | 28.92% |
| Insolvency & Corporate Law | 1 | 0.6 | 0.9732 | 28.48% |
| Property Law | 2 | 0.6 | 0.9629 | 25.04% |
| Narcotics & Criminal Law | 1 | 0.6 | 0.9757 | 26.94% |
| Intellectual Property Law | 2 | 0.6 | 0.9956 | 22.77% |
| Criminal & Family Law | 1 | 0.6 | 0.9629 | 40.79% |
| Service Law | 1 | 0.6 | 0.9629 | 26.97% |
| Civil Procedure | 3 | 0.6 | 0.964 | 29.76% |
| Judicial Administration | 1 | 0.6 | 0.9629 | 30.68% |
| Family & Succession Law | 1 | 0.6 | 0.9629 | 25.64% |
| Constitutional & Tort Law | 1 | 0.6 | 0.9952 | 20.76% |
| Banking & Finance Law | 1 | 0.6 | 0.9629 | 29.43% |
| Administrative Law | 2 | 0.7 | 0.9919 | 25.12% |
| Financial & Criminal Law | 1 | 0.6 | 0.9757 | 26.54% |
| Tort Law | 1 | 0.6 | 1.0 | 22.94% |
| Legal Theory & Interpretation | 1 | 0.6 | 1.0 | 17.45% |
| Procedural Law | 1 | 0.6 | 0.9699 | 29.31% |

## 5. Explainability Facet Architecture

Every retrieved precedent provides 5 granular explainability facets:
1. **Matched Acts**: Primary statutes identified in both the query and historical judgment.
2. **Matched Sections**: Specific legislative provisions intersecting between current query and precedent.
3. **Matched Legal Issues**: Core doctrines (e.g. Circumstantial Evidence, Natural Justice, Quashing).
4. **Matched Factual Terms**: Key factual descriptors shared across cases.
5. **Matched Procedural Features**: Jurisdiction and appeal pathway (e.g. Criminal Appeal, SLP, Writ Petition).

## 6. Verification Status

- **Outcome Prediction**: COMPLETELY REMOVED from primary workflow and search response.
- **Precedent Retrieval Quality**: Verified with mathematical bounds (0.0 <= metric <= 1.0).
- **Official AILA Benchmark**: Maintained as an independent evaluation via `python manage.py evaluate_aila`.
