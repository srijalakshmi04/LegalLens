# Legal Lens — Similarity and Precedent Retrieval Submission Report

> **Project Description**: Legal Lens is an explainable legal case similarity and precedent retrieval system for Indian Supreme Court judgments.
>
> **Strict Operational Boundary**: Pure legal case similarity and precedent retrieval. All outcome predictions (ALLOWED/DISMISSED, conviction/acquittal, future judgment outcomes) have been completely removed.

---

## 1. Primary Submission Metric

$$\text{TOP-1 SIMILARITY RETRIEVAL ACCURACY} = \frac{\text{Queries with Relevant Top-1 Precedent}}{\text{Total Benchmark Queries}} \times 100$$

| Primary Metric | Result | Target | Status |
| :--- | :--- | :--- | :---: |
| **TOP-1 SIMILARITY RETRIEVAL ACCURACY** | **90.00%** | **>=95.00%** | **NOT ACHIEVED** |
| **Queries with Relevant Top-1 Result** | **45 / 50** | - | - |
| **Total Benchmark Queries** | **50** | - | - |
| **Total Corpus Cases** | **2,914** | - | - |
| **Ground Truth Source** | **FIRE AILA 2019 Task 1** | - | - |

---

## 2. Precedent Retrieval Success Rates

| Retrieval Metric | Score | Correct / Total | Description |
| :--- | :--- | :--- | :--- |
| **Top-1 Retrieval Accuracy** | `90.00%` | `45 / 50` | #1 retrieved precedent is officially relevant |
| **Top-5 Retrieval Success** | `100.00%` | `50 / 50` | At least 1 relevant precedent retrieved in Top-5 |
| **Top-10 Retrieval Success** | `100.00%` | `50 / 50` | At least 1 relevant precedent retrieved in Top-10 |

---

## 3. Standard Information Retrieval (IR) Metrics

| IR Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Precision@5** | `0.4400` | Proportion of relevant precedents among the top 5 retrieved cases |
| **Precision@10** | `0.2820` | Proportion of relevant precedents among the top 10 retrieved cases |
| **Recall@5** | `0.7761` | Proportion of all relevant ground-truth precedents retrieved within top 5 |
| **Recall@10** | `0.8811` | Proportion of all relevant ground-truth precedents retrieved within top 10 |
| **MRR (Mean Reciprocal Rank)** | `0.9367` | Average reciprocal rank of the first relevant retrieved precedent |
| **NDCG@10** | `0.8361` | Normalized Discounted Cumulative Gain accounting for rank positions |
| **Mean Top-1 Similarity Score** | `80.49%` | Average calibrated similarity confidence of the primary retrieved precedent |

---

## 4. Query-Level Performance Summary

- **Total Queries Evaluated**: 50
- **Queries Achieving Top-1 Hit**: 45 (90.00%)
- **Queries Achieving Top-5 Hit**: 50 (100.00%)
- **Queries Achieving Top-10 Hit**: 50 (100.00%)
- **Queries Requiring Rank 2-5 for First Relevant Precedent**: 5

---

## 5. Submission Headline Summary

```text
==================================================
LEGAL LENS — SUBMISSION BENCHMARK
==================================================
TOTAL BENCHMARK QUERIES: 50
TOTAL CORPUS CASES: 2,914
GROUND TRUTH: FIRE AILA 2019 Task 1

TOP-1 SIMILARITY RETRIEVAL ACCURACY: 90.00%
QUERIES WITH RELEVANT TOP-1: 45 / 50

TOP-5 RETRIEVAL SUCCESS: 100.00% (50 / 50)
TOP-10 RETRIEVAL SUCCESS: 100.00% (50 / 50)

PRECISION@5: 0.4400
PRECISION@10: 0.2820
RECALL@5: 0.7761
RECALL@10: 0.8811
MRR: 0.9367
NDCG@10: 0.8361

MEAN SIMILARITY SCORE (TOP-1): 80.49%
==================================================

==================================================
PRIMARY METRIC HEADLINE
==================================================
TOP-1 SIMILARITY RETRIEVAL ACCURACY = 90.00%
95% TARGET = NOT ACHIEVED
CORRECT / TOTAL = 45 / 50
==================================================
```
