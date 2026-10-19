# Benchmark Scope and Limitations (BENCHMARK_SCOPE_AND_LIMITATIONS.md)

> ⚠️ **Historical / Superseded Experiment — Not Final Benchmark Result**
> *Note: This scope document describes an early 200-case subset evaluation. The official, ground-truth aligned evaluation was extended to all 2,914 candidate documents (`C1`–`C2914`) in `OFFICIAL_AILA_EVALUATION_REPORT.md` and `EVALUATION.md`.*

# Benchmark Scope

- **Evaluation Framework**: FIRE AILA 2019 Task 1 Precedent Retrieval
- **Processed Case Documents**: 200 case documents (Historical Subset) / 2,914 case documents (Official Ground-Truth Corpus)
- **Evaluation Queries**: 50 natural language legal situation queries
- **Mapping Coverage**: 100% ground-truth document identity alignment (`C1`–`C2914`)
- **Scope Boundary**: Evaluation coverage refers strictly to the 200 processed and mapped AILA evaluation cases available to the evaluator under the FIRE AILA 2019 Task 1 framework. Do not claim coverage of the complete 2,914-document corpus.

---

# Evaluation Metrics

The retrieval system is evaluated using standard Information Retrieval (IR) metrics:
- **Precision@5**: Proportion of relevant documents retrieved among top 5 candidates.
- **Precision@10**: Proportion of relevant documents retrieved among top 10 candidates.
- **Recall@10**: Proportion of total relevant documents retrieved in the top 10 results.
- **Mean Reciprocal Rank (MRR)**: Mean Reciprocal Rank (MRR) is computed by taking the reciprocal of the rank of the first relevant retrieved document for each query and averaging these reciprocal ranks across all evaluated queries.
- **NDCG@10**: Normalized Discounted Cumulative Gain at rank 10, assessing graded relevance ranking quality.
- **Mean Average Precision (MAP)**: Mean of Average Precision scores across all evaluated queries.

---

# Methodological Limitation

The current experimental evaluation assesses system performance over the available processed and mapped 200-case evaluation dataset subset under the FIRE AILA 2019 Task 1 evaluation framework, rather than claiming a complete evaluation of all 2,914 case documents in the original benchmark-scale target.

---

# Research Paper Safe Wording

For direct inclusion in research papers, theses, or viva presentations, use the following academically rigorous phrasing:

1. *"We evaluate Legal Lens under the FIRE AILA 2019 Task 1 Precedent Retrieval benchmark framework using a processed and mapped evaluation subset of 200 Supreme Court case documents and 50 expert legal queries."*
2. *"Our evaluation achieves 100% mapping coverage over the available 200-case evaluation subset, providing a controlled experimental baseline for multi-stage legal IR."*
3. *"All metrics—including Precision@K, Recall@K, NDCG@10, MAP, and Mean Reciprocal Rank (MRR)—are mathematically bounded in [0, 1.0] and computed directly from query ranking outputs without artificial scaling."*
