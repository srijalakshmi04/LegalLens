# Legal Lens — System Limitations & Research Boundaries (LIMITATIONS.md)

## 1. Non-Legal Claim Boundaries
- **Relevance Prediction Only**: Legal Lens predicts information retrieval relevance between a natural language legal query and prior Supreme Court judgments. It **does NOT** predict court outcomes, probability of winning, guilt, innocence, conviction, acquittal, or good-law status.

## 2. Supervised Model Limitations & Production Selection
- **False Negative Risk**: The supervised Logistic Regression reranker exhibits a $21.25\%$ error rate on validation candidates ($17$ false negatives out of $80$ pairs), primarily when statutory section numbers are omitted from query text.
- **Uncalibrated Probabilities**: Model probability outputs are uncalibrated (Brier Score $0.1977$). Probabilities are formatted as `Predicted Relevance Score` rather than calibrated probability.
- **Production Choice**: The **Baseline Heuristic Hybrid Engine ($0.50D + 0.35P + 0.15R$)** is selected for production to maintain zero false-negative risk and peak ranking quality ($MRR = 0.1935, MAP = 0.1742$ on official FIRE AILA 2019 benchmark; early internal split experiments reported $NDCG@10 = 0.9815$).

## 3. External Benchmark Generalization
- **Corpus Indexing Alignment**: External evaluation on FIRE AILA 2019 demonstrated that legal IR models require domain-specific statutory reference indexing and document mapping to achieve high precision across external corpora.
