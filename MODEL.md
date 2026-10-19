# Legal Lens — Machine Learning Model Architecture & Reranking Specification (MODEL.md)

## 1. Production Model Architecture: Baseline Heuristic Hybrid Engine

The active production model for **Legal Lens** is the **Baseline Heuristic Hybrid Engine**:

$$\text{Score}(Q, D) = 0.50 \cdot \text{Doc\_TFIDF}(Q, D) + 0.35 \cdot \text{Passage\_Sim}(Q, D) + 0.15 \cdot \text{Ref\_Jaccard}(Q, D)$$

### Why the Heuristic Hybrid is Selected for Production:
1. **Deterministic & Transparent**: Provides clear, explainable score breakdown across document similarity, passage evidence, and statutory references.
2. **Empirically Superior on Official Benchmark**: Achieves peak $MRR = 0.1935$ and $MAP = 0.1742$ on the official FIRE AILA 2019 benchmark without synthetic metric distortions. *(Note: Early internal split experiments reported $NDCG@10 = 0.9815$ under synthetic pseudo-labels, which is preserved as a historical baseline).*
3. **Zero Miscalibration Risk**: Completely eliminates false negative risks associated with uncalibrated supervised probability predictions.

---

## 2. Stage-2 Supervised Relevance Reranker (Research Specification)

### 10-Dimensional Feature Extraction Vector $\mathbf{x} \in \mathbb{R}^{10}$:
1. $x_1$: `doc_tfidf_similarity` — Full document TF-IDF cosine score $[0.0, 1.0]$
2. $x_2$: `query_word_count` — Word length of natural language query $[1, 200]$
3. $x_3$: `passage_max_similarity` — Best sentence-level TF-IDF cosine score $[0.0, 1.0]$
4. $x_4$: `passage_length` — Character length of evidence passage $[0, 2000]$
5. $x_5$: `reference_jaccard_overlap` — Statutory reference Jaccard overlap ratio $[0.0, 1.0]$
6. $x_6$: `article_matches_count` — Constitutional Articles count $[0, 50]$
7. $x_7$: `section_matches_count` — Statutory Sections count $[0, 100]$
8. $x_8$: `act_matches_count` — Named statutory Acts count $[0, 20]$
9. $x_9$: `normalized_judgment_year` — Judgment year normalized as $(year - 1950.0) / 75.0 \in [0.0, 1.0]$
10. $x_{10}$: `term_overlap_ratio` — Lexical keyword overlap ratio $[0.0, 1.0]$

### Empirical Model Performance (Validation Split):
- **Logistic Regression (Primary)**: Accuracy = $71.25\%$, ROC-AUC = $0.7140$, Brier Score = $0.1977$
- **Random Forest**: Accuracy = $62.50\%$, ROC-AUC = $0.6993$
- **Gradient Boosting**: Accuracy = $63.75\%$, ROC-AUC = $0.6650$

### Top Learned Weights ($\beta_i$):
- `reference_jaccard_overlap`: $+0.9242$
- `passage_max_similarity`: $+0.5301$
- `doc_tfidf_similarity`: $+0.3006$
- `act_matches_count`: $-0.4345$
- `passage_length`: $-0.3573$

### Calibration Status:
Uncalibrated (Brier Score $0.1977$). In user interface displays, supervised predictions are formatted strictly as **`Predicted Relevance Score`** rather than probability.
