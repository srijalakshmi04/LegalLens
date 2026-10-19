# Legal Lens — Similarity Prediction Accuracy Benchmark Report

## 1. Executive Summary
- **Primary Task**: Case-to-Case Binary Similarity Prediction (SIMILAR vs NOT SIMILAR)
- **Description**: "Legal Lens is an explainable legal case similarity and precedent retrieval system for Indian Supreme Court judgments."
- **Ground Truth**: Official FIRE AILA 2019 Task 1 Precedent Retrieval judgments (`relevance_judgments_priorcases.txt`)
- **Leakage Prevention**: Query-level split (35 Train, 7 Val, 8 Held-out Test queries)
- **Decision Threshold**: Validation set only (61.0%)

---

## 2. Primary Classification Results (Held-Out Test Pairs: N = 2000)

| Metric | Value | Target | Status |
| :--- | :--- | :--- | :---: |
| **SIMILARITY PREDICTION ACCURACY** | **56.75%** | **>=95.00%** | **NOT ACHIEVED** |
| **Correct Predictions** | **1135 / 2000** | - | - |
| **Incorrect Predictions** | **865 / 2000** | - | - |
| **Total Test Pairs** | **2000** | >=2,000 | ACHIEVED |
| **Similar Pairs (Positive)** | **26** | - | - |
| **Not Similar Pairs (Negative)** | **1974** | - | - |
| **Precision** | **0.0173** | - | - |
| **Recall** | **0.5769** | - | - |
| **F1 Score** | **0.0335** | - | - |
| **Macro F1 Score** | **0.3775** | - | - |
| **Balanced Accuracy** | **57.21%** | - | - |
| **ROC-AUC** | **0.5689** | - | - |

---

## 3. Confusion Matrix
```text
[[TN=1120  FP=854]
 [FN=11  TP=15]]
```
- **True Negatives**: 1120
- **False Positives**: 854
- **False Negatives**: 11
- **True Positives**: 15

---

## 4. Separate Precedent Retrieval Metrics (Ranking Quality)
| Retrieval Metric | Score |
| :--- | :--- |
| **Precision@5** | 0.2000 |
| **Precision@10** | 0.1000 |
| **Recall@5** | 0.3333 |
| **Recall@10** | 0.3333 |
| **MRR** | 0.4375 |
| **NDCG@10** | 0.4539 |

---

## 5. Balanced Test Subset (50/50 Pairs)
- **Total Pairs**: 52
- **Balanced Subset Accuracy**: 65.38%
- **Balanced Subset Balanced Accuracy**: 65.38%
- **Balanced Subset ROC-AUC**: 0.7071

---

## 6. Final Headline
```text
========================================================
SIMILARITY PREDICTION ACCURACY = 56.75%
95% TARGET = NOT ACHIEVED
CORRECT / TOTAL = 1135 / 2000
========================================================
```
