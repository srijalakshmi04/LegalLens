# Pre-Decision Legal Case Retrieval & Outcome Prediction Report (V2)

## Primary Headline
### **CURRENT-CASE PRE-DECISION SIMILARITY PREDICTION ACCURACY = 60.78%**
### **95% TARGET = NOT ACHIEVED**
- **Required for 95%:** 2,345 / 2468 correct
- **Actual Verified:** 1500 / 2468 correct (60.78%)

---

## 1. Classification Performance (Frozen Held-Out Test Set = 2,468 Cases)
| Metric | Value |
| :--- | :--- |
| **Total Test Cases** | 2468 |
| **Correct Predictions** | **1500** |
| **Incorrect Predictions** | 968 |
| **Accuracy** | **60.78%** |
| **Balanced Accuracy** | 59.35% |
| **Macro F1** | 0.5925 |
| **Weighted F1** | 0.6013 |
| **Coverage** | 100.0% |
| **Abstention Rate** | 0.0% |
| **Brier Score (Calibration)** | 0.2374 |

### Confusion Matrix
- **Labels:** `['ALLOWED', 'DISMISSED']`
- `ALLOWED` as `ALLOWED`: 511
- `ALLOWED` as `DISMISSED`: 585
- `DISMISSED` as `ALLOWED`: 383
- `DISMISSED` as `DISMISSED`: 989

---

## 2. Phase 9: Comprehensive Baseline Comparison (8 Architectures)
| Code | Architecture Model | Validation Acc | Final Test Acc | Test Correct / 2468 |
| :--- | :--- | :--- | :--- | :--- |
| **A** | Existing direct classifier | 62.44% | **60.41%** | 1491 / 2468 |
| **B** | Existing retrieval-only (flat vote) | 59.6% | **57.94%** | 1430 / 2468 |
| **C** | Semantic retrieval (Dense 128-dim SVD) | 59.32% | **57.7%** | 1424 / 2468 |
| **D** | Hybrid hierarchical retrieval (Facts+Issues+Statute+Proc) | 59.04% | **58.55%** | 1445 / 2468 |
| **E** | Outcome-aware CBR retrieval (Directional agreement) | 59.04% | **58.51%** | 1444 / 2468 |
| **F** | Direct classifier + similarity evidence | 63.05% | **60.78%** | 1500 / 2468 |
| **G** | Similarity + structured procedural features | 62.4% | **61.83%** | 1526 / 2468 |
| **H** | Final combined model (CBR + Direct Stacking) | 63.05% | **60.78%** | 1500 / 2468 |

---

## 3. Separate Precedent Retrieval Quality Metrics
*(Evaluates precedent retrieval relevance, strictly decoupled from outcome prediction)*

| Retrieval Metric | Score | Description |
| :--- | :--- | :--- |
| **Precision@5** | **0.9439** | Proportion of top-5 retrieved cases with statutory/issue relevance |
| **Precision@10** | **0.9137** | Proportion of top-10 retrieved cases with statutory/issue relevance |
| **Recall@5** | **0.3028** | Share of relevant precedents captured in top-5 |
| **Recall@10** | **0.5528** | Share of relevant precedents captured in top-10 |
| **MRR (Mean Reciprocal Rank)** | **0.9804** | Reciprocal rank of the first relevant historical precedent |
| **NDCG@10** | **0.978** | Normalized discounted cumulative gain at rank 10 |

---

## 4. Hierarchical CBR Architecture & Leakage Safeguards
- **Model Selected:** `Model H: Hierarchical CBR + Direct Probabilistic Stacking`
- **Selected Retrieval K:** `20`
- **CBR Stacking Alpha:** `0.80` (selected on Validation)

### Component Weights:
1. **Facts Similarity:** `0.25`
2. **Dense Latent Semantic Analysis (128-dim SVD):** `0.20`
3. **Statutory Provisions Similarity (Sections + Acts):** `0.20`
4. **Legal Issues / Question of Law Similarity:** `0.15`
5. **Procedural Posture Similarity:** `0.20`

### Leakage Safeguards Strictly Enforced:
- **Pre-decision text only:** first 3500 chars (no final order or operative disposition)
- **Temporal integrity:** strictly historical cases decided before target prediction timestamp
- **No target retrieval:** target case ID strictly masked out
- **No case-family leakage:** exact and duplicate titles excluded

*Generated on 2026-09-21 21:17:01*