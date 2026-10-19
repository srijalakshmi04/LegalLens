# Legal Lens — Case-to-Case Similarity Prediction Benchmark Report

> **System Task**: Predict whether a historical Supreme Court case is **SIMILAR** or **NOT SIMILAR** to a current legal case. No ALLOWED/DISMISSED outcome prediction is performed.

## 1. Primary Classification Benchmark

- **Primary Metric**: `SIMILARITY PREDICTION ACCURACY = 66.25%`
- **95% Target Status**: **NOT ACHIEVED**
- **Correct Predictions**: `53 / 80`
- **Total Test Pairs**: `80` (`30` SIMILAR, `50` NOT SIMILAR)
- **Hard Negatives Evaluated**: `30` difficult pairs sharing same Act/section but differing legal doctrines.

## 2. Detailed Classification Metrics

| Metric | Score | Note |
|---|---|---|
| **Accuracy** | `66.25%` | Overall correct / total held-out test pairs |
| **Balanced Accuracy** | `59.67%` | Macro average of sensitivity and specificity |
| **Precision (SIMILAR)** | `0.5882` | Fraction of predicted SIMILAR pairs that are genuine precedents |
| **Recall (SIMILAR)** | `0.3333` | Fraction of genuine precedents detected |
| **F1 Score** | `0.4255` | Harmonic mean of precision and recall |
| **Macro F1** | `0.5933` | Unweighted mean across SIMILAR and NOT SIMILAR classes |
| **ROC-AUC** | `0.6633` | Area under the receiver operating characteristic curve |

## 3. Confusion Matrix

```
                Predicted NOT SIMILAR    Predicted SIMILAR
Actual NOT SIMILAR:     43                      7              
Actual SIMILAR:         20                      10             
```

## 4. Separate Precedent Retrieval Metrics (Information Retrieval)

| Retrieval Metric | Score |
|---|---|
| **Precision@5** | `0.64` |
| **Precision@10** | `0.4` |
| **Recall@5** | `1.0` |
| **Recall@10** | `1.0` |
| **MRR** | `0.9` |
| **NDCG@10** | `0.914` |

## 5. Hard Negatives Analysis

Hard negatives represent cases that share the same statutory Act, Section, or general vocabulary as the query, but apply a completely different legal doctrine or factual context. The model explicitly scores these lower to prevent false-positive precedent retrieval based solely on superficial keyword overlap.
