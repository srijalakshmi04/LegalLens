# Legal Case Similarity Ground Truth Documentation

## 1. Overview & Research Foundation
Legal Lens evaluates case-to-case legal similarity prediction using verified ground truth from the **FIRE AILA 2019 (Artificial Intelligence for Legal Assistance) Task 1 (Precedent Retrieval Track)**.

The benchmark consists of:
- **Queries**: 50 detailed natural-language legal situations / dispute scenarios describing facts, legal issues, statutes, and procedural posture.
- **Candidate Case Documents**: 2,914 Indian Supreme Court judgments (C1.txt to C2914.txt), comprising full unredacted decisions from 1950 to the late 2010s.
- **Relevance Judgments (qrels)**: 145,700 official human-annotated binary judgments (external_benchmarks/AILA_2019/raw/relevance_judgments_priorcases.txt) where each candidate case is assessed by legal experts for precedential relevance to each query.

---

## 2. Official Relevance Mapping to Binary Similarity

The official AILA qrels assign relevance grades:
- Grade 1: The historical case serves as an official relevant precedent for the legal situation.
- Grade 0: The historical case does not serve as a precedent (irrelevant or distinguishing).

### Documented Mapping Rule:
- Relevance Grade = 1 -> SIMILAR (y = 1)
- Relevance Grade = 0 -> NOT SIMILAR (y = 0)

The original official qrels file is preserved 100% intact without alteration. No labels are fabricated or synthetically inverted.

---

## 3. Query-Level Non-Leaking Split

To strictly prevent data leakage across queries and candidate cases, splitting is performed strictly at the **QUERY** level:
- **Total Queries**: 50 queries (AILA_Q1 to AILA_Q50)
- **Train Split (35 queries, 70%)**: Used for feature extraction, representation fitting, and training candidate pair classifiers.
- **Validation Split (7 queries, 14%)**: Used exclusively for hyperparameter comparison, probability calibration (isotonic/Platt scaling), and decision threshold optimization.
- **Held-Out Test Split (8 queries, 16%)**: Strictly untouched during training, calibration, and threshold selection. Evaluated exactly once with the locked model.

Candidate cases for a test query are evaluated only within the test context; no query or candidate cross-pair leakage exists between splits.

---

## 4. Benchmark Construction & Pair Sampling Protocol

Because the full corpus contains 195 positive precedents and 145,505 non-relevant pairs across 50 queries (reflecting the true needle-in-a-haystack nature of legal precedent discovery), the benchmark evaluates two complementary configurations:

### A. The Balanced 50/50 Benchmark Subset
To evaluate classification performance without extreme class imbalance distortion:
1. **Positive Pairs (50%)**: All official relevant precedents ( = 1$) for the split queries.
2. **Negative Pairs (50%)**: Matched 1:1 negative candidates ( = 0$), subdivided into:
   - **Hard Negatives (50% of negatives)**: Historical cases sharing the same legal domain, Acts, statutory sections, or high TF-IDF/BM25 vocabulary overlap with the query, but officially judged 0 (NOT SIMILAR). This directly penalizes models that rely on superficial keyword matching.
   - **Distant Negatives (50% of negatives)**: Randomly sampled corpus judgments from unrelated legal domains.

### B. The Large Candidate Evaluation Pool (>10,000 Labelled Pairs)
For the held-out test queries, all 2,914 candidates per query ( \times 2,914 = 23,312$ officially judged pairs) provide a large-scale real-world test of the similarity model\'s retrieval and discrimination capabilities.

---

## 5. Separation from Outcome Prediction & AILA Retrieval

1. **Zero ALLOWED / DISMISSED Leakage**:
   The similarity model predicts strictly **SIMILAR vs NOT SIMILAR**. Final judgment outcomes, dispositions (ALLOWED, DISMISSED, CONVICTION, ACQUITTAL), and post-decision judicial remarks are completely excluded from all feature representations.
2. **Separation from evaluate_aila**:
   python manage.py evaluate_aila remains the official IR benchmark measuring ranking quality (P@K, MRR, NDCG@10).
   python manage.py evaluate_similarity_accuracy measures pair-level classification accuracy, precision, recall, F1, balanced accuracy, and hard-negative accuracy.
