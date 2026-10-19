# Legal Lens — Dataset & Extraction Audit Specification (DATASET.md)

## 1. Corpus Statistics
- **Total Indexed Supreme Court Judgments**: 26,688
- **Timeline Coverage**: 1950 – 2025 (75 Years of Jurisprudence)
- **Database Engine**: SQLite3 (`3.77 GB`)
- **Indexed Matrix Rows**: 26,683 (`case_matrix_v2.npz`)
- **Vocabulary Size**: 120,000 TF-IDF features (`tfidf_vectorizer_v2.joblib`)

---

## 2. Extraction Audit Breakdown
- **Complete Extracted Text**: 26,647 Judgments ($99.85\%$)
- **Partial Extraction**: 36 Judgments ($0.13\%$)
- **Scanned PDF Text Extraction**: 5 Judgments ($0.02\%$)

---

## 3. Ground-Truth Benchmark Dataset Specification
- **Benchmark Location**: `cases/data/relevance_benchmark.json`
- **Total Benchmark Queries**: 50 Queries across 10 Legal Domains
- **Annotation Status**: `Human annotation pending`
- **Query-Level Splits**:
  - `Train`: 30 Queries ($60\%$)
  - `Validation`: 10 Queries ($20\%$)
  - `Test`: 10 Queries ($20\%$)
