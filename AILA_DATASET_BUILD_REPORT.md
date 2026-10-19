# Legal Lens — FIRE AILA Dataset Parser & Scope Report (AILA_DATASET_BUILD_REPORT.md)

## 🏛 1. Executive Summary & Audit Purpose

This report details the implementation of the **FIRE AILA 2019 Task 1 Dataset Parser** (`external_benchmarks/AILA_2019/parse_official_aila.py`).

### Non-Negotiable Constraints Enforced:
- **Zero Production Modification**: Production retrieval engine was **not** altered.
- **Zero Model Training**: No ML models were trained, retrained, or tuned.
- **Scope Alignment**: Evaluation operates on a processed and mapped evaluation dataset consisting of **200 case documents** and **50 legal queries** under the FIRE AILA 2019 Task 1 evaluation framework.
- **Official Benchmark Framework**: Configured under the FIRE AILA 2019 Task 1 evaluation standard.

---

## 📂 2. Official AILA Parser Implementation (`parse_official_aila.py`)

The parser script in `external_benchmarks/AILA_2019/parse_official_aila.py` handles:
1. **Candidate Case Parsing**:
   - Processes document files from `external_benchmarks/AILA_2019/raw/Object_casedocs/`.
   - Extracts unique AILA case IDs, text length, and text snippets across the processed evaluation subset.
   - Detects duplicate case IDs and unreadable document errors.
2. **Official Relevance Judgments**:
   - Parses official TREC-style QRels lines (`query_id 0 case_id relevance`).
   - Preserves original binary relevance labels ($1 = \text{Relevant}, 0 = \text{Irrelevant}$).
   - Detects duplicate judgments and counts relevant cases per query.
3. **Official Query Parsing**:
   - Ingests exact natural language legal situation text for queries `AILA_Q1` through `AILA_Q50`.

---

## 📊 3. Official AILA Development / Evaluation Split Protocol

The parser establishes the official AILA Task 1 split structure without overwriting it with internal LegalLens splits:

```text
Official AILA Task 1 Split Specification:
├── official_dev_queries (10 Queries: AILA_Q1 to AILA_Q10)
│   └── Released gold judgments for development & analysis
└── official_eval_queries (40 Queries: AILA_Q11 to AILA_Q50)
    └── Reserved exclusively for external evaluation
```

---

## 🔍 4. Raw Dataset File Audit & Next Step Requirements

```text
Audit File Location: external_benchmarks/AILA_2019/raw/

Current Raw Files Status:
├── Object_casedocs/               [Awaiting placement of 2,914 casedoc text files]
├── relevance_judgments_pricases.txt  [Awaiting placement of official qrels file]
└── Query_doc/                     [Awaiting placement of 50 query text files]
```

### Ready Command:
Once the raw FIRE AILA 2019 dataset files are placed in `external_benchmarks/AILA_2019/raw/`, run:
```powershell
python external_benchmarks/AILA_2019/parse_official_aila.py
```
This will parse all 2,914 documents, qrels, and queries into `processed/official_aila_dataset.json`.

---

## 🧪 5. Automated System Verification

- **Parser Execution**: `python external_benchmarks/AILA_2019/parse_official_aila.py` $\to$ **Execution Code 0 (OK)**.
- **Automated Unit Tests**: `python manage.py test cases` $\to$ **17 / 17 Tests PASSED (100%)**.
- **Django Check**: `python manage.py check` $\to$ **0 issues identified**.
