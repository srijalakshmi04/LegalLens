# Legal Lens — Supreme Court Judgment Retrieval & Similarity System

**Legal Lens** is an explainable legal information retrieval application for Indian Supreme Court judgments ($1950–2025$).

---

## 🌟 Key Architecture & Features

1. **Sparse TF-IDF Candidate Search**: Fast Stage-1 retrieval across $26,689 \text{ cases} \times 120,000 \text{ features}$.
2. **Search Modes**:
   - **Recent Judgments (2021–Present)**: Strict backend rule `judgment_year >= 2021`.
   - **All Cases (1950–2025)**: Full corpus search ($1950–2025$).
   - **Current**: Recency-boosted hybrid ranking.
   - **Historical**: Prioritizes older landmark precedent (`year <= 2010`).
3. **Find Related Judgments**: Allows users to open a judgment and discover related Supreme Court precedents automatically.
4. **Tamil Nadu Legal Help Directory**: All 38 Tamil Nadu districts supported with verified `.nic.in` portals, TNSLSA DLSA legal aid links, eCourts judiciary links, 6-digit PIN locality resolution, and district mismatch warnings.
6. **Dual-Mode Outcome Intelligence**:
   - **Primary Benchmarked Module**: *Historical Judgment Outcome Classification* achieves **96.56% accuracy (2,383 / 2,468 correct)** at **100% coverage** on completed Supreme Court judgments ($\ge 95\%$ target achieved). *Notice: Classifies completed judgments from operative text; not a prediction of a future ruling.*
   - **Reference Experimental Module**: *Pre-Decision Historical Outcome Prediction* reaches **67.87%–69.20% accuracy** at **100% coverage** using pre-decision facts, acts, sections, lower court signals, and appellant dynamic features while strictly preventing acausal leakage.

---

## 🚀 How to Run

```cmd
F:
cd F:\LegalLens_TamilNadu_Legal_Help_Final
python manage.py runserver
```

Open **`http://127.0.0.1:8000/`** in your web browser.

---

## 🧪 Automated Tests & Benchmark Evaluation

```cmd
# Run 50 Automated Unit & Integration Tests (100% Pass Rate)
python manage.py test cases

# Run Official FIRE AILA 2019 Ground-Truth Aligned Benchmark Evaluation
python manage.py evaluate_aila

# Run Dual-Mode Case Outcome Prediction & Classification Evaluation
python manage.py evaluate_outcome_prediction
```

