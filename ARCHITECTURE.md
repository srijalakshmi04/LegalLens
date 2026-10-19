# Legal Lens — System Architecture Specification (ARCHITECTURE.md)

## 1. Overview & Positioning

**Legal Lens** implements a **Two-Stage Explainable Legal Information Retrieval Architecture** for Indian Supreme Court judgments ($1950–2025$).

```text
                                USER QUERY
                                    ↓
                           Legal Query Analysis
                                    ↓
                         Legal Query Expansion
                                    ↓
                      Stage 1: TF-IDF Candidate Search
                      (Sparse Matrix: 26,683 × 120,000)
                                    ↓
                             Top-N Candidates
                                    ↓
                       Feature Extraction Layer
        ┌───────────────────────────┼───────────────────────────┐
        ↓                           ↓                           ↓
 Lexical Features            Passage Features          Statutory Features
(TF-IDF Cosine, (Best Sentence Cosine, (Article, Section,
 Term Overlap)               Match Density)             Act Jaccard Overlap)
        └───────────────────────────┬───────────────────────────┘
                                    ↓
                        Stage 2: Supervised Model
                    (Logistic / Gradient Boosting)
                                    ↓
                       Predicted Relevance Score
                                    ↓
                         Stage 2 Reranking
                                    ↓
                   Top-K Supreme Court Judgments
```

---

## 2. Stage-1 Candidate Retrieval
- **Sparse Matrix Index**: $26,683 \text{ rows} \times 120,000 \text{ features}$ (`case_matrix_v2.npz`).
- **Vectorizer**: `TfidfVectorizer(sublinear_tf=True, max_features=120000, ngram_range=(1,2))`.
- **Query Expansion**: `expand_common_legal_language` maps plain-language terms to formal judicial vocabulary.
- **Search Modes**:
  - `Recent`: Hard filter `judgment_year >= 2021`.
  - `All`: Full corpus search.
  - `Current`: Recency-boosted ranking.
  - `Historical`: Prioritizes older landmark judgments (`year <= 2010`).

---

## 3. Stage-2 Supervised Reranking Layer (10 Dimensions)
When activated after human benchmark verification, Stage 2 extracts:
1. `doc_tfidf_similarity`
2. `query_word_count`
3. `passage_max_similarity`
4. `passage_length`
5. `reference_jaccard_overlap`
6. `article_matches_count`
7. `section_matches_count`
8. `act_matches_count`
9. `normalized_judgment_year`
10. `term_overlap_ratio`

---

## 4. Current Operational Engine
- **Active Engine**: Baseline LegalLens Hybrid ($0.50 \times \text{Doc} + 0.35 \times \text{Passage} + 0.15 \times \text{Ref}$).
- **Supervised Model Status**: `Supervised relevance model not trained yet (Human annotation pending)`.
