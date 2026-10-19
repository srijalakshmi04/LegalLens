import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from cases.services.similarity import (
    load_similarity_model,
    clean_text,
    expand_common_legal_language,
    find_best_passage,
    normalize_legal_references,
    jaccard_overlap,
    matched_terms,
)

FEATURE_NAMES = [
    "doc_tfidf_similarity",
    "query_word_count",
    "passage_max_similarity",
    "passage_length",
    "reference_jaccard_overlap",
    "article_matches_count",
    "section_matches_count",
    "act_matches_count",
    "normalized_judgment_year",
    "term_overlap_ratio",
]

def extract_features_for_pair(query_text, legal_case, vectorizer=None, query_vector=None):
    """
    Extracts a 10-dimensional feature vector for a (query, candidate_case) pair.
    Used for 2-stage supervised relevance reranking and model training.
    """
    if vectorizer is None or query_vector is None:
        v, _, _, _ = load_similarity_model()
        vectorizer = v
        cleaned_query = clean_text(query_text)
        expanded_query = expand_common_legal_language(cleaned_query)
        query_vector = vectorizer.transform([expanded_query])
    else:
        cleaned_query = clean_text(query_text)
        expanded_query = expand_common_legal_language(cleaned_query)

    # 1. Document TF-IDF Cosine Similarity
    case_vec = vectorizer.transform([legal_case.cleaned_text or legal_case.raw_text or ""])
    doc_score = float(linear_kernel(query_vector, case_vec).ravel()[0])

    # 2. Query Word Count
    query_words = cleaned_query.split()
    query_word_count = float(len(query_words))

    # 3. Passage Max Similarity & 4. Passage Length
    best_passage, passage_score = find_best_passage(
        legal_case.raw_text or legal_case.cleaned_text or "",
        expanded_query,
        vectorizer,
        query_vector,
    )
    passage_length = float(len(best_passage))

    # 5. Legal Reference Jaccard Overlap
    query_refs = normalize_legal_references(cleaned_query)
    case_refs = normalize_legal_references(
        " ".join([
            legal_case.legal_sections or "",
            legal_case.acts or "",
            best_passage,
        ])
    )
    ref_overlap = float(jaccard_overlap(query_refs, case_refs))

    # 6. Constitutional Article Matches
    query_articles = {r for r in query_refs if r.startswith("article_")}
    case_articles = {r for r in case_refs if r.startswith("article_")}
    article_matches = float(len(query_articles & case_articles))

    # 7. Statutory Section Matches
    query_sections = {r for r in query_refs if r.startswith("section_")}
    case_sections = {r for r in case_refs if r.startswith("section_")}
    section_matches = float(len(query_sections & case_sections))

    # 8. Statutory Act Matches
    query_acts = {r for r in query_refs if not (r.startswith("article_") or r.startswith("section_"))}
    case_acts = {r for r in case_refs if not (r.startswith("article_") or r.startswith("section_"))}
    act_matches = float(len(query_acts & case_acts))

    # 9. Normalized Judgment Year
    year = float(legal_case.judgment_year or 1950)
    norm_year = max(0.0, min(1.0, (year - 1950.0) / 75.0))

    # 10. Term Overlap Ratio
    terms = matched_terms(cleaned_query, best_passage or legal_case.raw_text or "")
    term_ratio = float(len(terms)) / max(1.0, float(len(set(query_words))))

    feature_vector = np.array([
        doc_score,
        query_word_count,
        float(passage_score),
        passage_length,
        ref_overlap,
        article_matches,
        section_matches,
        act_matches,
        norm_year,
        term_ratio,
    ], dtype=np.float64)

    return feature_vector, best_passage
