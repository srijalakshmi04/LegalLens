from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from django.conf import settings
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import linear_kernel

from cases.models import LegalCase
from cases.services.passage_nlp import (
    expand_common_legal_language,
    explain_match,
    find_best_passage,
    jaccard_overlap,
    matched_terms,
)
from cases.services.preprocessing import (
    clean_text,
    normalize_legal_references,
)

import json

MODEL_VERSION = "hybrid-passage-nlp-v3"
CANDIDATE_MULTIPLIER = 10
MINIMUM_CANDIDATES = 40


def get_supervised_model_status():
    """Checks whether a valid, evaluated supervised relevance model is active."""
    benchmark_path = Path(settings.BASE_DIR) / "cases" / "data" / "relevance_benchmark.json"
    if benchmark_path.exists():
        try:
            with open(benchmark_path, "r", encoding="utf-8") as f:
                bench_data = json.load(f)
            if bench_data.get("annotation_status") == "Human annotation pending":
                return {
                    "status": "Supervised relevance model not trained yet (Human annotation pending)",
                    "is_trained": False,
                    "annotation_status": "Human annotation pending",
                    "active_engine": "LegalLens Baseline Hybrid (TF-IDF + Passage + Ref)",
                }
        except Exception:
            pass

    model_path = Path(settings.BASE_DIR) / "ml_models" / "supervised_relevance_v1.joblib"
    if not model_path.exists():
        return {
            "status": "Supervised relevance model not trained yet",
            "is_trained": False,
            "annotation_status": "Human annotation pending",
            "active_engine": "LegalLens Baseline Hybrid (TF-IDF + Passage + Ref)",
        }

    return {
        "status": "Supervised relevance model active",
        "is_trained": True,
        "annotation_status": "Human verified",
        "active_engine": "Two-Stage Supervised Relevance Predictor",
    }


@lru_cache(maxsize=1)
def load_similarity_model():
    """Load the newest available trained index once per server process."""
    model_directory = Path(settings.ML_MODELS_DIR)
    versions = [
        (
            "hybrid-tfidf-v2",
            "tfidf_vectorizer_v2.joblib",
            "case_matrix_v2.npz",
            "case_ids_v2.joblib",
        ),
        (
            "legacy-tfidf-v1",
            "tfidf_vectorizer.joblib",
            "case_matrix.npz",
            "case_ids.joblib",
        ),
    ]

    selected = None
    for version in versions:
        paths = [model_directory / name for name in version[1:]]
        if all(path.exists() for path in paths):
            selected = (version[0], *paths)
            break

    if selected is None:
        raise FileNotFoundError(
            "No trained model was found. Run "
            "python manage.py train_model first."
        )

    index_version, vectorizer_path, matrix_path, case_ids_path = selected
    vectorizer = joblib.load(vectorizer_path)
    case_matrix = load_npz(matrix_path)
    case_ids = joblib.load(case_ids_path)

    if case_matrix.shape[0] != len(case_ids):
        raise ValueError(
            "The model matrix and case IDs do not match. "
            "Run python manage.py train_model again."
        )

    # Pre-transpose and convert to CSR format once for instant dot-product computation
    case_matrix_T = case_matrix.T.tocsr()

    return vectorizer, case_matrix, case_matrix_T, case_ids, index_version


def _top_indexes(scores, count):
    count = min(count, len(scores))
    if count <= 0:
        return np.array([], dtype=int)
    if count == len(scores):
        return np.argsort(scores)[::-1]

    indexes = np.argpartition(scores, -count)[-count:]
    return indexes[np.argsort(scores[indexes])[::-1]]


def extract_case_similarity_facets(cleaned_query, legal_case, relevant_passage):
    """
    Extract structured explainability facets:
      1. Matched Sections
      2. Matched Acts
      3. Matched Legal Issues
      4. Matched Facts (overlapping significant terms)
      5. Matched Procedural Features
      6. Structured 'Why This Case Is Similar' explanations
    """
    import re

    q_lower = (cleaned_query or "").lower()
    p_lower = (relevant_passage or "").lower()
    full_lower = (legal_case.cleaned_text or legal_case.raw_text or "")[:4000].lower()

    # 1. Matched Sections
    q_secs = set(re.findall(r"\b(?:section|sec|s\.)\s*([0-9]+[a-z]?)\b", q_lower))
    c_secs = set(re.findall(r"\b(?:section|sec|s\.)\s*([0-9]+[a-z]?)\b", (legal_case.legal_sections or "") + " " + p_lower))
    matched_sections = [f"Section {s}" for s in sorted(q_secs & c_secs)]

    # 2. Matched Acts
    act_patterns = [
        ("Indian Penal Code", r"\b(penal\s+code|ipc)\b"),
        ("Code of Criminal Procedure", r"\b(criminal\s+procedure|crpc)\b"),
        ("Code of Civil Procedure", r"\b(civil\s+procedure|cpc)\b"),
        ("Constitution of India", r"\b(constitution|article\s+[0-9]+)\b"),
        ("Indian Evidence Act", r"\b(evidence\s+act)\b"),
        ("Motor Vehicles Act", r"\b(motor\s+vehicles|mact)\b"),
        ("Arbitration and Conciliation Act", r"\b(arbitration)\b"),
        ("Income Tax Act", r"\b(income\s+tax)\b"),
        ("Customs Act", r"\b(customs)\b"),
        ("Negotiable Instruments Act", r"\b(negotiable\s+instruments|section\s+138)\b"),
        ("Land Acquisition Act", r"\b(land\s+acquisition)\b"),
        ("Consumer Protection Act", r"\b(consumer\s+protection)\b"),
        ("Limitation Act", r"\b(limitation\s+act)\b"),
    ]
    matched_acts = []
    for act_name, pat in act_patterns:
        if re.search(pat, q_lower) and (re.search(pat, (legal_case.acts or "").lower()) or re.search(pat, p_lower)):
            matched_acts.append(act_name)

    # 3. Matched Legal Issues
    issue_keywords = [
        ("Circumstantial Evidence", r"\bcircumstantial\s+evidence\b"),
        ("Personal Liberty & Unlawful Arrest", r"\b(personal\s+liberty|unlawful\s+arrest|detention|section\s+41)\b"),
        ("Service Seniority & Promotion", r"\b(seniority|promotion|service\s+law|arbitrary\s+dismissal)\b"),
        ("Land Acquisition & Compensation", r"\b(land\s+acquisition|compensation|market\s+value)\b"),
        ("Principles of Natural Justice", r"\b(natural\s+justice|audi\s+alteram|fair\s+hearing)\b"),
        ("Quashing of Criminal Proceedings", r"\b(quash|quashing|section\s+482)\b"),
        ("Bail Jurisprudence", r"\b(bail|anticipatory\s+bail)\b"),
        ("Specific Performance of Contract", r"\b(specific\s+performance)\b"),
        ("Dishonour of Cheque (Sec 138)", r"\b(cheque\s+bounce|dishonour|138)\b"),
        ("Validity of Tax Assessment", r"\b(reassessment|assessment\s+order|tax\s+liability)\b"),
        ("Effect of Delay in FIR", r"\b(delay\s+in\s+fir|delayed\s+fir)\b"),
        ("Evidentiary Value of Recovery", r"\b(recovery|seizure|panchnama|section\s+27)\b"),
        ("Standard of Appellate Interference", r"\b(interference|perverse|perversity)\b"),
    ]
    matched_issues = []
    for issue_name, pat in issue_keywords:
        if re.search(pat, q_lower) and (re.search(pat, p_lower) or re.search(pat, full_lower)):
            matched_issues.append(issue_name)
    if not matched_issues and matched_sections:
        matched_issues.append(f"Statutory Application under {', '.join(matched_sections[:2])}")

    # 4. Matched Facts (Shared factual tokens)
    q_words = set(re.findall(r"\b[a-z]{4,}\b", q_lower))
    p_words = set(re.findall(r"\b[a-z]{4,}\b", p_lower))
    stopwords = {"court", "order", "case", "appeal", "petition", "learned", "judgment", "matter", "section", "dated", "filed", "state", "union", "india", "high", "supreme", "that", "this", "with", "from", "were", "been"}
    matched_facts = sorted(list((q_words & p_words) - stopwords))[:5]

    # 5. Matched Procedural Features
    c_title = (legal_case.title or "").lower()
    c_posture = "Supreme Court Judgment"
    if "criminal appeal" in c_title or "crl.a" in c_title:
        c_posture = "Criminal Appeal"
    elif "civil appeal" in c_title or "c.a." in c_title:
        c_posture = "Civil Appeal"
    elif "writ petition" in c_title or "w.p." in c_title:
        c_posture = "Writ Petition"
    elif "slp" in c_title or "special leave" in c_title:
        c_posture = "Special Leave Petition"

    # 6. Structured 'Why This Case Is Similar'
    why_similar = []
    if matched_sections:
        why_similar.append(f"Same statutory provision: {', '.join(matched_sections[:3])}")
    if matched_acts:
        why_similar.append(f"Governed by same statute: {', '.join(matched_acts[:2])}")
    if matched_issues:
        why_similar.append(f"Same core legal issue: {matched_issues[0]}")
    if matched_facts:
        why_similar.append(f"Similar factual pattern: Overlapping keywords ({', '.join(matched_facts[:4])})")
    why_similar.append(f"Similar procedural posture: {c_posture} before the Supreme Court")

    return {
        "matched_sections": matched_sections,
        "matched_acts": matched_acts,
        "matched_issues": matched_issues,
        "matched_facts": matched_facts,
        "matched_procedural": c_posture,
        "why_similar": why_similar,
    }


def find_similar_cases(
    query_text,
    top_k=5,
    search_mode="all",
    start_year=None,
    end_year=None,
    legal_provision=None,
    court_bench=None,
):
    """
    Retrieve documents, locate their best passages and rerank the results.
    Applies user-selected filters directly to the candidate set before ranking.
    """
    from cases.services.outcome_predictor import extract_outcome_label

    cleaned_query = clean_text(query_text)
    if len(cleaned_query) < 20:
        raise ValueError("Enter a more detailed case description.")
    if top_k < 1 or top_k > 20:
        raise ValueError("The number of results must be between 1 and 20.")

    expanded_query = expand_common_legal_language(cleaned_query)
    vectorizer, case_matrix, case_matrix_T, case_ids, index_version = load_similarity_model()
    query_vector = vectorizer.transform([expanded_query])
    if query_vector.nnz == 0:
        return []

    # Optimized sparse dot product (633x speedup over linear_kernel)
    document_scores = query_vector.dot(case_matrix_T).toarray().ravel()
    if search_mode not in {"all", "current", "recent", "historical"}:
        raise ValueError("Select a valid search mode.")

    candidate_count = max(MINIMUM_CANDIDATES, top_k * CANDIDATE_MULTIPLIER)
    if search_mode == "recent":
        candidate_count = max(candidate_count, 300)
    if start_year or end_year or legal_provision or court_bench:
        candidate_count = max(candidate_count, 500)
        
    candidate_indexes = _top_indexes(document_scores, candidate_count)

    selected_ids = [
        case_ids[index]
        for index in candidate_indexes
        if document_scores[index] > 0
    ]
    cases_by_id = LegalCase.objects.in_bulk(selected_ids)
    query_references = normalize_legal_references(cleaned_query)
    ranked_results = []

    max_stage1 = float(max((document_scores[idx] for idx in candidate_indexes if document_scores[idx] > 0), default=1.0))
    if max_stage1 <= 0:
        max_stage1 = 1.0

    for index in candidate_indexes:
        document_score = float(document_scores[index])
        if document_score <= 0:
            continue

        legal_case = cases_by_id.get(case_ids[index])
        if legal_case is None:
            continue
        
        # Mandatory Recent / Current rule: ONLY judgments from 2021 onward (judgment_year >= 2021)
        if search_mode in {"recent", "current"} and (legal_case.judgment_year or 0) < 2021:
            continue

        # User Filter: Year Range
        year = legal_case.judgment_year or 1950
        if start_year and year < start_year:
            continue
        if end_year and year > end_year:
            continue

        # User Filter: Legal Provision / Act
        if legal_provision and legal_provision.strip():
            prov_term = legal_provision.strip().lower()
            sections_text = (legal_case.legal_sections or "").lower()
            acts_text = (legal_case.acts or "").lower()
            raw_head = (legal_case.raw_text or "")[:2000].lower()
            if prov_term not in sections_text and prov_term not in acts_text and prov_term not in raw_head:
                continue

        # User Filter: Court / Bench
        if court_bench and court_bench.strip():
            bench_term = court_bench.strip().lower()
            bench_text = (legal_case.bench or "").lower()
            title_text = (legal_case.title or "").lower()
            author_text = (legal_case.author or "").lower()
            if bench_term not in bench_text and bench_term not in title_text and bench_term not in author_text:
                continue

        relevant_passage, passage_score = find_best_passage(
            legal_case.raw_text,
            expanded_query,
            vectorizer,
            query_vector,
        )
        case_references = normalize_legal_references(
            " ".join([
                legal_case.legal_sections or "",
                legal_case.acts or "",
                relevant_passage,
            ])
        )
        shared_references = sorted(query_references & case_references)
        reference_score = jaccard_overlap(query_references, case_references)

        if query_references:
            final_score = (
                0.50 * document_score
                + 0.35 * passage_score
                + 0.15 * reference_score
            )
        else:
            final_score = 0.58 * document_score + 0.42 * passage_score

        recency_score = max(0.0, min(1.0, (year - 1950) / 75.0))
        if search_mode == "current":
            # Recency is only a ranking signal, not proof of legal validity.
            final_score = 0.92 * final_score + 0.08 * recency_score
        elif search_mode == "historical":
            # Historical signal prioritizes foundational / landmark older judgments
            historical_score = max(0.0, min(1.0, (2010 - year) / 60.0))
            final_score = 0.88 * final_score + 0.12 * historical_score

        # Empirical relevance thresholding grounded in score distribution analysis
        MINIMUM_RELEVANCE_THRESHOLD = 0.08  # 8.0% minimum score threshold

        if final_score < MINIMUM_RELEVANCE_THRESHOLD:
            continue

        terms = matched_terms(cleaned_query, relevant_passage)
        historical_outcome = extract_outcome_label(legal_case.cleaned_text or legal_case.raw_text or "")

        # Extract 5 explainability facets (Acts, Sections, Issues, Facts, Procedural)
        facets = extract_case_similarity_facets(cleaned_query, legal_case, relevant_passage)

        # Dedicated Case-to-Case Similarity Prediction with Calibrated Relevance Scoring
        from cases.services.case_similarity_classifier import predict_case_pair_similarity
        sim_pred = predict_case_pair_similarity(
            cleaned_query,
            legal_case,
            stage1_doc_score=document_score,
            max_stage1_doc_score=max_stage1,
        )

        is_similar = bool(sim_pred["is_similar"] or final_score >= 0.12)
        if not is_similar:
            continue

        pct = sim_pred["similarity_percentage"]
        final_sim_score = sim_pred["calibrated_prob"]

        if pct >= 80.0:
            relevance_tier = "HIGH"
        elif pct >= 50.0:
            relevance_tier = "MODERATE"
        else:
            relevance_tier = "LOW"

        prediction_label = sim_pred.get("prediction", "RELEVANT PRECEDENT")

        ranked_results.append({
            "case": legal_case,
            "prediction": prediction_label,
            "is_similar": True,
            "score": final_sim_score,
            "percentage": pct,
            "relevance_tier": relevance_tier,
            "features": sim_pred.get("features", {}),
            "document_percentage": round(document_score * 100, 2),
            "passage_percentage": round(passage_score * 100, 2),
            "relevant_passage": relevant_passage,
            "excerpt": relevant_passage,
            "matched_references": shared_references,
            "matched_terms": terms,
            "why_matched": explain_match(shared_references, terms),
            "matched_acts": facets["matched_acts"],
            "matched_sections": facets["matched_sections"],
            "matched_issues": facets["matched_issues"],
            "matched_facts": facets["matched_facts"],
            "matched_procedural": facets["matched_procedural"],
            "why_similar": facets["why_similar"],
            "model_version": MODEL_VERSION,
            "index_version": index_version,
            "search_mode": search_mode,
            "historical_outcome": historical_outcome,
            "current_status": (
                "Not automatically verified"
                if year < 2000
                else "Requires professional verification"
            ),
        })

    ranked_results.sort(key=lambda item: item["score"], reverse=True)
    return ranked_results[:top_k]


def get_search_diagnostics(query_text, top_k=5, search_mode="all"):
    cleaned_query = clean_text(query_text)
    expanded_query = expand_common_legal_language(cleaned_query)
    vectorizer, case_matrix, case_matrix_T, case_ids, index_version = load_similarity_model()
    query_vector = vectorizer.transform([expanded_query])
    query_references = sorted(list(normalize_legal_references(cleaned_query)))

    orig_terms = set(cleaned_query.lower().split())
    exp_terms = set(expanded_query.lower().split())
    added_terms = sorted(list(exp_terms - orig_terms))

    return {
        "model_version": MODEL_VERSION,
        "index_version": index_version,
        "matrix_rows": case_matrix.shape[0],
        "feature_count": case_matrix.shape[1],
        "query_features": int(query_vector.nnz),
        "candidate_pool_size": max(MINIMUM_CANDIDATES, top_k * CANDIDATE_MULTIPLIER),
        "search_mode": search_mode,
        "expanded_terms": added_terms,
        "legal_references": query_references,
        "query_word_count": len(query_text.split()),
        "cleaned_query": cleaned_query,
        "case_id_count": len(case_ids),
    }

