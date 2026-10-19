import re
import logging
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
from django.conf import settings

from cases.models import LegalCase
from cases.services.passage_nlp import (
    expand_common_legal_language,
    find_best_passage,
)
from cases.services.preprocessing import (
    clean_text,
    normalize_legal_references,
)
from cases.services.similarity import (
    load_similarity_model,
    extract_case_similarity_facets,
)

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "semantic_cosine",
    "tfidf_cosine",
    "bm25_score",
    "issue_similarity",
    "act_similarity",
    "section_similarity",
    "fact_similarity",
    "procedure_similarity",
    "passage_similarity",
    "title_similarity",
    "max_chunk_similarity",
    "mean_chunk_similarity",
    "top_chunk_similarity",
]

LEGACY_FEATURE_NAMES = [
    "semantic_similarity",
    "tfidf_similarity",
    "legal_issue_similarity",
    "act_similarity",
    "section_similarity",
    "factual_similarity",
    "procedural_similarity",
    "passage_similarity",
]

LEGAL_ISSUE_PATTERNS = [
    ("Quashing of Criminal Proceedings (Sec 482)", r"\b(quash|quashing|482|inherent powers|compromise|private dispute|settlement)\b"),
    ("Default Bail (Sec 167(2))", r"\b(default bail|167\(2\)|60 days|90 days|indefeasible right|charge sheet)\b"),
    ("Anticipatory Bail (Sec 438)", r"\b(anticipatory bail|438|apprehension of arrest)\b"),
    ("Regular Bail (Sec 439)", r"\b(regular bail|439|grant of bail|cancellation of bail)\b"),
    ("Personal Liberty & Unlawful Arrest (Art 21 / Sec 41)", r"\b(unlawful arrest|preventive detention|article 21|section 41|habeas corpus|custodial)\b"),
    ("Land Acquisition & Market Value Compensation", r"\b(land acquisition|compensation|market value|section 4|section 23|solatium)\b"),
    ("Specific Performance of Contract (Sec 16)", r"\b(specific performance|readiness and willingness|agreement to sell|section 16|16\(c\))\b"),
    ("Rejection of Plaint (Order 7 Rule 11)", r"\b(rejection of plaint|order 7 rule 11|cause of action|barred by law|order vii rule 11)\b"),
    ("Dishonour of Cheque (Sec 138 NI Act)", r"\b(cheque bounce|dishonour of cheque|section 138|negotiable instruments|statutory notice)\b"),
    ("Income Tax Assessment Reopening (Sec 147/148)", r"\b(reopening|re-assessment|section 147|section 148|escapement of income|reason to believe)\b"),
    ("Motor Accident Claims Compensation (MACT)", r"\b(motor accident|mact|multiplier|future prospects|permanent disability|tribunal compensation)\b"),
    ("Circumstantial Evidence (Panchsheel)", r"\b(circumstantial evidence|chain of circumstances|last seen|panchsheel)\b"),
    ("Dying Declaration (Sec 32)", r"\b(dying declaration|section 32|truthfulness of dying declaration)\b"),
    ("Arbitration Award Setting Aside (Sec 34/37)", r"\b(arbitration|section 34|setting aside award|patent illegality|public policy|section 37)\b"),
    ("Right to Equality & Arbitrary Tender (Art 14)", r"\b(article 14|arbitrary|reasonable classification|public tender|government contract)\b"),
    ("Principles of Natural Justice & Fair Hearing", r"\b(natural justice|audi alteram|fair hearing|inquiry report|opportunity of hearing)\b"),
    ("Insolvency & Moratorium (IBC Sec 14/9/7)", r"\b(insolvency|ibc|moratorium|section 14|section 9|section 7|nclt|operational creditor)\b"),
    ("Consumer Protection & Medical Deficiency", r"\b(consumer dispute|medical negligence|deficiency of service|consumer protection)\b"),
    ("Environmental Law & Precautionary Principle", r"\b(environmental clearance|national green tribunal|precautionary principle|polluter pays|ngt)\b"),
    ("Trademark Infringement & Deceptive Similarity", r"\b(trademark|infringement|passing off|deceptively similar|intellectual property)\b"),
    ("Service Seniority & Promotion Rules", r"\b(seniority|promotion|quota rota|service rules|cadre|retrospective seniority)\b"),
    ("NDPS Mandatory Search Procedure (Sec 50)", r"\b(ndps|narcotic|section 50|personal search|gazetted officer|commercial quantity|section 37)\b"),
    ("Hindu Marriage Cruelty & Divorce (Sec 13)", r"\b(hindu marriage|divorce|cruelty|desertion|section 13|maintenance|restitution)\b"),
]


LEGAL_ACT_PATTERNS = [
    ("Penal Code", r"\b(indian penal code|ipc|penal code)\b"),
    ("CrPC", r"\b(code of criminal procedure|crpc|cr\.p\.c|criminal procedure code)\b"),
    ("CPC", r"\b(code of civil procedure|cpc|c\.p\.c|civil procedure code)\b"),
    ("Evidence Act", r"\b(evidence act|indian evidence act)\b"),
    ("Constitution", r"\b(constitution of india|constitution|article \d+)\b"),
    ("Land Acquisition", r"\b(land acquisition act|land acquisition)\b"),
    ("Negotiable Instruments", r"\b(negotiable instruments act|ni act|section 138)\b"),
    ("NDPS", r"\b(ndps act|narcotic drugs|psychotropic substances)\b"),
    ("Income Tax", r"\b(income tax act|income-tax)\b"),
    ("Motor Vehicles", r"\b(motor vehicles act|mact)\b"),
    ("Arbitration", r"\b(arbitration and conciliation act|arbitration act)\b"),
    ("Specific Relief", r"\b(specific relief act)\b"),
    ("Prevention of Corruption", r"\b(prevention of corruption act)\b"),
    ("Companies Act", r"\b(companies act)\b"),
    ("Consumer Protection", r"\b(consumer protection act)\b"),
]

LEGAL_SECTION_PATTERNS = r"\b(?:section|sec|s\.|article|art|order|rule)\s*([0-9]+[a-z]?)\b"

PROCEDURAL_POSTURE_PATTERNS = [
    ("Criminal Appeal", r"\bcriminal appeal\b"),
    ("Civil Appeal", r"\bcivil appeal\b"),
    ("Special Leave Petition", r"\b(?:special leave petition|slp)\b"),
    ("Writ Petition", r"\bwrit petition\b"),
    ("Bail Application", r"\b(?:bail application|regular bail|anticipatory bail|default bail)\b"),
    ("Review Petition", r"\breview petition\b"),
    ("Contempt Petition", r"\bcontempt petition\b"),
]


def _extract_statutes_for_classifier(text):
    tl = (text or "").lower()
    acts = {name for name, pat in LEGAL_ACT_PATTERNS if re.search(pat, tl)}
    secs = set(re.findall(LEGAL_SECTION_PATTERNS, tl))
    return acts, secs


def _extract_postures_for_classifier(text):
    tl = (text or "").lower()
    return {name for name, pat in PROCEDURAL_POSTURE_PATTERNS if re.search(pat, tl)}


@lru_cache(maxsize=1)
def load_case_similarity_classifier():
    """Load the trained and calibrated binary case-similarity classifier artifact."""
    v2_path = Path(settings.BASE_DIR) / "ml_models" / "similarity_classifier_v2.joblib"
    if v2_path.exists():
        return joblib.load(v2_path)

    v1_path = Path(settings.BASE_DIR) / "ml_models" / "similarity_classifier_v1.joblib"
    if v1_path.exists():
        return joblib.load(v1_path)

    # Lazy build if neither is present
    from scratch.build_similarity_model import artifact
    return artifact


@lru_cache(maxsize=32)
def _get_query_classifier_context(cleaned_q):
    """Precompute and cache query representations across candidate evaluations."""
    artifact = load_case_similarity_classifier()
    tfidf_v2 = artifact.get("tfidf")
    lsa = artifact.get("lsa")
    expanded_q = expand_common_legal_language(cleaned_q)
    q_vec_v2 = tfidf_v2.transform([expanded_q]) if tfidf_v2 is not None else None
    q_sem = lsa.transform(q_vec_v2)[0] if (lsa is not None and q_vec_v2 is not None) else None
    q_tokens = cleaned_q.lower().split()
    q_acts, q_secs = _extract_statutes_for_classifier(cleaned_q)
    q_post = _extract_postures_for_classifier(cleaned_q)
    q_title_words = [w for w in q_tokens[:10] if len(w) > 3]

    return {
        "expanded_q": expanded_q,
        "q_vec_v2": q_vec_v2,
        "q_sem": q_sem,
        "q_tokens": q_tokens,
        "q_acts": q_acts,
        "q_secs": q_secs,
        "q_post": q_post,
        "q_title_words": q_title_words,
    }


def extract_case_pair_similarity_features(
    query_text,
    legal_case,
    stage1_doc_score=None,
    max_stage1_doc_score=None
):
    """
    Extract the pairwise similarity features between a current case query
    and a candidate Supreme Court judgment. Matches the training and calibration
    distributions of similarity_classifier_v2 to eliminate score saturation.
    """
    artifact = load_case_similarity_classifier()
    vectorizer, case_matrix, case_matrix_T, case_ids, _ = load_similarity_model()
    cleaned_q = clean_text(query_text)
    q_ctx = _get_query_classifier_context(cleaned_q)

    expanded_q = q_ctx["expanded_q"]
    q_vec = vectorizer.transform([expanded_q])

    cid = legal_case.id
    raw_head = legal_case.raw_text or legal_case.cleaned_text or ""
    best_passage, passage_sim = find_best_passage(raw_head, expanded_q, vectorizer, q_vec)
    facets = extract_case_similarity_facets(cleaned_q, legal_case, best_passage)

    if artifact.get("version") == "2.0.0":
        tfidf_v2 = artifact.get("tfidf", vectorizer)
        lsa = artifact.get("lsa")
        q_sem = q_ctx["q_sem"]
        q_vec_v2 = q_ctx["q_vec_v2"]

        c_raw = clean_text(legal_case.cleaned_text or legal_case.raw_text or "")
        c_tfidf = tfidf_v2.transform([c_raw[:10000]])
        c_sem = lsa.transform(c_tfidf)[0] if lsa is not None else None

        # 1. Semantic Cosine (Dense LSA)
        if q_sem is not None and c_sem is not None:
            sem_cos = float(np.dot(q_sem, c_sem))
        else:
            sem_cos = 0.50

        # 2. TF-IDF Cosine
        tfidf_cos = float(q_vec_v2.dot(c_tfidf.T).toarray()[0][0]) if q_vec_v2 is not None else float(passage_sim)

        # 3. Normalized Stage 1 / BM25 Score (bounded in [0.85, 0.98], matching training pool distribution)
        if stage1_doc_score is not None and max_stage1_doc_score and max_stage1_doc_score > 0:
            rel_stage1 = min(1.0, max(0.0, float(stage1_doc_score) / float(max_stage1_doc_score)))
            bm25_norm = 0.85 + 0.13 * rel_stage1
        else:
            rel_stage1 = min(1.0, max(0.0, tfidf_cos / 0.20))
            bm25_norm = 0.85 + 0.13 * rel_stage1

        # 4. Statutory Overlap
        c_acts, c_secs = _extract_statutes_for_classifier(c_raw[:4000])
        q_acts = q_ctx["q_acts"]
        q_secs = q_ctx["q_secs"]
        act_overlap = len(q_acts & c_acts) / max(1, len(q_acts | c_acts)) if (q_acts or c_acts) else 0.0
        sec_overlap = len(q_secs & c_secs) / max(1, len(q_secs | c_secs)) if (q_secs or c_secs) else 0.0
        issue_sim = 0.6 * act_overlap + 0.4 * sec_overlap

        # 5. Token Jaccard Fact Similarity (First 100 query tokens vs First 200 case tokens)
        words = c_raw.split()
        c_tokens = words[:200]
        q_set = set(q_ctx["q_tokens"][:100])
        c_set = set(c_tokens)
        fact_sim = len(q_set & c_set) / max(1, len(q_set | c_set))

        # 6. Procedural Posture Similarity
        c_post = _extract_postures_for_classifier(c_raw[:2000])
        q_post = q_ctx["q_post"]
        proc_sim = 1.0 if (q_post & c_post) else (0.5 if (not q_post and not c_post) else 0.0)

        # 7. Multi-Chunk Semantic Passage Similarities
        chunks = [' '.join(words[j:j+350]) for j in range(0, min(len(words), 350*8), 350)]
        if not chunks:
            chunks = [c_raw[:1000]]
        c_chunks_tfidf = tfidf_v2.transform(chunks)
        c_chunks_sem = lsa.transform(c_chunks_tfidf)
        chunk_sims = np.dot(c_chunks_sem, q_sem) if q_sem is not None else np.array([passage_sim])
        max_chunk_sim = float(np.max(chunk_sims))
        mean_chunk_sim = float(np.mean(chunk_sims))
        top_chunk_sim = float(np.mean(np.sort(chunk_sims)[-min(3, len(chunk_sims)):]))
        passage_sim_val = max_chunk_sim

        # 8. Case Title Alignment
        c_title = (legal_case.title or "").lower()
        q_title_words = q_ctx["q_title_words"]
        t_matches = sum(1 for w in q_title_words if w in c_title)
        title_sim = min(1.0, t_matches / max(1, len(q_title_words)))

        feats = np.array([
            sem_cos,
            tfidf_cos,
            bm25_norm,
            issue_sim,
            act_overlap,
            sec_overlap,
            fact_sim,
            proc_sim,
            passage_sim_val,
            title_sim,
            max_chunk_sim,
            mean_chunk_sim,
            top_chunk_sim,
        ], dtype=np.float64)
        active_names = FEATURE_NAMES
    else:
        # Legacy fallback
        q_lower = cleaned_q.lower()
        c_lower = raw_head[:4000].lower()
        q_matched_issues = [name for name, pat in LEGAL_ISSUE_PATTERNS if re.search(pat, q_lower)]
        c_matched_issues = [name for name, pat in LEGAL_ISSUE_PATTERNS if re.search(pat, c_lower)]
        issue_sim = float(len(set(q_matched_issues) & set(c_matched_issues))) / float(len(q_matched_issues)) if q_matched_issues else 0.0

        q_refs = normalize_legal_references(cleaned_q)
        c_refs = normalize_legal_references(" ".join([legal_case.acts or "", legal_case.legal_sections or "", best_passage]))
        q_acts = {r for r in q_refs if not (r.startswith("article_") or r.startswith("section_"))}
        c_acts = {r for r in c_refs if not (r.startswith("article_") or r.startswith("section_"))}
        act_sim = len(q_acts & c_acts) / max(1, len(q_acts | c_acts)) if (q_acts or c_acts) else 0.0

        q_secs = {r for r in q_refs if r.startswith("section_") or r.startswith("article_")}
        c_secs = {r for r in c_refs if r.startswith("section_") or r.startswith("article_")}
        sec_sim = len(q_secs & c_secs) / max(1, len(q_secs | c_secs)) if (q_secs or c_secs) else 0.0

        q_words = set(cleaned_q.lower().split())
        c_words = set(raw_head[:2000].lower().split())
        fact_sim = len(q_words & c_words) / max(1, len(q_words | c_words))
        proc_sim = 0.5
        sem_sim = 0.40 * float(passage_sim) + 0.35 * float(passage_sim) + 0.15 * issue_sim + 0.10 * act_sim

        feats = np.array([
            sem_sim,
            float(passage_sim),
            issue_sim,
            act_sim,
            sec_sim,
            fact_sim,
            proc_sim,
            float(passage_sim),
        ], dtype=np.float64)
        active_names = LEGACY_FEATURE_NAMES

    return feats, facets, best_passage, active_names


def predict_case_pair_similarity(
    query_text,
    legal_case,
    stage1_doc_score=None,
    max_stage1_doc_score=None
):
    """
    Computes calibrated pairwise relevance score between a query and a candidate Supreme Court case.
    Returns calibrated probability (0-100%) and classification label.
    """
    artifact = load_case_similarity_classifier()
    decision_threshold = artifact.get("threshold", 0.50)

    feats, facets, best_passage, active_names = extract_case_pair_similarity_features(
        query_text,
        legal_case,
        stage1_doc_score=stage1_doc_score,
        max_stage1_doc_score=max_stage1_doc_score,
    )

    if artifact.get("version") == "2.0.0":
        scaler = artifact.get("scaler")
        clf = artifact.get("classifier")
        calibrator = artifact.get("calibrator")

        feats_scaled = scaler.transform([feats]) if scaler is not None else [feats]
        score = clf.decision_function(feats_scaled).reshape(-1, 1)
        calibrated_prob = float(calibrator.predict_proba(score)[0, 1]) if calibrator is not None else float(clf.predict_proba(feats_scaled)[0, 1])
    else:
        base_clf = artifact["base_clf"]
        k_scale = artifact.get("k_scale", 1.0)
        logit_threshold = artifact.get("logit_threshold", 0.0)
        raw_logit = float(base_clf.decision_function([feats])[0])
        calibrated_prob = float(1.0 / (1.0 + np.exp(-k_scale * (raw_logit - logit_threshold))))

    similarity_pct = round(calibrated_prob * 100.0, 1)
    is_similar = bool(calibrated_prob >= decision_threshold)
    prediction_label = "RELEVANT PRECEDENT" if is_similar else "NOT RELEVANT"

    feature_dict = {name: round(float(val), 4) for name, val in zip(active_names, feats)}

    return {
        "prediction": prediction_label,
        "is_similar": is_similar,
        "similarity_percentage": similarity_pct,
        "calibrated_prob": calibrated_prob,
        "decision_threshold": round(decision_threshold * 100.0, 1),
        "features": feature_dict,
        "facets": facets,
        "best_passage": best_passage,
    }
