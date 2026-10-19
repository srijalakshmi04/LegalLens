import re

import numpy as np
from sklearn.metrics.pairwise import linear_kernel

from cases.services.preprocessing import clean_text


COMMON_LANGUAGE_EXPANSIONS = {
    "upi": "unauthorized financial transaction digital payment fraud online deception cyber financial offence banking",
    "stolen": "misappropriation unauthorized taking theft property loss financial loss",
    "stole": "misappropriation unauthorized taking theft property loss financial loss",
    "money": "financial funds monetary deposit misappropriation transaction",
    "cyber": "cyber offence digital transaction online fraud unauthorized access",
    "phishing": "digital deception unauthorized banking transaction financial fraud",
    "online": "digital transaction online portal cyber service",
    "bank": "banking transaction financial institution deposit",
    "detained": "detention preventive detention custody personal liberty",
    "detain": "detention preventive detention custody personal liberty",
    "arrested": "arrest detention custody bail criminal procedure",
    "police": "police investigation arrest custody criminal procedure",
    "fired": "termination dismissal employment service law",
    "job": "employment service labour termination",
    "property": "property ownership possession title transfer",
    "land": "land property possession acquisition title",
    "rent": "tenancy landlord tenant lease eviction",
    "divorce": "matrimonial marriage divorce maintenance custody",
    "maintenance": "maintenance alimony matrimonial support",
    "cheated": "cheating fraud deception breach of trust",
    "fraud": "fraud cheating deception misrepresentation",
    "loan": "loan debt recovery interest banking",
    "tax": "taxation assessment revenue income tax",
}

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def expand_common_legal_language(text):
    """Add legal equivalents without removing the user's original words."""
    cleaned = clean_text(text)
    lowered = cleaned.lower()
    expansions = []

    for common_word, legal_terms in COMMON_LANGUAGE_EXPANSIONS.items():
        if re.search(rf"\b{re.escape(common_word)}\b", lowered):
            expansions.append(legal_terms)

    return f"{cleaned} {' '.join(expansions)}".strip()


def significant_tokens(text):
    """Return useful lowercase words for passage preselection and explainability."""
    ignored = {
        "about", "above", "after", "again", "against", "all", "also", "and",
        "any", "are", "aren't", "because", "been", "before", "being", "below",
        "between", "both", "but", "can", "cannot", "case", "could", "couldn't",
        "court", "did", "didn't", "does", "doesn't", "doing", "done", "down",
        "during", "each", "etc", "even", "few", "for", "from", "further",
        "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "her",
        "here", "hers", "herself", "him", "himself", "his", "how", "into",
        "its", "itself", "just", "made", "make", "more", "most", "mustn't",
        "myself", "not", "off", "once", "one", "only", "our", "ours",
        "ourselves", "out", "over", "own", "per", "person", "said", "same",
        "she", "should", "shouldn't", "some", "someone", "state", "such",
        "than", "that", "the", "their", "theirs", "them", "themselves", "then",
        "there", "these", "they", "this", "those", "through", "too", "under",
        "until", "upon", "very", "was", "wasn't", "well", "were", "weren't",
        "what", "when", "where", "which", "while", "who", "whom", "why",
        "with", "won't", "would", "wouldn't", "you", "your", "yours", "yourself",
    }
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(text or "")
        if token.lower() not in ignored and len(token) >= 3
    }


def split_into_passages(text, sentences_per_passage=3):
    """Split a judgment into short overlapping, readable passages."""
    cleaned = clean_text(text)
    if not cleaned:
        return []

    sentences = [
        sentence.strip()
        for sentence in SENTENCE_BOUNDARY.split(cleaned)
        if len(sentence.strip()) >= 35
    ]
    if not sentences:
        return [cleaned[:1200]]

    passages = []
    for index in range(len(sentences)):
        passage = " ".join(
            sentences[index:index + sentences_per_passage]
        ).strip()
        if len(passage) >= 80:
            passages.append(passage[:1600])
    return passages


def preselect_passages(text, query_text, maximum=40):
    """Use word overlap to avoid vectorizing every judgment sentence."""
    query_tokens = significant_tokens(query_text)
    passages = split_into_passages(text)
    if len(passages) <= maximum:
        return passages

    ranked = []
    for position, passage in enumerate(passages):
        overlap = len(query_tokens & significant_tokens(passage))
        ranked.append((overlap, -position, passage))
    ranked.sort(reverse=True)
    return [item[2] for item in ranked[:maximum]]


def find_best_passage(text, query_text, vectorizer, query_vector):
    """Return the passage with the highest query-level TF-IDF score."""
    passages = preselect_passages(text, query_text)
    if not passages:
        return "", 0.0

    passage_matrix = vectorizer.transform(passages)
    scores = linear_kernel(query_vector, passage_matrix).ravel()
    if not len(scores):
        return passages[0], 0.0

    best_index = int(np.argmax(scores))
    return passages[best_index], float(scores[best_index])


def jaccard_overlap(left, right):
    """Return a normalized overlap value between zero and one."""
    left = set(left)
    right = set(right)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def matched_terms(query_text, passage, maximum=8):
    """Return readable words shared by the query and selected passage."""
    matches = sorted(
        significant_tokens(query_text) & significant_tokens(passage),
        key=lambda value: (-len(value), value),
    )
    return matches[:maximum]


def explain_match(matched_references, terms):
    """Create an evidence-based explanation grounded strictly in actual retrieval evidence."""
    parts = []
    if matched_references:
        parts.append(
            "Matched statutory provisions: "
            + ", ".join(matched_references)
            + "."
        )
    if terms:
        parts.append("Shared legal concepts: " + ", ".join(terms) + ".")
    if not parts:
        return (
            "Selected based on document-level TF-IDF similarity and passage "
            "textual alignment."
        )
    return " ".join(parts)
