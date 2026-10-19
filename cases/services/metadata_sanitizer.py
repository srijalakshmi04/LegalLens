"""
Metadata Sanitizer for Legal Lens Precedents.
Ensures that party names, acts, sections, citations, and bench information
are completely clean, authoritative, and free of leaked judgment body text.
Provides graceful fallback to 'Not available' when genuinely unavailable.
"""

import re
import datetime
from cases.services.outcome_predictor import extract_outcome_label

KNOWN_ACT_PATTERNS = [
    ("Constitution of India", r"\b(?:constitution\s+of\s+india)\b"),
    ("Indian Penal Code, 1860", r"\b(?:indian\s+penal\s+code|ipc)\b"),
    ("Code of Criminal Procedure, 1973", r"\b(?:code\s+of\s+criminal\s+procedure|crpc|cr\.p\.c)\b"),
    ("Code of Civil Procedure, 1908", r"\b(?:code\s+of\s+civil\s+procedure|cpc|c\.p\.c)\b"),
    ("Workmen's Compensation Act, 1923", r"\b(?:workmen[\'s\s]+compensation\s+act|employee[\'s\s]+compensation\s+act)\b"),
    ("Insolvency and Bankruptcy Code, 2016", r"\b(?:insolvency\s+and\s+bankruptcy\s+code|ibc)\b"),
    ("Arbitration and Conciliation Act, 1996", r"\b(?:arbitration\s+and\s+conciliation\s+act|arbitration\s+act)\b"),
    ("Companies Act", r"\b(?:companies\s+act(?:\s*,\s*(?:1956|2013))?)\b"),
    ("Income Tax Act, 1961", r"\b(?:income\s+tax\s+act)\b"),
    ("Indian Evidence Act, 1872", r"\b(?:indian\s+evidence\s+act|evidence\s+act)\b"),
    ("Motor Vehicles Act, 1988", r"\b(?:motor\s+vehicles\s+act|mact)\b"),
    ("Specific Relief Act, 1963", r"\b(?:specific\s+relief\s+act)\b"),
    ("Negotiable Instruments Act, 1881", r"\b(?:negotiable\s+instruments\s+act|ni\s+act)\b"),
    ("Transfer of Property Act, 1882", r"\b(?:transfer\s+of\s+property\s+act)\b"),
    ("Land Acquisition Act, 1894", r"\b(?:land\s+acquisition\s+act)\b"),
    ("Consumer Protection Act", r"\b(?:consumer\s+protection\s+act)\b"),
    ("Prevention of Corruption Act, 1988", r"\b(?:prevention\s+of\s+corruption\s+act)\b"),
    ("Limitation Act, 1963", r"\b(?:limitation\s+act)\b"),
    ("NDPS Act, 1985", r"\b(?:ndps\s+act|narcotic\s+drugs\s+and\s+psychotropic\s+substances\s+act)\b"),
    ("Air Force Act, 1950", r"\b(?:air\s+force\s+act)\b"),
    ("Army Act, 1950", r"\b(?:army\s+act)\b"),
]

BAD_PARTY_NARRATIVE = [
    "judgment", "order", "appeal", "leave granted", "slp", "section", "versus", " v ", " vs ",
    "writ", "in the supreme court", "dated", "learned counsel", "appellant has", "respondent has",
    "interlocutory", "civil", "criminal", "coram", "hon'ble", "submits that", "held that",
    "paragraph", "court", "matter", "accused", "petitioners(s)", "appellants(s)",
]


def sanitize_party_name(name_str):
    """Validate whether party string is a genuine name or leaked judgment narrative."""
    if not name_str or len(name_str.strip()) < 2 or len(name_str.strip()) > 90:
        return None
    s = name_str.strip()
    s_lower = s.lower()
    if any(m in s_lower for m in BAD_PARTY_NARRATIVE):
        return None
    if s[0].islower():
        return None
    if re.search(r"\b(?:is|was|are|were|that|having|being|passed|stated|appellants|respondents)\b", s_lower):
        return None
    # Strip procedural tags
    s = re.sub(r"^(?:Petitioner|Appellant|Respondent|Accused)\s*[:\-\s]+", "", s, flags=re.I).strip()
    return s if len(s) >= 2 else None


def extract_parties(title, petitioner, respondent):
    """Extract clean petitioner and respondent, falling back to case title if DB field is corrupted."""
    clean_p = sanitize_party_name(petitioner)
    clean_r = sanitize_party_name(respondent)

    if (not clean_p or not clean_r) and title:
        m = re.split(r"\s+(?:vs\.?|v\.?|versus)\s+", title, maxsplit=1, flags=re.I)
        if len(m) == 2:
            p_cand = m[0].strip()
            r_cand = re.split(r"\s+on\s+\d{1,2}\s+[A-Za-z]+,?\s+\d{4}", m[1], flags=re.I)[0].strip()
            r_cand = re.split(r"\s+dated\s+", r_cand, flags=re.I)[0].strip()
            r_cand = re.split(r"\s+on\s+\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{4}", r_cand, flags=re.I)[0].strip()

            if not clean_p:
                clean_p = sanitize_party_name(p_cand)
            if not clean_r:
                clean_r = sanitize_party_name(r_cand)

    return clean_p or "Not available", clean_r or "Not available"


def sanitize_acts(acts_str, text_snippet=""):
    """Extract valid statutory Acts, preventing judgment narrative leak."""
    bad_narrative = [
        "appellant", "respondent", "judgment", "learned", "appeal", "order", "court",
        "tribunal", "dated", "compensation", "passed", "held", "filed", "allowed",
        "dismissed", "section", "brief", "giving rise", "under:-", "http", "kanoon",
        "nutshell", "deceased", "accused", "plaintiff", "defendant",
    ]
    clean_acts = []

    if acts_str:
        pre_cleaned = re.split(r"[-–—]\s*(?:Ss?|Scope|Section|Art)", acts_str)[0].strip()
        candidates = [c.strip() for c in pre_cleaned.split(",") if c.strip()]
        for c in candidates:
            if re.search(r"\b(Act|Code|Constitution|Rules|Regulation)\b", c, re.I):
                c_clean = re.sub(r"^[,\.\s\-\:]+", "", c).strip()
                c_clean = re.sub(r"[,\.\s\-\:]+$", "", c_clean).strip()
                is_bad = any(w in c_clean.lower() for w in bad_narrative) or len(c_clean) > 75 or c_clean[0].islower()
                if not is_bad and len(c_clean) >= 4:
                    if c_clean not in clean_acts:
                        clean_acts.append(c_clean)

    if not clean_acts and text_snippet:
        for formal_name, pat in KNOWN_ACT_PATTERNS:
            if re.search(pat, text_snippet, re.IGNORECASE):
                if formal_name not in clean_acts:
                    clean_acts.append(formal_name)
                    if len(clean_acts) >= 3:
                        break

    return ", ".join(clean_acts[:3]) if clean_acts else "Not available"


def sanitize_sections(sec_str):
    """Sanitize and deduplicate legal sections and statutory articles."""
    if not sec_str:
        return "Not available"
    matches = re.finditer(
        r"\b(Section|Sec\.|s\.|Order|Article|Art\.)\s+([0-9]+[A-Za-z]*(?:\s*\([0-9a-zA-Z]+\))*|[IVXLCDM]+)\b",
        sec_str,
        re.I,
    )
    cleaned = []
    seen = set()
    for m in matches:
        prefix_raw = m.group(1).lower()
        if "sec" in prefix_raw or prefix_raw == "s.":
            prefix = "Section"
        elif "art" in prefix_raw:
            prefix = "Article"
        elif "order" in prefix_raw:
            prefix = "Order"
        else:
            prefix = "Section"

        num = m.group(2).strip()
        if num.lower() in ("did", "the", "a", "an", "is", "or", "and", "i", "v"):
            # Skip invalid one-word filler words
            if num.lower() in ("did", "the", "a", "an", "is", "or", "and"):
                continue
        formatted = f"{prefix} {num}"
        if formatted.lower() not in seen:
            seen.add(formatted.lower())
            cleaned.append(formatted)

    return ", ".join(cleaned[:8]) if cleaned else "Not available"


def sanitize_bench(bench_str):
    """Ensure bench Coram is clean judge names without sentence narrative."""
    if not bench_str or len(bench_str.strip()) < 3:
        return "Not available"
    b = bench_str.strip()
    if len(b) > 120 or any(w in b.lower() for w in ["judgment", "order", "appeal", "dated", "filed", "vs."]):
        return "Not available"
    return b


def sanitize_citations(citation_str):
    """Ensure citation is clean and valid."""
    if not citation_str or len(citation_str.strip()) < 4:
        return "Not available"
    c = citation_str.strip()
    if len(c) > 100 or any(w in c.lower() for w in ["judgment", "order", "appellant", "respondent"]):
        return "Not available"
    c = re.sub(r"[,\.\s\-\:]+$", "", c).strip()
    return c if len(c) >= 4 else "Not available"


def get_sanitized_case_metadata(legal_case):
    """
    Returns an authoritative, sanitized metadata dictionary for a LegalCase model instance.
    No judgment text leaks into metadata fields.
    Falls back gracefully to 'Not available' when data is genuinely absent.
    """
    text_snippet = (legal_case.cleaned_text or legal_case.raw_text or "")[:5000]

    petitioner, respondent = extract_parties(
        legal_case.title, legal_case.petitioner, legal_case.respondent
    )

    acts = sanitize_acts(legal_case.acts, text_snippet)
    sections = sanitize_sections(legal_case.legal_sections)
    bench = sanitize_bench(legal_case.bench)
    citations = sanitize_citations(legal_case.citations)

    # Judgment Date / Year
    if legal_case.judgment_date:
        judgment_date_str = legal_case.judgment_date.strftime("%d %B %Y")
    elif legal_case.judgment_year:
        judgment_date_str = str(legal_case.judgment_year)
    else:
        judgment_date_str = "Not available"

    judgment_year_str = str(legal_case.judgment_year) if legal_case.judgment_year else "Not available"

    # Judicial disposition (historical outcome)
    raw_outcome = extract_outcome_label(text_snippet)
    historical_outcome = raw_outcome if raw_outcome else "Not available"

    return {
        "petitioner": petitioner,
        "respondent": respondent,
        "acts": acts,
        "legal_sections": sections,
        "bench": bench,
        "citations": citations,
        "judgment_date": judgment_date_str,
        "judgment_year": judgment_year_str,
        "historical_outcome": historical_outcome,
    }
