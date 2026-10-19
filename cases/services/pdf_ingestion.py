import re
import hashlib
import datetime
import logging
from io import BytesIO

logger = logging.getLogger(__name__)

GENERIC_PLACEHOLDERS = {
    "APPELLANT", "APPELLANTS", "APPELLANT(S)", "APPELLANTS(S)",
    "RESPONDENT", "RESPONDENTS", "RESPONDENT(S)", "RESPONDENTS(S)",
    "PETITIONER", "PETITIONERS", "PETITIONER(S)", "PETITIONERS(S)",
    "VERSUS", "VS", "VS.", "V.", "V",
    "IN THE SUPREME COURT OF INDIA",
    "CIVIL APPELLATE JURISDICTION", "CRIMINAL APPELLATE JURISDICTION",
    "JUDGMENT", "ORDER", "DATED", "SUPREME COURT OF INDIA",
    "OTHER", "OTHERS", "ANR.", "ANR", "ORS.", "ORS", "UNKNOWN PARTY",
}


def extract_pdf_text(file_obj_or_bytes):
    """
    Extract raw text, page count, and character count from a PDF document.
    Uses PyMuPDF (fitz) with fallback to pypdf.
    """
    if isinstance(file_obj_or_bytes, bytes):
        raw_bytes = file_obj_or_bytes
    elif hasattr(file_obj_or_bytes, "read"):
        raw_bytes = file_obj_or_bytes.read()
    else:
        with open(file_obj_or_bytes, "rb") as f:
            raw_bytes = f.read()

    text_pages = []
    page_count = 0

    # Try PyMuPDF (fitz)
    try:
        import fitz
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        page_count = len(doc)
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text_pages.append(page_text)
        doc.close()
    except Exception as e1:
        logger.warning(f"PyMuPDF extraction failed or unavailable: {e1}. Trying pypdf...")
        try:
            import pypdf
            reader = pypdf.PdfReader(BytesIO(raw_bytes))
            page_count = len(reader.pages)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text_pages.append(t)
        except Exception as e2:
            logger.error(f"pypdf extraction failed: {e2}")
            raise ValueError(f"Could not extract text from PDF document: {str(e2)}")

    full_text = "\n\n".join(text_pages).strip()
    char_count = len(full_text)
    word_count = len(full_text.split())

    requires_ocr = char_count < 100 or word_count < 15

    return {
        "raw_text": full_text,
        "page_count": page_count,
        "char_count": char_count,
        "word_count": word_count,
        "requires_ocr": requires_ocr,
        "file_bytes": raw_bytes,
    }


def calculate_file_checksum(raw_bytes):
    """Calculate SHA-256 checksum for PDF bytes."""
    return hashlib.sha256(raw_bytes).hexdigest()


def is_generic_placeholder(part_str):
    """Check if a string is a generic placeholder or procedural label."""
    if not part_str or len(part_str.strip()) < 3:
        return True
    cleaned = re.sub(r"[^\w\s]", "", part_str.upper()).strip()
    if cleaned in GENERIC_PLACEHOLDERS:
        return True
    base_cleaned = re.sub(r"S$", "", cleaned)
    return base_cleaned in GENERIC_PLACEHOLDERS


def extract_citation(text):
    """
    Extract Supreme Court citations with evidence snippet and numerical confidence.
    Supports INSC neutral, SCC, AIR, and SCR formats.
    """
    if not text:
        return {
            "value": "",
            "year_extracted": None,
            "confidence_score": 0.0,
            "confidence": "NOT_FOUND",
            "evidence": "No text available",
        }

    search_window = text[:4000]

    # 1. Neutral Citation (e.g., 2026 INSC 442)
    insc_match = re.search(r"\b(\d{4}\s+INSC\s+\d+)\b", search_window, re.IGNORECASE)
    if insc_match:
        cit = insc_match.group(1).upper()
        yr = int(cit.split()[0])
        snippet = _extract_snippet(search_window, insc_match.start(), insc_match.end())
        return {
            "value": cit,
            "year_extracted": yr,
            "confidence_score": 0.95,
            "confidence": "HIGH",
            "evidence": snippet,
        }

    # 2. SCC Citation (e.g., (2025) 4 SCC 123 or 2025 SCC OnLine SC 456)
    scc_match = re.search(r"\b(\(\d{4}\)\s*\d+\s+SCC\s+\d+|\d{4}\s+SCC\s+OnLine\s+SC\s+\d+)\b", search_window, re.IGNORECASE)
    if scc_match:
        cit = scc_match.group(1)
        yr_m = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", cit)
        yr = int(yr_m.group(1)) if yr_m else None
        snippet = _extract_snippet(search_window, scc_match.start(), scc_match.end())
        return {
            "value": cit,
            "year_extracted": yr,
            "confidence_score": 0.90,
            "confidence": "HIGH",
            "evidence": snippet,
        }

    # 3. AIR Citation (e.g., AIR 2024 SC 123)
    air_match = re.search(r"\b(AIR\s+\d{4}\s+SC\s+\d+)\b", search_window, re.IGNORECASE)
    if air_match:
        cit = air_match.group(1)
        yr_m = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", cit)
        yr = int(yr_m.group(1)) if yr_m else None
        snippet = _extract_snippet(search_window, air_match.start(), air_match.end())
        return {
            "value": cit,
            "year_extracted": yr,
            "confidence_score": 0.85,
            "confidence": "MEDIUM",
            "evidence": snippet,
        }

    # 4. SCR Citation (e.g., [2023] 2 S.C.R. 456)
    scr_match = re.search(r"\b(\[\d{4}\]\s*\d+\s+S\.C\.R\.\s*\d+)\b", search_window, re.IGNORECASE)
    if scr_match:
        cit = scr_match.group(1)
        yr_m = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", cit)
        yr = int(yr_m.group(1)) if yr_m else None
        snippet = _extract_snippet(search_window, scr_match.start(), scr_match.end())
        return {
            "value": cit,
            "year_extracted": yr,
            "confidence_score": 0.75,
            "confidence": "MEDIUM",
            "evidence": snippet,
        }

    return {
        "value": "",
        "year_extracted": None,
        "confidence_score": 0.0,
        "confidence": "NOT_FOUND",
        "evidence": "No citation matched in document header",
    }


def extract_judgment_date(text, citation_year=None):
    """
    Extract full judgment date and year with evidence snippet and numerical confidence.
    """
    if not text:
        return {
            "date": None,
            "year": citation_year,
            "confidence_score": 0.70 if citation_year else 0.0,
            "confidence": "MEDIUM" if citation_year else "NOT_FOUND",
            "evidence": f"Year derived from citation ({citation_year})" if citation_year else "No text available",
        }

    search_window = text[:5000]

    # Explicit prefix patterns (HIGH confidence: 0.95)
    prefix_patterns = [
        r"(?:DATE OF JUDGMENT|DATED|DECIDED ON|PRONOUNCED ON|JUDGMENT DATED)\s*[:\-\s]+\s*([0-9]{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s*,?\s*[0-9]{4})",
        r"(?:DATE OF JUDGMENT|DATED|DECIDED ON|PRONOUNCED ON|JUDGMENT DATED)\s*[:\-\s]+\s*([0-9]{1,2}[\/\.-][0-9]{1,2}[\/\.-][0-9]{4})",
    ]

    for pat in prefix_patterns:
        match = re.search(pat, search_window, re.IGNORECASE)
        if match:
            date_str = match.group(1).strip()
            parsed_date = _parse_date_string(date_str)
            if parsed_date:
                snippet = _extract_snippet(search_window, match.start(), match.end())
                return {
                    "date": parsed_date.isoformat(),
                    "year": parsed_date.year,
                    "confidence_score": 0.95,
                    "confidence": "HIGH",
                    "evidence": snippet,
                }

    # General date patterns (MEDIUM confidence: 0.80)
    general_patterns = [
        r"\b([0-9]{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+[0-9]{4})\b",
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+[0-9]{1,2}\s*,?\s*[0-9]{4})\b",
        r"\b([0-9]{1,2}[\/\.-][0-9]{1,2}[\/\.-][0-9]{4})\b",
    ]

    for pat in general_patterns:
        match = re.search(pat, search_window, re.IGNORECASE)
        if match:
            date_str = match.group(1).strip()
            parsed_date = _parse_date_string(date_str)
            if parsed_date:
                snippet = _extract_snippet(search_window, match.start(), match.end())
                return {
                    "date": parsed_date.isoformat(),
                    "year": parsed_date.year,
                    "confidence_score": 0.80,
                    "confidence": "MEDIUM",
                    "evidence": snippet,
                }

    # Year fallback from citation (MEDIUM confidence: 0.75)
    if citation_year and 1950 <= citation_year <= datetime.datetime.now().year + 1:
        return {
            "date": None,
            "year": citation_year,
            "confidence_score": 0.75,
            "confidence": "MEDIUM",
            "evidence": f"Derived from neutral citation ({citation_year})",
        }

    # Year fallback from text scan (LOW confidence: 0.50)
    year_match = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", search_window)
    if year_match:
        found_year = int(year_match.group(1))
        snippet = _extract_snippet(search_window, year_match.start(), year_match.end())
        return {
            "date": None,
            "year": found_year,
            "confidence_score": 0.50,
            "confidence": "LOW",
            "evidence": snippet,
        }

    return {
        "date": None,
        "year": None,
        "confidence_score": 0.0,
        "confidence": "NOT_FOUND",
        "evidence": "No judgment date or year detected in top 5,000 characters",
    }


def _parse_date_string(date_str):
    cleaned = re.sub(r"(st|nd|rd|th)", "", date_str, flags=re.IGNORECASE).strip()
    for fmt in ("%d %B %Y", "%B %d %Y", "%d %b %Y", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
        try:
            return datetime.datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def extract_case_title_and_parties(text, filename=""):
    """
    Multi-page party and case title extraction with strict generic placeholder rejection.
    Searches first 3 pages (up to 8,000 chars) for VERSUS / VS / v. blocks.
    Returns individual petitioner and respondent confidence scores and evidence snippets.
    """
    if not text:
        return {
            "title": "",
            "petitioner": "",
            "respondent": "",
            "confidence_score": 0.0,
            "confidence": "NOT_FOUND",
            "petitioner_confidence_score": 0.0,
            "petitioner_confidence": "NOT_FOUND",
            "respondent_confidence_score": 0.0,
            "respondent_confidence": "NOT_FOUND",
            "evidence": "No text available",
        }

    search_window = text[:8000]

    vs_matches = list(re.finditer(r"\b(VERSUS|VS\.|VS|\bv\.\b)\b", search_window, re.IGNORECASE))

    for m in vs_matches:
        vs_start = m.start()
        vs_end = m.end()

        pre_text = search_window[max(0, vs_start - 600):vs_start]
        post_text = search_window[vs_end:min(len(search_window), vs_end + 600)]

        pre_lines = [l.strip() for l in pre_text.splitlines() if l.strip()]
        post_lines = [l.strip() for l in post_text.splitlines() if l.strip()]

        raw_p = ""
        for line in reversed(pre_lines):
            if re.search(r"IN THE SUPREME COURT|APPEAL NO|SLP|CRIMINAL|CIVIL|JURISDICTION|REPORTABLE|NON-REPORTABLE|SECTION|IN THE HIGH COURT", line, re.I):
                continue
            cleaned_line = _clean_party_line(line)
            if cleaned_line and not is_generic_placeholder(cleaned_line):
                raw_p = cleaned_line
                break

        raw_r = ""
        for line in post_lines:
            if re.search(r"JUDGMENT|ORDER|DATED|WITH|CORAM|BEFORE|JUSTICE|ADVOCATE|DECREE", line, re.I):
                break
            cleaned_line = _clean_party_line(line)
            if cleaned_line and not is_generic_placeholder(cleaned_line):
                raw_r = cleaned_line
                break

        if raw_p and raw_r and not is_generic_placeholder(raw_p) and not is_generic_placeholder(raw_r):
            title = f"{raw_p} v. {raw_r}"
            snippet = _extract_snippet(search_window, max(0, vs_start - 100), min(len(search_window), vs_end + 100))
            return {
                "title": title[:300],
                "petitioner": raw_p[:200],
                "respondent": raw_r[:200],
                "confidence_score": 0.95,
                "confidence": "HIGH",
                "petitioner_confidence_score": 0.95,
                "petitioner_confidence": "HIGH",
                "respondent_confidence_score": 0.95,
                "respondent_confidence": "HIGH",
                "evidence": snippet,
            }

    # Fallback: filename check (if clean filename available)
    if filename and not filename.lower().startswith("document") and not filename.lower().startswith("judgment"):
        clean_name = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
        if (" V " in clean_name or " Vs " in clean_name or " Versus " in clean_name) and not is_generic_placeholder(clean_name):
            parts = re.split(r"\s+V\s+|\s+Vs\s+|\s+Versus\s+", clean_name, flags=re.I)
            p_part = parts[0].strip() if len(parts) > 0 else ""
            r_part = parts[1].strip() if len(parts) > 1 else ""
            if not is_generic_placeholder(p_part) and not is_generic_placeholder(r_part):
                return {
                    "title": f"{p_part} v. {r_part}"[:300],
                    "petitioner": p_part[:200],
                    "respondent": r_part[:200],
                    "confidence_score": 0.50,
                    "confidence": "LOW",
                    "petitioner_confidence_score": 0.50,
                    "petitioner_confidence": "LOW",
                    "respondent_confidence_score": 0.50,
                    "respondent_confidence": "LOW",
                    "evidence": f"Inferred from PDF source filename ({filename})",
                }

    return {
        "title": "",
        "petitioner": "",
        "respondent": "",
        "confidence_score": 0.0,
        "confidence": "NOT_FOUND",
        "petitioner_confidence_score": 0.0,
        "petitioner_confidence": "NOT_FOUND",
        "respondent_confidence_score": 0.0,
        "respondent_confidence": "NOT_FOUND",
        "evidence": "No valid party names or VERSUS structure matched in top 8,000 characters",
    }


def _clean_party_line(line):
    cleaned = re.sub(r"\s*[\.\s\-\:]*\s*(?:APPELLANT\(S\)|PETITIONER\(S\)|RESPONDENT\(S\)|APPELLANT|PETITIONER|RESPONDENT|ACCUSED|COMPLAINANT)\b.*$", "", line, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^[0-9\.\-\s]+", "", cleaned).strip()
    return cleaned


def extract_bench_judges(text):
    """
    Extract bench / judge names strictly from CORAM, BEFORE, or HON'BLE sections.
    Uses tight windowing to prevent paragraph or case description text from bleeding in.
    """
    if not text:
        return {
            "value": "",
            "confidence_score": 0.0,
            "confidence": "NOT_FOUND",
            "evidence": "No text available",
        }

    search_window = text[:5000]

    # Explicit CORAM / BEFORE heading match
    coram_match = re.search(r"(?:CORAM|BEFORE|PRESENT)\s*[:\-\s]+\s*([^\n\r]+(?:\n[^\n\r]+){0,2})", search_window, re.IGNORECASE)
    if coram_match:
        coram_block = coram_match.group(1).strip()
        judges = _extract_judge_names_from_block(coram_block)
        if judges:
            snippet = _extract_snippet(search_window, coram_match.start(), coram_match.end())
            return {
                "value": ", ".join(judges)[:250],
                "confidence_score": 0.95,
                "confidence": "HIGH",
                "evidence": snippet,
            }

    # Standalone HON'BLE MR. JUSTICE scan
    judges = re.findall(r"\b(HON'BLE\s+(?:MR\.|MS\.|DR\.)?\s*JUSTICE\s+[A-Z\.\s]{3,40})\b", search_window, re.IGNORECASE)
    if judges:
        unique_judges = list(dict.fromkeys([" ".join(j.split()) for j in judges if len(j.split()) <= 6]))[:3]
        if unique_judges:
            return {
                "value": ", ".join(unique_judges)[:250],
                "confidence_score": 0.80,
                "confidence": "MEDIUM",
                "evidence": f"Found judge references: {', '.join(unique_judges)}",
            }

    return {
        "value": "",
        "confidence_score": 0.0,
        "confidence": "NOT_FOUND",
        "evidence": "Could not confidently extract bench information from top 5,000 characters",
    }


def _extract_judge_names_from_block(block):
    lines = [l.strip() for l in block.splitlines() if l.strip()]
    valid_judges = []
    for line in lines[:3]:
        if re.search(r"JUDGMENT|ORDER|CIVIL|CRIMINAL|APPEAL|DATED|SLP|SECTION", line, re.I):
            break
        clean = re.sub(r"^(BEFORE|CORAM|PRESENT|HON'BLE|MR\.|MS\.|DR\.|JUSTICE|\:|\-|\s)+", "", line, flags=re.I).strip()
        if len(clean) > 3 and not is_generic_placeholder(clean):
            valid_judges.append(f"Justice {clean.title()}")
    return valid_judges


def _extract_snippet(text, start, end, max_length=150):
    """Helper to extract a short, readable evidence snippet around a match location."""
    s = max(0, start - 20)
    e = min(len(text), end + 50)
    snippet = " ".join(text[s:e].split())
    if len(snippet) > max_length:
        snippet = snippet[:max_length] + "..."
    return snippet


def extract_judgment_metadata(text, filename=""):
    """
    Comprehensive Supreme Court judgment metadata extraction with evidence snippets and confidence levels.
    """
    if not text:
        return {}

    # 1. Citations
    citation_info = extract_citation(text)

    # 2. Judgment Date & Year
    date_info = extract_judgment_date(text, citation_year=citation_info["year_extracted"])

    # 3. Case Title & Parties
    title_info = extract_case_title_and_parties(text, filename=filename)

    # 4. Bench / Judges
    bench_info = extract_bench_judges(text)

    # 5. Case Number
    search_window = text[:3500]
    case_number = ""
    case_num_confidence = "NOT_FOUND"
    case_num_score = 0.0
    case_num_evidence = "No case number matched"
    case_num_match = re.search(
        r"\b((?:CIVIL|CRIMINAL)\s+APPEAL\s+(?:NO\.|NOS\.)\s*[\d\s\-\/]+OF\s+\d{4}|SLP\s*\([C|CRL]\)\s*(?:NO\.|NOS\.)\s*[\d\s\-\/]+OF\s+\d{4}|WRIT\s+PETITION\s*\([C|CRL]\)\s*(?:NO\.|NOS\.)\s*[\d\s\-\/]+OF\s+\d{4})\b",
        search_window,
        re.IGNORECASE,
    )
    if case_num_match:
        case_number = " ".join(case_num_match.group(1).split())
        case_num_confidence = "HIGH"
        case_num_score = 0.95
        case_num_evidence = _extract_snippet(search_window, case_num_match.start(), case_num_match.end())

    critical_confidence_low = (
        title_info["confidence"] in ("LOW", "NOT_FOUND") or
        date_info["confidence"] in ("LOW", "NOT_FOUND")
    )

    return {
        # Flat values for backward compatibility
        "title": title_info["title"],
        "judgment_year": date_info["year"] or datetime.datetime.now().year,
        "judgment_date": date_info["date"],
        "citations": citation_info["value"],
        "bench": bench_info["value"] or "Could not confidently extract bench information",
        "petitioner": title_info["petitioner"],
        "respondent": title_info["respondent"],
        "case_number": case_number,
        "court": "Supreme Court of India",
        
        # Structured confidence & evidence fields
        "title_info": title_info,
        "citation_info": citation_info,
        "date_info": date_info,
        "bench_info": bench_info,
        "case_number_info": {
            "value": case_number,
            "confidence_score": case_num_score,
            "confidence": case_num_confidence,
            "evidence": case_num_evidence,
        },
        "critical_confidence_low": critical_confidence_low,
    }
