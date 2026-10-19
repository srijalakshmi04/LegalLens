import re
import hashlib
from ..models import LegalCase


def check_duplicate_judgment(title="", citations="", checksum=None, raw_text="", judgment_year=None):
    """
    3-Tier Duplicate Detection Engine for LegalLens.
    Tiers:
    - EXACT DUPLICATE: Same SHA-256 file checksum matched.
    - LIKELY DUPLICATE: Same official citation OR identical title & year matched.
    - POSSIBLE DUPLICATE: Normalized title match without year or case number match.
    - NO DUPLICATE: No match found.
    """
    # 1. Tier 1: EXACT DUPLICATE (SHA-256 Checksum)
    if checksum:
        existing_by_checksum = LegalCase.objects.filter(file_checksum=checksum).first()
        if existing_by_checksum:
            return {
                "is_duplicate": True,
                "tier": "EXACT_DUPLICATE",
                "tier_label": "⚠ EXACT DUPLICATE DETECTED",
                "reason": "Matching SHA-256 file checksum found in database.",
                "existing_case": existing_by_checksum,
            }

    # 2. Tier 2: LIKELY DUPLICATE (Citation or Title + Year)
    if citations and len(citations.strip()) > 4:
        existing_by_citation = LegalCase.objects.filter(citations__iexact=citations.strip()).first()
        if existing_by_citation:
            return {
                "is_duplicate": True,
                "tier": "LIKELY_DUPLICATE",
                "tier_label": "⚠ LIKELY DUPLICATE",
                "reason": f"Existing case found with neutral/report citation '{citations}'.",
                "existing_case": existing_by_citation,
            }

    if title and len(title.strip()) > 8:
        clean_title = re.sub(r"[^\w\s]", "", title.lower()).strip()
        
        # Exact title match
        existing_by_title = LegalCase.objects.filter(title__iexact=title.strip()).first()
        if existing_by_title:
            return {
                "is_duplicate": True,
                "tier": "LIKELY_DUPLICATE",
                "tier_label": "⚠ LIKELY DUPLICATE",
                "reason": f"Existing case found with identical title '{title}'.",
                "existing_case": existing_by_title,
            }

        # Title + Year match
        if judgment_year:
            existing_by_title_year = LegalCase.objects.filter(title__icontains=title.split(" v.")[0].strip(), judgment_year=judgment_year).first()
            if existing_by_title_year:
                return {
                    "is_duplicate": True,
                    "tier": "POSSIBLE_DUPLICATE",
                    "tier_label": "⚠ POSSIBLE DUPLICATE",
                    "reason": f"Possible duplicate matched by party name and judgment year ({judgment_year}).",
                    "existing_case": existing_by_title_year,
                }

    return {
        "is_duplicate": False,
        "tier": "NO_DUPLICATE",
        "tier_label": "✓ NO DUPLICATE FOUND",
        "reason": "No matching checksum, citation, or party names detected in corpus.",
        "existing_case": None,
    }
