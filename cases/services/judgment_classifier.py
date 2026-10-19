import re

# Rule-based explainable legal categories with keyword and statutory triggers
CATEGORY_RULES = {
    "Criminal Law": {
        "statutes": {"ipc", "crpc", "indian penal code", "code of criminal procedure", "pocso", "ndps", "prevention of corruption"},
        "keywords": {"bail", "arrest", "custody", "police", "fir", "accused", "conviction", "acquittal", "sentence", "murder", "ipc", "crpc", "charge sheet", "offence", "penal", "custodial", "session court", "section 302", "section 307", "section 376", "section 420", "section 498a"},
    },
    "Constitutional Law": {
        "statutes": {"constitution of india", "article 14", "article 19", "article 21", "article 32", "article 226", "article 136"},
        "keywords": {"fundamental rights", "writ petition", "mandamus", "certiorari", "habeas corpus", "constitutional bench", "ultra vires", "article 21", "article 14", "sovereign", "constitutional validity", "parliamentary"},
    },
    "Commercial & Insolvency Law": {
        "statutes": {"ibc", "insolvency and bankruptcy code", "companies act", "negotiable instruments act", "nclt", "nclat", "sebi"},
        "keywords": {"insolvency", "bankruptcy", "nclat", "nclt", "liquidation", "corporate debtor", "creditor", "financial creditor", "cheque bounce", "section 138", "promoter", "shares", "debentures"},
    },
    "Property & Land Law": {
        "statutes": {"transfer of property act", "land acquisition act", "registration act", "stamp act", "rent control act"},
        "keywords": {"possession", "title", "land acquisition", "compensation", "tenant", "eviction", "lease", "mortgage", "partition", "ownership", "easement", "revenue records", "patta"},
    },
    "Family & Matrimonial Law": {
        "statutes": {"hindu marriage act", "special marriage act", "guardians and wards act", "hindu succession act", "protection of women from domestic violence"},
        "keywords": {"marriage", "matrimonial", "divorce", "maintenance", "custody", "alimony", "section 125", "domestic violence", "stridhan", "husband", "wife", "guardianship"},
    },
    "Service & Pension Law": {
        "statutes": {"central civil services", "service rules", "pension act", "administrative tribunals act", "cat"},
        "keywords": {"pension", "promotion", "dismissal", "termination", "suspension", "salary", "service rules", "disciplinary proceedings", "cat", "seniority", "gratuity", "retiral benefits"},
    },
    "Tax & Revenue Law": {
        "statutes": {"income tax act", "gst", "central excise act", "customs act", "sales tax"},
        "keywords": {"income tax", "assessment year", "revenue", "customs", "excise", "gst", "taxable", "assessee", "deduction", "section 148", "commissioner of income tax"},
    },
    "Arbitration & Conciliation": {
        "statutes": {"arbitration and conciliation act"},
        "keywords": {"arbitration", "arbitral tribunal", "award", "arbitrator", "section 11", "section 34", "section 9", "arbitral award"},
    },
    "Labour & Industrial Law": {
        "statutes": {"industrial disputes act", "factories act", "minimum wages act", "workmen compensation"},
        "keywords": {"workman", "workmen", "industrial dispute", "trade union", "strike", "lockout", "retrenchment", "reinstatement", "labour court"},
    },
    "Consumer Protection": {
        "statutes": {"consumer protection act"},
        "keywords": {"consumer", "deficiency of service", "unfair trade practice", "district forum", "national commission", "state commission", "complainant"},
    },
    "Administrative & Public Law": {
        "statutes": {"right to information act", "lokpal", "public records"},
        "keywords": {"natural justice", "quasi judicial", "administrative discretion", "bias", "locus standi", "public interest litigation", "pil"},
    },
    "Civil Procedure & Remedies": {
        "statutes": {"code of civil procedure", "cpc", "specific relief act", "limitation act"},
        "keywords": {"injunction", "specific performance", "cpc", "res judicata", "order 39", "section 9 cpc", "plaint", "written statement", "decree", "execution petition"},
    },
}


def classify_judgment(text, acts="", sections=""):
    """
    Genuine rule-based explainable legal classifier for Supreme Court judgments.
    Computes keyword and statutory trigger scores, confidence percentages, and explainable evidence.
    """
    if not text:
        return {
            "primary_category": "General Legal Dispute",
            "primary_confidence_pct": 0,
            "primary_confidence_label": "LOW",
            "secondary_categories": [],
            "evidence_list": ["• General legal terminology (No explicit statutory triggers detected)"],
            "scores": {},
        }

    combined_text = (text[:8000] + " " + (acts or "") + " " + (sections or "")).lower()
    scores = {}
    evidence_map = {}

    for category, rules in CATEGORY_RULES.items():
        score = 0
        cat_evidence = []

        for stat in rules["statutes"]:
            if stat in combined_text:
                score += 5
                cat_evidence.append(f"• Statute/Code matched: {stat.title()}")

        for kw in rules["keywords"]:
            if re.search(r"\b" + re.escape(kw) + r"\b", combined_text):
                score += 2 if len(kw) > 6 else 1
                if len(cat_evidence) < 4:
                    cat_evidence.append(f"• Key term detected: '{kw}'")

        scores[category] = score
        evidence_map[category] = cat_evidence

    sorted_categories = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_score = sorted_categories[0][1] if sorted_categories else 0

    if top_score > 0:
        primary_category = sorted_categories[0][0]
        primary_confidence_pct = min(95, max(35, int((top_score / 14.0) * 100)))
        if primary_confidence_pct >= 65:
            confidence_label = "HIGH"
        elif primary_confidence_pct >= 45:
            confidence_label = "MEDIUM"
        else:
            confidence_label = "LOW"

        primary_evidence = evidence_map.get(primary_category, [])
        if not primary_evidence:
            primary_evidence = [f"• Terminology matched in {primary_category} domain"]

        secondary_categories = []
        for cat, s in sorted_categories[1:4]:
            # Thresholding: Only include secondary category if score >= 3 AND score >= 35% of top_score
            if s >= 3 and s >= (0.35 * top_score):
                cat_conf_pct = min(90, max(20, int((s / 14.0) * 100)))
                secondary_categories.append({
                    "category": cat,
                    "score": s,
                    "confidence_pct": cat_conf_pct,
                    "evidence": evidence_map.get(cat, []),
                })
    else:
        primary_category = "General Legal Dispute"
        primary_confidence_pct = 20
        confidence_label = "LOW"
        secondary_categories = []
        primary_evidence = ["• General judicial language detected without specific statutory triggers"]

    return {
        "primary_category": primary_category,
        "primary_confidence_pct": primary_confidence_pct,
        "primary_confidence_label": confidence_label,
        "secondary_categories": secondary_categories,
        "classification_evidence": primary_evidence,
        "scores": {cat: s for cat, s in sorted_categories[:5] if s > 0},
    }
