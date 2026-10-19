import re


CATEGORY_RULES = [
    {
        "name": "Criminal law",
        "lawyer": "Criminal lawyer",
        "keywords": {
            "arrest", "arrested", "bail", "crime", "criminal", "custody",
            "detained", "detention", "fir", "murder", "police", "prison",
            "theft", "trial",
        },
        "explanation": (
            "Your description appears to involve police action, detention, "
            "an alleged offence or criminal procedure."
        ),
        "steps": [
            "Keep copies of the FIR, notice, arrest memo and court papers.",
            "Write down the dates, police station and names of the people involved.",
            "Speak to a criminal lawyer promptly if anyone is detained or summoned.",
        ],
    },
    {
        "name": "Constitutional rights",
        "lawyer": "Constitutional or writ lawyer",
        "keywords": {
            "constitution", "constitutional", "equality", "freedom",
            "fundamental", "government", "liberty", "rights", "state",
            "writ",
        },
        "explanation": (
            "Your description may concern government action or the protection "
            "of a constitutional right."
        ),
        "steps": [
            "Preserve the government order, notice or communication being challenged.",
            "Prepare a clear timeline of how the action affected your rights.",
            "Consult a lawyer who handles constitutional and writ matters.",
        ],
    },
    {
        "name": "Family law",
        "lawyer": "Family lawyer",
        "keywords": {
            "alimony", "child", "custody", "divorce", "domestic", "dowry",
            "family", "husband", "maintenance", "marriage", "wife",
        },
        "explanation": (
            "Your description appears to involve marriage, maintenance, "
            "domestic relations or child custody."
        ),
        "steps": [
            "Keep marriage, residence, income and communication records.",
            "Record important dates and any existing court or police proceedings.",
            "Consult a family lawyer before signing any settlement.",
        ],
    },
    {
        "name": "Property and tenancy law",
        "lawyer": "Property or civil lawyer",
        "keywords": {
            "boundary", "eviction", "house", "land", "landlord", "lease",
            "ownership", "possession", "property", "rent", "tenant", "title",
        },
        "explanation": (
            "Your description appears to concern property ownership, possession, "
            "rent, tenancy or eviction."
        ),
        "steps": [
            "Collect the sale deed, lease, tax receipts and ownership records.",
            "Keep copies of notices, payments and communications.",
            "Consult a property or civil lawyer before transferring possession.",
        ],
    },
    {
        "name": "Employment and labour law",
        "lawyer": "Labour or employment lawyer",
        "keywords": {
            "dismissed", "employee", "employer", "employment", "fired", "job",
            "salary", "service", "suspended", "termination", "wages", "work",
        },
        "explanation": (
            "Your description appears to involve employment, salary, suspension "
            "or termination from service."
        ),
        "steps": [
            "Keep the appointment letter, salary records and termination notice.",
            "Save emails, messages and internal complaints.",
            "Consult a labour or employment lawyer about the correct forum.",
        ],
    },
    {
        "name": "Consumer law",
        "lawyer": "Consumer dispute lawyer",
        "keywords": {
            "bought", "consumer", "defect", "defective", "delivery", "product",
            "refund", "seller", "service", "warranty",
        },
        "explanation": (
            "Your description may involve a defective product, deficient service, "
            "refund or another consumer dispute."
        ),
        "steps": [
            "Keep invoices, warranty documents and payment records.",
            "Save written complaints and the seller's responses.",
            "Consult a consumer dispute lawyer if the business does not resolve it.",
        ],
    },
    {
        "name": "Banking and debt law",
        "lawyer": "Banking or debt-recovery lawyer",
        "keywords": {
            "bank", "borrowed", "debt", "emi", "interest", "loan", "mortgage",
            "recovery", "repayment",
        },
        "explanation": (
            "Your description appears to involve a bank, loan, repayment, "
            "interest or debt recovery."
        ),
        "steps": [
            "Collect the loan agreement, account statement and all notices.",
            "Do not ignore a court, tribunal or recovery notice.",
            "Consult a banking or debt-recovery lawyer before agreeing to terms.",
        ],
    },
    {
        "name": "Cyber and online law",
        "lawyer": "Cybercrime lawyer",
        "keywords": {
            "account", "cyber", "hacked", "internet", "mobile", "online",
            "otp", "scam", "social", "website",
        },
        "explanation": (
            "Your description may involve an online account, cybercrime, "
            "digital fraud or misuse of electronic information."
        ),
        "steps": [
            "Preserve screenshots, URLs, transaction IDs and message records.",
            "Report financial fraud quickly to the bank and appropriate authority.",
            "Consult a cybercrime lawyer if loss or identity misuse is involved.",
        ],
    },
]

DEFAULT_CATEGORY = {
    "name": "General civil dispute",
    "lawyer": "Civil lawyer",
    "explanation": (
        "The description does not strongly match one category, but it may involve "
        "a general dispute concerning rights, obligations or compensation."
    ),
    "steps": [
        "Collect contracts, notices, receipts and relevant communications.",
        "Write a chronological account of what happened.",
        "Consult a civil lawyer to identify the correct legal remedy and forum.",
    ],
}


def _words(text):
    return set(re.findall(r"[a-z][a-z0-9-]{2,}", (text or "").lower()))


def classify_issue(issue):
    """Classify an everyday description using transparent keyword evidence."""
    words = _words(issue)
    best_category = DEFAULT_CATEGORY
    best_matches = set()

    for category in CATEGORY_RULES:
        matches = words & category["keywords"]
        if len(matches) > len(best_matches):
            best_category = category
            best_matches = matches

    return {
        "name": best_category["name"],
        "lawyer": best_category["lawyer"],
        "explanation": best_category["explanation"],
        "steps": best_category["steps"],
        "matched_words": sorted(best_matches),
    }


def build_plain_language_assistance(issue):
    """Return plain-language category and lawyer guidance."""
    category = classify_issue(issue)
    return {
        "category": category,
    }
