
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def clean_text(text):
    """Clean PDF text while preserving important legal words."""

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    # Remove Indian Kanoon page footer.
    text = re.sub(
        r"Indian Kanoon\s*-\s*https?://indiankanoon\.org/doc/\d+/?",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove control characters.
    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
        " ",
        text,
    )

    # Remove lines containing only page numbers.
    text = re.sub(
        r"(?m)^\s*\d+\s*$",
        " ",
        text,
    )

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_pdf_text(pdf_path):
    """Extract complete text and metadata from one PDF."""

    reader = PdfReader(str(pdf_path))
    pages = []

    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""

        pages.append(page_text)

    full_text = "\n".join(pages).strip()

    metadata = {
        "title": "",
        "author": "",
    }

    if reader.metadata:
        metadata["title"] = str(
            reader.metadata.title or ""
        ).strip()

        metadata["author"] = str(
            reader.metadata.author or ""
        ).strip()

    return full_text, len(reader.pages), metadata


def title_from_filename(filename):
    """Convert the PDF filename into a readable case title."""

    stem = Path(filename).stem

    stem = re.sub(
        r"_on_\d{1,2}_[A-Za-z]+_\d{4}(?:_\d+)?$",
        "",
        stem,
        flags=re.IGNORECASE,
    )

    stem = re.sub(r"_\d+$", "", stem)
    stem = stem.replace("_", " ")
    stem = re.sub(r"\s+", " ", stem)

    return stem.strip()


def date_from_filename(filename):
    """Extract date from filenames such as _on_19_May_1950."""

    match = re.search(
        r"_on_(\d{1,2})_([A-Za-z]+)_(\d{4})",
        filename,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    day = int(match.group(1))
    month_name = match.group(2).lower()
    year = int(match.group(3))

    month = MONTHS.get(month_name)

    if not month:
        return None

    try:
        return datetime(year, month, day).date()
    except ValueError:
        return None


def extract_line(text, label):
    """Extract a single labelled line from the PDF text."""

    pattern = rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$"
    match = re.search(pattern, text)

    if match:
        return match.group(1).strip()

    return ""


def extract_between(text, start_label, end_labels, limit=5000):
    """Extract text found between two judgment headings."""

    end_pattern = "|".join(
        re.escape(label) for label in end_labels
    )

    pattern = (
        rf"(?is){re.escape(start_label)}\s*:?\s*"
        rf"(.*?)"
        rf"(?=\n\s*(?:{end_pattern})\s*:?\s*|\Z)"
    )

    match = re.search(pattern, text)

    if not match:
        return ""

    value = re.sub(r"\s+", " ", match.group(1)).strip()

    return value[:limit]


def extract_case_information(
    raw_text,
    filename,
    pdf_metadata,
    folder_year,
):
    """Extract structured information from one judgment."""

    title = pdf_metadata.get("title", "").strip()

    if not title or title.lower() in {"none", "untitled"}:
        title = title_from_filename(filename)

    # Remove metadata text added after the case title.
    title = re.sub(
        r"\s*:\s*\.\.\.\s*on\s+.+$",
        "",
        title,
        flags=re.IGNORECASE,
    )

    title = re.sub(
        r"\s+on\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}$",
        "",
        title,
        flags=re.IGNORECASE,
    )

    title = title.strip()

    judgment_date = date_from_filename(filename)

    petitioner = extract_between(
        raw_text,
        "PETITIONER",
        ["Vs.", "VS", "RESPONDENT"],
        limit=500,
    )

    respondent = extract_between(
        raw_text,
        "RESPONDENT",
        [
            "DATE OF JUDGMENT",
            "BENCH",
            "CITATION",
        ],
        limit=500,
    )

    acts = extract_between(
        raw_text,
        "ACT",
        [
            "HEADNOTE",
            "JUDGMENT",
            "ORDER",
        ],
        limit=5000,
    )

    return {
        "title": title or title_from_filename(filename),
        "judgment_year": (
            judgment_date.year
            if judgment_date
            else folder_year
        ),
        "judgment_date": judgment_date,
        "citations": extract_line(
            raw_text,
            "Equivalent citations",
        ),
        "author": extract_line(raw_text, "Author"),
        "bench": extract_line(raw_text, "Bench"),
        "petitioner": petitioner,
        "respondent": respondent,
        "acts": acts,
    }
LEGAL_REFERENCE_PATTERN = re.compile(
    r"\b(?:"
    r"Articles?\s+\d+[A-Za-z]?(?:\(\d+\))*"
    r"|"
    r"Sections?\s+\d+[A-Za-z]?(?:\(\d+\))*"
    r"|"
    r"Order\s+[IVXLCDM\d]+(?:\s+Rule\s+\d+)?"
    r")\b",
    flags=re.IGNORECASE,
)


def extract_legal_references(text, maximum=100):
    """Extract unique constitutional and statutory references."""

    references = []
    seen = set()

    for match in LEGAL_REFERENCE_PATTERN.finditer(text or ""):
        reference = " ".join(match.group(0).split())
        normalized = reference.lower()

        if normalized in seen:
            continue

        seen.add(normalized)
        references.append(reference)

        if len(references) >= maximum:
            break

    return references


def normalize_legal_references(text):
    """Return normalized references for comparison."""

    return {
        reference.lower()
        for reference in extract_legal_references(text)
    }


def build_search_document(legal_case):
    """
    Build a weighted training document.

    Legal provisions receive extra weight while complete judgment
    text remains the primary source.
    """

    sections = legal_case.legal_sections or ""
    acts = legal_case.acts or ""

    parts = [
        legal_case.title or "",
        legal_case.citations or "",
        acts,
        acts,
        sections,
        sections,
        sections,
        legal_case.cleaned_text or "",
    ]

    return " ".join(
        part.strip()
        for part in parts
        if part and part.strip()
    )