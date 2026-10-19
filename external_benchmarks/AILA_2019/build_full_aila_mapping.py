import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'legal_system.settings')
django.setup()

from cases.models import LegalCase

AILA_DIR = PROJECT_ROOT / "external_benchmarks" / "AILA_2019"
PROCESSED_FILE = AILA_DIR / "processed" / "official_aila_parsed_dataset.json"
MAPPINGS_DIR = AILA_DIR / "mappings"
MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)

if not PROCESSED_FILE.exists():
    print(f"Error: Processed dataset missing at {PROCESSED_FILE}")
    sys.exit(1)

with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
    aila_data = json.load(f)

casedocs = aila_data.get("parsed_cases", [])
print(f"Loaded {len(casedocs)} official AILA case documents for full mapping audit...")

# Get all LegalLens cases indexed in database
legallens_cases = list(LegalCase.objects.all())
print(f"Loaded {len(legallens_cases)} LegalLens database cases.")

# Pre-index LegalLens cases by title & ID
ll_by_id = {str(lc.id): lc for lc in legallens_cases}
ll_by_title = {lc.title.strip().lower(): lc for lc in legallens_cases if lc.title}

mapping = {}
verified_count = 0
unmapped_count = 0

for c in casedocs:
    acid = c["aila_case_id"]
    
    # Matching Strategy:
    # 1. Match numeric ID (e.g. C1 -> LegalCase ID 1)
    numeric_id = acid.replace("C", "").replace("AILA_C", "").strip()
    matched_lc = None
    mapping_method = "UNMAPPED"

    if numeric_id in ll_by_id:
        matched_lc = ll_by_id[numeric_id]
        mapping_method = "EXACT_NUMERIC_ID_MATCH"

    if matched_lc:
        verified_count += 1
        mapping[acid] = {
            "legallens_case_id": matched_lc.id,
            "legallens_title": matched_lc.title,
            "mapping_method": mapping_method,
            "status": "VERIFIED"
        }
    else:
        unmapped_count += 1
        mapping[acid] = {
            "legallens_case_id": None,
            "legallens_title": None,
            "mapping_method": "NONE",
            "status": "UNMAPPED"
        }

coverage_pct = round((verified_count / len(casedocs)) * 100.0, 2) if casedocs else 0.0

output_mapping_path = MAPPINGS_DIR / "full_case_id_mapping.json"
with open(output_mapping_path, "w", encoding="utf-8") as f:
    json.dump({
        "total_aila_cases": len(casedocs),
        "mapped_cases": verified_count,
        "unmapped_cases": unmapped_count,
        "mapping_coverage_pct": coverage_pct,
        "mappings": mapping
    }, f, indent=2)

print("\n--- FULL AILA CASE MAPPING COMPLETED ---")
print(f"Total AILA Cases    : {len(casedocs)}")
print(f"Mapped AILA Cases   : {verified_count}")
print(f"Unmapped AILA Cases : {unmapped_count}")
print(f"Mapping Coverage    : {coverage_pct}%")
print(f"Saved full mapping to {output_mapping_path}")
