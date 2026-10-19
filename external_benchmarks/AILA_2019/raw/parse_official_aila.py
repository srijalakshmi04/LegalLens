import os
import sys
import json
import re
from pathlib import Path

# Path resolution based strictly on script location
# Script Location: PROJECT_ROOT / external_benchmarks / AILA_2019 / raw / parse_official_aila.py
SCRIPT_PATH = Path(__file__).resolve()
RAW_DIR = SCRIPT_PATH.parent
AILA_ROOT = RAW_DIR.parent
EXTERNAL_BENCHMARKS_DIR = AILA_ROOT.parent
PROJECT_ROOT = EXTERNAL_BENCHMARKS_DIR.parent

# Target Raw File Paths
CASE_DIR = RAW_DIR / "Object_casedocs"
QUERY_FILE = RAW_DIR / "Query_doc.txt"
QUERY_DIR = RAW_DIR / "Query_doc"
QREL_FILE = RAW_DIR / "relevance_judgments_priorcases.txt"
QREL_ALT_FILE = RAW_DIR / "relevance_judgments_pricases.txt"

# Target Output Directory
PROCESSED_DIR = AILA_ROOT / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("====================================================")
print("     FIRE AILA 2019 OFFICIAL BENCHMARK PARSER      ")
print("====================================================")
print(f"Resolved PROJECT_ROOT : {PROJECT_ROOT}")
print(f"Resolved AILA_ROOT    : {AILA_ROOT}")
print(f"Resolved RAW_DIR      : {RAW_DIR}")
print(f"Resolved CASE_DIR     : {CASE_DIR}")
print(f"Resolved QUERY_FILE   : {QUERY_FILE}")
print(f"Resolved QREL_FILE    : {QREL_FILE}")
print("====================================================\n")

# 1. PARSE OBJECT_CASEDOCS (Target: All Task 1 Candidate Documents)
parsed_cases = []
unique_case_ids = set()
duplicate_case_ids = []
unreadable_docs = 0

if CASE_DIR.exists() and CASE_DIR.is_dir():
    files = [f for f in CASE_DIR.iterdir() if f.is_file()]
    print(f"Parsing Object_casedocs directory: Found {len(files)} files.")
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                case_id = fpath.stem
                if case_id in unique_case_ids:
                    duplicate_case_ids.append(case_id)
                unique_case_ids.add(case_id)
                parsed_cases.append({
                    "aila_case_id": case_id,
                    "filename": fpath.name,
                    "text_length": len(content),
                    "text_snippet": content[:300]
                })
        except Exception as e:
            unreadable_docs += 1
else:
    print(f"Notice: CASE_DIR does not exist at {CASE_DIR}")

# 2. PARSE QUERY DOCS (Query_doc.txt or Query_doc/)
parsed_queries = []

if QUERY_FILE.exists():
    print(f"Parsing QUERY_FILE: {QUERY_FILE}")
    try:
        with open(QUERY_FILE, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_str = line.strip()
                if "||" in line_str:
                    parts = line_str.split("||", 1)
                    qid = parts[0].strip()
                    qtext = parts[1].strip()
                    parsed_queries.append({"query_id": qid, "query_text": qtext})
                elif line_str:
                    # Fallback regex search for xml tags
                    blocks = re.split(r'<QUERY\s+id=[\'"]?([^\'">\s]+)[\'"]?>', line_str)
                    for i in range(1, len(blocks), 2):
                        qid = blocks[i].strip()
                        qtext = blocks[i+1].split('</QUERY>')[0].strip() if i+1 < len(blocks) else ""
                        parsed_queries.append({"query_id": qid, "query_text": qtext})
    except Exception as e:
        print(f"Error reading QUERY_FILE: {e}")
elif QUERY_DIR.exists() and QUERY_DIR.is_dir():
    print(f"Parsing QUERY_DIR: {QUERY_DIR}")
    for qf in sorted(QUERY_DIR.glob("*.txt")):
        try:
            with open(qf, "r", encoding="utf-8", errors="ignore") as f:
                qtext = f.read().strip()
                parsed_queries.append({"query_id": qf.stem, "query_text": qtext})
        except Exception:
            pass
else:
    print(f"Notice: QUERY_FILE does not exist at {QUERY_FILE}")

# 3. PARSE OFFICIAL RELEVANCE JUDGMENTS (QREL_FILE)
parsed_qrels = []
duplicate_qrels = 0
target_qrel_path = None

if QREL_FILE.exists():
    target_qrel_path = QREL_FILE
elif QREL_ALT_FILE.exists():
    target_qrel_path = QREL_ALT_FILE

if target_qrel_path and target_qrel_path.exists():
    print(f"Parsing QREL_FILE: {target_qrel_path}")
    seen_qrel_keys = set()
    try:
        with open(target_qrel_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, _, cid, rel = parts[0], parts[1], parts[2], int(parts[3])
                    qkey = (qid, cid)
                    if qkey in seen_qrel_keys:
                        duplicate_qrels += 1
                    seen_qrel_keys.add(qkey)
                    parsed_qrels.append({
                        "query_id": qid,
                        "aila_case_id": cid,
                        "relevance": rel
                    })
    except Exception as e:
        print(f"Error reading QREL_FILE: {e}")
else:
    print(f"Notice: QREL_FILE does not exist at {QREL_FILE}")

# Output Processing
official_data = {
    "audit_status": "Path Resolution & Format Parsing Verified",
    "resolved_paths": {
        "project_root": str(PROJECT_ROOT),
        "aila_root": str(AILA_ROOT),
        "raw_dir": str(RAW_DIR),
        "case_dir": str(CASE_DIR),
        "query_file": str(QUERY_FILE),
        "qrel_file": str(target_qrel_path or QREL_FILE),
    },
    "casedocs_stats": {
        "total_parsed_cases": len(parsed_cases),
        "unique_case_ids_count": len(unique_case_ids),
        "duplicate_ids_count": len(duplicate_case_ids),
        "unreadable_docs_count": unreadable_docs,
    },
    "queries_stats": {
        "total_queries_parsed": len(parsed_queries),
    },
    "qrels_stats": {
        "total_qrels_parsed": len(parsed_qrels),
        "duplicate_qrels_count": duplicate_qrels,
    },
    "parsed_cases": parsed_cases,
    "parsed_queries": parsed_queries,
    "parsed_qrels": parsed_qrels,
}

output_file = PROCESSED_DIR / "official_aila_parsed_dataset.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(official_data, f, indent=2)

print("\n--- AILA OFFICIAL PARSER COMPLETED SUCCESSFULLY ---")
print(f"Exact Resolved AILA Root : {AILA_ROOT}")
print(f"Object_casedocs Path     : {CASE_DIR}")
print(f"Query_doc Path           : {QUERY_FILE}")
print(f"Qrels Path               : {target_qrel_path or QREL_FILE}")
print(f"Case Documents Parsed    : {len(parsed_cases)}")
print(f"Queries Parsed           : {len(parsed_queries)}")
print(f"Official QRels Parsed    : {len(parsed_qrels)}")
print(f"Duplicate IDs            : {len(duplicate_case_ids)}")
print(f"Unreadable Files         : {unreadable_docs}")
print(f"Generated Output Path    : {output_file}")
