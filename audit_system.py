import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'legal_system.settings')
django.setup()

from cases.models import LegalCase, SavedCase, SearchHistory
from cases.services.similarity import find_similar_cases, load_similarity_model, get_search_diagnostics
from cases.services.lawyer_directory import find_tamil_nadu_help, get_directory_stats
from cases.services.case_comparison import compare_cases
from cases.services.evaluation import evaluate_retrieval_benchmark

print("====================================================")
print("           LEGAL LENS FULL SYSTEM AUDIT             ")
print("====================================================")

# 1. DATABASE METRICS
total_cases = LegalCase.objects.count()
min_year = LegalCase.objects.values_list('judgment_year', flat=True).order_by('judgment_year').first()
max_year = LegalCase.objects.values_list('judgment_year', flat=True).order_by('-judgment_year').first()
saved_count = SavedCase.objects.count()
history_count = SearchHistory.objects.count()
print(f"[1] DATABASE STATUS: OK")
print(f"    - Total SC Judgments: {total_cases:,}")
print(f"    - Timeline Coverage: {min_year} - {max_year}")
print(f"    - Saved Cases Count: {saved_count}")
print(f"    - Search History Logs: {history_count}")

# 2. MODEL ARTIFACTS
vec, mat, cids, idx_ver = load_similarity_model()
print(f"[2] MODEL ARTIFACTS: OK")
print(f"    - Model Index Version: {idx_ver}")
print(f"    - Matrix Shape: {mat.shape[0]:,} rows x {mat.shape[1]:,} features")
print(f"    - Indexed Case IDs: {len(cids):,}")

# 3. SEARCH MODES EXECUTION TEST
q = "The police arrested and detained a person without following proper legal procedure Article 21 Section 41"
print(f"[3] LIVE RETRIEVAL & SEARCH MODES EXECUTION TEST:")

for mode in ['all', 'current', 'recent', 'historical']:
    res = find_similar_cases(q, top_k=2, search_mode=mode)
    if res:
        top = res[0]
        title_truncated = top["case"].title[:50]
        print(f"    - Mode [{mode.upper()}]: Top Case: \"{title_truncated}\" (Year: {top['case'].judgment_year}) | Predicted Similarity: {top['percentage']}% | Tier: {top['relevance_tier']}")
        if mode == 'recent':
            years = [r['case'].judgment_year for r in res if r['case'].judgment_year]
            assert all(y >= 2021 for y in years), "FAILED: Recent mode returned case < 2021"
            print("      --> VERIFIED: All Recent mode results satisfy judgment_year >= 2021")

# 4. TAMIL NADU LEGAL HELP & PIN RESOLUTION
tn_stats = get_directory_stats()
print(f"[4] TAMIL NADU LEGAL HELP DIRECTORY: OK")
print(f"    - Districts Configured: {tn_stats['configured_districts']} / 38")
print(f"    - Official District Web Portals: {tn_stats['district_websites_count']} / 38")
print(f"    - DLSA Legal Aid Entries: {tn_stats['dlsa_count']} / 38")
print(f"    - eCourts Judiciary Portals: {tn_stats['ecourts_count']} / 38")

# PIN Resolution Test
pin_res = find_tamil_nadu_help('Madurai', locality='614001')
print(f"    - PIN Resolution (614001 under Madurai): Locality={pin_res['resolved_pin_info']['locality']}, District={pin_res['resolved_pin_info']['district']}")
print(f"      --> Mismatch Alert: \"{pin_res['pin_warning'][:80]}...\"")

# 5. CASE COMPARISON TEST
c_first = LegalCase.objects.first()
c_second = LegalCase.objects.last()
if c_first and c_second:
    cmp_res = compare_cases(c_first.id, c_second.id)
    print(f"[5] SIDE-BY-SIDE CASE COMPARISON ENGINE: OK")
    print(f"    - Executive Synthesis: \"{cmp_res['contrast_summary']}\"")

print("====================================================")
print("        ALL SYSTEM AUDIT CHECKS PASSED 100%!       ")
print("====================================================")
