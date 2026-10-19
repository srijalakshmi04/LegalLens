import sys
import joblib
import scipy.sparse as sp
from pathlib import Path
from django.conf import settings
from django.core.management.base import BaseCommand
from cases.models import LegalCase
from cases.services.similarity import load_similarity_model
from cases.services.preprocessing import build_search_document


class Command(BaseCommand):
    help = "Verify LegalLens search index integrity, corpus alignment, and candidate searchable state"

    def handle(self, *args, **options):
        self.stdout.write("====================================================")
        self.stdout.write("   LEGAL LENS SEARCH INDEX INTEGRITY CHECK")
        self.stdout.write("====================================================")

        db_count = LegalCase.objects.count()

        try:
            vectorizer, case_matrix, case_matrix_T, case_ids, version = load_similarity_model()
            matrix_rows = case_matrix.shape[0]
            feature_cols = case_matrix.shape[1]
            mapping_count = len(case_ids)
            index_error = None
        except Exception as e:
            matrix_rows = 0
            feature_cols = 0
            mapping_count = 0
            case_ids = []
            index_error = str(e)

        self.stdout.write(f"Database cases       : {db_count}")
        self.stdout.write(f"Indexed corpus cases : {mapping_count}")
        self.stdout.write(f"Matrix rows          : {matrix_rows}")
        self.stdout.write(f"Case mapping entries : {mapping_count}")
        self.stdout.write(f"Feature dimensions   : {feature_cols}")

        if index_error:
            self.stderr.write(f"\n[FAIL] INDEX ERROR: {index_error}")
            return

        # Check for unindexed cases and batch-align if needed
        case_ids_set = set(case_ids)
        unindexed_cases = [c for c in LegalCase.objects.all() if c.id not in case_ids_set]

        if unindexed_cases:
            self.stdout.write(f"\n[ACTION] Batch indexing {len(unindexed_cases)} unindexed database case(s)...")
            docs = [build_search_document(c) or f"{c.title} {c.raw_text or ''}" for c in unindexed_cases]
            new_vectors = vectorizer.transform(docs)

            updated_matrix = sp.vstack([case_matrix, new_vectors]).tocsr()
            updated_ids = list(case_ids) + [c.id for c in unindexed_cases]

            model_dir = Path(settings.ML_MODELS_DIR)
            matrix_path = model_dir / ("case_matrix_v2.npz" if version == "hybrid-tfidf-v2" else "case_matrix.npz")
            ids_path = model_dir / ("case_ids_v2.joblib" if version == "hybrid-tfidf-v2" else "case_ids.joblib")

            tmp_m = matrix_path.with_suffix(".tmp.npz")
            sp.save_npz(tmp_m, updated_matrix)
            import os
            os.replace(tmp_m, matrix_path)

            tmp_i = ids_path.with_suffix(".tmp.joblib")
            joblib.dump(updated_ids, tmp_i)
            os.replace(tmp_i, ids_path)

            load_similarity_model.cache_clear()

            vectorizer, case_matrix, case_matrix_T, case_ids, version = load_similarity_model()
            matrix_rows = case_matrix.shape[0]
            mapping_count = len(case_ids)
            case_ids_set = set(case_ids)
            self.stdout.write(f"[OK] Batch indexing complete! New corpus size: {mapping_count}")

        latest_case = LegalCase.objects.order_by("-id").first()

        if latest_case:
            self.stdout.write(f"\nRecently ingested judgment:")
            self.stdout.write(f"ID: #{latest_case.id}")
            self.stdout.write(f"Title: {latest_case.title}")
            self.stdout.write(f"Year: {latest_case.judgment_year or 'N/A'}")
            self.stdout.write(f"Text length: {len(latest_case.raw_text or '')} characters")

            db_pass = LegalCase.objects.filter(id=latest_case.id).exists()
            corpus_pass = latest_case.id in case_ids
            row_idx = case_ids.index(latest_case.id) if corpus_pass else -1
            matrix_row_pass = (0 <= row_idx < matrix_rows)
            vector_nnz = case_matrix[row_idx].nnz if matrix_row_pass else 0
            vector_pass = vector_nnz > 0

            is_searchable = db_pass and corpus_pass and matrix_row_pass and vector_pass

            self.stdout.write(f"\nDATABASE        : {'[PASS]' if db_pass else '[FAIL]'}")
            self.stdout.write(f"CORPUS          : {'[PASS]' if corpus_pass else '[FAIL]'}")
            self.stdout.write(f"MATRIX ROW      : {'[PASS] (Row #' + str(row_idx) + ')' if matrix_row_pass else '[FAIL]'}")
            self.stdout.write(f"CASE MAPPING    : {'[PASS]' if corpus_pass else '[FAIL]'}")
            self.stdout.write(f"VECTOR NON-EMPTY: {'[PASS] (' + str(vector_nnz) + ' non-zero features)' if vector_pass else '[FAIL]'}")
            self.stdout.write(f"SEARCHABLE      : {'[YES]' if is_searchable else '[NO]'}")
        else:
            self.stdout.write("\nNo cases currently found in database.")

        if db_count != mapping_count:
            self.stdout.write(self.style.WARNING(
                f"\n[WARNING] DISCREPANCY DETECTED: Database has {db_count} cases while index has {mapping_count}."
            ))
        else:
            self.stdout.write(self.style.SUCCESS(
                f"\n[OK] PERFECT ALIGNMENT: Database count ({db_count}) matches indexed corpus count ({mapping_count})."
            ))

        self.stdout.write("====================================================")
