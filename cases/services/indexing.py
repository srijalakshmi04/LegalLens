import os
import time
import shutil
import logging
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from django.conf import settings

from cases.models import LegalCase
from cases.services.preprocessing import build_search_document
from cases.services.similarity import load_similarity_model

logger = logging.getLogger(__name__)


def auto_index_legal_case(legal_case):
    """
    Automatically index a newly committed LegalCase into the production TF-IDF vector space matrix.
    Appends the document vector to the sparse matrix, updates case ID mapping, invalidates the lru_cache,
    and performs post-index validation & smoke verification.
    Uses atomic file replacement to prevent corrupting matrix files during concurrent access.
    """
    start_time = time.perf_counter()

    try:
        # 1. Load active similarity model
        vectorizer, case_matrix, case_matrix_T, case_ids, index_version = load_similarity_model()

        # Check if already present
        if legal_case.id in case_ids:
            logger.info(f"Case ID #{legal_case.id} is already present in search index.")
            row_idx = case_ids.index(legal_case.id)
            return _validate_and_build_status(legal_case, case_matrix, case_ids, row_idx, start_time, index_version)

        # 2. Build search document text & transform vector
        doc_text = build_search_document(legal_case)
        if not doc_text or len(doc_text.strip()) < 10:
            doc_text = f"{legal_case.title} {legal_case.raw_text or ''}"

        new_vector = vectorizer.transform([doc_text])

        # 3. Stack vector onto case_matrix
        if sp.issparse(case_matrix):
            updated_matrix = sp.vstack([case_matrix, new_vector]).tocsr()
        else:
            updated_matrix = sp.csr_matrix(np.vstack([case_matrix, new_vector.toarray()]))

        updated_case_ids = list(case_ids) + [legal_case.id]

        # 4. Save updated artifacts using atomic replace
        model_dir = Path(settings.ML_MODELS_DIR)

        if index_version == "hybrid-tfidf-v2":
            matrix_path = model_dir / "case_matrix_v2.npz"
            ids_path = model_dir / "case_ids_v2.joblib"
        else:
            matrix_path = model_dir / "case_matrix.npz"
            ids_path = model_dir / "case_ids.joblib"

        _atomic_save_npz(matrix_path, updated_matrix)
        _atomic_dump_joblib(ids_path, updated_case_ids)

        # 5. Invalidate lru_cache
        load_similarity_model.cache_clear()

        # 6. Reload and verify
        v_new, m_new, mT_new, ids_new, ver_new = load_similarity_model()

        row_idx = ids_new.index(legal_case.id) if legal_case.id in ids_new else -1

        return _validate_and_build_status(legal_case, m_new, ids_new, row_idx, start_time, ver_new)

    except Exception as e:
        logger.exception(f"Automatic indexing failed for Case #{legal_case.id}: {e}")
        return {
            "is_indexed": False,
            "is_searchable": False,
            "status_label": "INDEXING_FAILED",
            "db_status": "[SAVED]",
            "index_status": "[FAILED]",
            "search_status": "[NOT SEARCHABLE]",
            "error": str(e),
            "elapsed_seconds": round(time.perf_counter() - start_time, 3),
        }


def _atomic_save_npz(target_path, matrix):
    tmp_path = target_path.with_suffix(".tmp.npz")
    sp.save_npz(tmp_path, matrix)
    os.replace(tmp_path, target_path)


def _atomic_dump_joblib(target_path, obj):
    tmp_path = target_path.with_suffix(".tmp.joblib")
    joblib.dump(obj, tmp_path)
    os.replace(tmp_path, target_path)


def _validate_and_build_status(legal_case, matrix, case_ids, row_idx, start_time, index_version):
    """
    Post-index validation and smoke test.
    Checks:
    A. Case exists in database
    B. Case ID exists in case-to-row mapping
    C. Matrix row index is valid
    D. Matrix row vector is non-empty (has non-zero TF-IDF features)
    """
    db_pass = LegalCase.objects.filter(id=legal_case.id).exists()
    corpus_pass = legal_case.id in case_ids
    matrix_row_pass = (0 <= row_idx < matrix.shape[0])
    vector_nnz = matrix[row_idx].nnz if (matrix_row_pass and sp.issparse(matrix)) else 0
    vector_pass = vector_nnz > 0

    is_searchable = db_pass and corpus_pass and matrix_row_pass and vector_pass

    elapsed = round(time.perf_counter() - start_time, 3)

    return {
        "is_indexed": True,
        "is_searchable": is_searchable,
        "status_label": "SEARCHABLE" if is_searchable else "VALIDATION_FAILED",
        "db_status": "[SAVED]",
        "index_status": "[INDEXED]",
        "search_status": "[SEARCHABLE]" if is_searchable else "[PENDING]",
        "case_id": legal_case.id,
        "matrix_row": row_idx,
        "corpus_size": matrix.shape[0],
        "feature_count": matrix.shape[1],
        "non_zero_features": int(vector_nnz),
        "elapsed_seconds": elapsed,
        "index_version": index_version,
        "validations": {
            "database": "[PASS]" if db_pass else "[FAIL]",
            "corpus": "[PASS]" if corpus_pass else "[FAIL]",
            "matrix_row": "[PASS]" if matrix_row_pass else "[FAIL]",
            "case_mapping": "[PASS]" if corpus_pass else "[FAIL]",
            "vector_non_empty": "[PASS]" if vector_pass else "[LOW_NNZ]",
        }
    }
