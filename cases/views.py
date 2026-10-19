import json
import datetime
from pathlib import Path
import hashlib
import hmac
import logging
from time import perf_counter

from django.conf import settings
from django.db.models import Count, Max, Min
from django.shortcuts import get_object_or_404, redirect, render
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.urls import reverse

from .forms import CaseSearchForm, LawyerFinderForm, PlainLanguageAssistantForm
from .models import LegalCase, ModelTrainingRun, SavedCase, SearchHistory
from .services.similarity import (
    find_similar_cases,
    get_search_diagnostics,
    load_similarity_model,
)
from .services.legal_assistant import build_plain_language_assistance
from .services.lawyer_directory import find_tamil_nadu_help, get_directory_stats
from .services.case_comparison import compare_cases
from .services.evaluation import evaluate_retrieval_benchmark


logger = logging.getLogger(__name__)


def create_query_fingerprint(query_text):
    normalized_query = " ".join(query_text.lower().split())
    return hmac.new(
        settings.SECRET_KEY.encode("utf-8"),
        normalized_query.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def sanitize_query_for_display(query_text, max_len=75):
    """
    Produce a privacy-preserving, safe, truncated display string for user interface log items.
    Prevents raw sensitive data storage while providing clear user readability.
    """
    if not query_text:
        return "Legal Search Query"
    cleaned = " ".join(query_text.strip().split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len].rstrip() + "..."


def get_statistics():
    return LegalCase.objects.aggregate(
        total_cases=Count("id"),
        first_year=Min("judgment_year"),
        last_year=Max("judgment_year"),
    )


def _get_session_key(request):
    if not request.session.session_key:
        request.session.create()
    return request.session.session_key


def _is_authenticated(request):
    user = getattr(request, "user", None)
    return bool(user and user.is_authenticated)


def _get_request_user(request):
    user = getattr(request, "user", None)
    return user if (user and user.is_authenticated) else None


def home(request):
    initial_data = {}
    if request.method == "GET" and request.GET.get("q"):
        initial_data["case_description"] = request.GET.get("q")

    form = CaseSearchForm(request.POST or None, initial=initial_data if not request.POST else None)
    statistics = get_statistics()

    if request.method == "POST" and form.is_valid():
        query_text = form.cleaned_data["case_description"]
        result_count = form.cleaned_data.get("result_count") or 5
        search_mode = form.cleaned_data.get("search_mode") or "all"
        start_year = form.cleaned_data.get("start_year")
        end_year = form.cleaned_data.get("end_year")
        legal_provision = form.cleaned_data.get("legal_provision")
        court_bench = form.cleaned_data.get("court_bench")

        applied_filters = {
            "start_year": start_year,
            "end_year": end_year,
            "legal_provision": legal_provision,
            "court_bench": court_bench,
            "search_mode": search_mode,
        }

        started_at = perf_counter()

        try:
            results = find_similar_cases(
                query_text=query_text,
                top_k=result_count,
                search_mode=search_mode,
                start_year=start_year,
                end_year=end_year,
                legal_provision=legal_provision,
                court_bench=court_bench,
            )
        except (ValueError, FileNotFoundError) as error:
            form.add_error(None, str(error))
        except Exception:
            logger.exception("Unexpected similarity-search error.")
            form.add_error(None, "Search failed. Please try again.")
        else:
            processing_time_ms = round((perf_counter() - started_at) * 1000)
            top_score = results[0]["percentage"] if results else None
            model_version = (
                results[0].get("model_version", "hybrid-tfidf-v2")
                if results
                else "hybrid-tfidf-v2"
            )

            # Pure Precedent Retrieval & Explainable Similarity Search (No Outcome Prediction)
            outcome_prediction = None

            SearchHistory.objects.create(
                query_fingerprint=create_query_fingerprint(query_text),
                sanitized_query=sanitize_query_for_display(query_text),
                search_mode=search_mode,
                query_word_count=len(query_text.split()),
                results_count=len(results),
                top_similarity_score=top_score,
                processing_time_ms=processing_time_ms,
                model_version=model_version,
                user=_get_request_user(request),
            )

            try:
                diagnostics = get_search_diagnostics(
                    query_text=query_text,
                    top_k=result_count,
                    search_mode=search_mode,
                )
            except Exception:
                diagnostics = {
                    "model_version": model_version,
                    "index_version": "Unavailable",
                    "matrix_rows": statistics.get("total_cases") or 0,
                    "feature_count": 120000,
                    "query_features": 0,
                    "candidate_pool_size": max(40, result_count * 10),
                    "search_mode": search_mode,
                    "expanded_terms": [],
                    "legal_references": [],
                    "query_word_count": len(query_text.split()),
                    "case_id_count": 0,
                }

            has_refs = bool(diagnostics.get("legal_references"))
            score_weights = {
                "document": 0.50 if has_refs else 0.58,
                "passage": 0.35 if has_refs else 0.42,
                "reference": 0.15 if has_refs else 0.00,
                "recency": 0.08 if search_mode == "current" else 0.00,
            }

            # Check saved statuses
            session_key = _get_session_key(request)
            from django.db.models import Q
            if _is_authenticated(request):
                saved_case_ids = set(
                    SavedCase.objects.filter(
                        Q(user=request.user) | Q(session_key=session_key)
                    ).values_list("legal_case_id", flat=True)
                )
            else:
                saved_case_ids = set(
                    SavedCase.objects.filter(session_key=session_key).values_list("legal_case_id", flat=True)
                )
            from .services.metadata_sanitizer import sanitize_citations
            for item in results:
                item["sanitized_citations"] = sanitize_citations(item["case"].citations)
                item["is_saved"] = item["case"].id in saved_case_ids

            return render(
                request,
                "cases/results.html",
                {
                    "form": form,
                    "query_text": query_text,
                    "results": results,
                    "statistics": statistics,
                    "processing_time_ms": processing_time_ms,
                    "model_version": model_version,
                    "search_mode": search_mode,
                    "diagnostics": diagnostics,
                    "score_weights": score_weights,
                    "outcome_prediction": outcome_prediction,
                    "applied_filters": applied_filters,
                },
            )

    return render(
        request,
        "cases/home.html",
        {
            "form": form,
            "statistics": statistics,
        },
    )


def case_detail(request, case_id):
    from .services.metadata_sanitizer import get_sanitized_case_metadata
    from django.db.models import Q
    legal_case = get_object_or_404(LegalCase, id=case_id)
    session_key = _get_session_key(request)
    if _is_authenticated(request):
        is_saved = SavedCase.objects.filter(
            Q(user=request.user) | Q(session_key=session_key), legal_case=legal_case
        ).exists()
    else:
        is_saved = SavedCase.objects.filter(
            session_key=session_key, legal_case=legal_case
        ).exists()

    sanitized_meta = get_sanitized_case_metadata(legal_case)

    return render(
        request,
        "cases/case_detail.html",
        {
            "legal_case": legal_case,
            "is_saved": is_saved,
            "meta": sanitized_meta,
            "historical_outcome": sanitized_meta["historical_outcome"],
        },
    )


def related_cases(request, case_id):
    from .services.metadata_sanitizer import get_sanitized_case_metadata
    from django.db.models import Q
    source_case = get_object_or_404(LegalCase, id=case_id)
    source_meta = get_sanitized_case_metadata(source_case)
    query_text = f"{source_case.title} {source_case.raw_text[:800]}"
    raw_results = find_similar_cases(query_text, top_k=6, search_mode="all")
    
    # Filter out current case
    related_results = [r for r in raw_results if r["case"].id != source_case.id][:5]
    
    session_key = _get_session_key(request)
    if _is_authenticated(request):
        saved_ids = set(
            SavedCase.objects.filter(
                Q(user=request.user) | Q(session_key=session_key)
            ).values_list("legal_case_id", flat=True)
        )
    else:
        saved_ids = set(
            SavedCase.objects.filter(session_key=session_key).values_list(
                "legal_case_id", flat=True
            )
        )

    for item in related_results:
        item["is_saved"] = item["case"].id in saved_ids

    return render(
        request,
        "cases/related_cases.html",
        {
            "source_case": source_case,
            "source_meta": source_meta,
            "related_results": related_results,
        },
    )



def dashboard(request):
    statistics = get_statistics()

    cases_by_year = list(
        LegalCase.objects.values("judgment_year")
        .annotate(total=Count("id"))
        .order_by("judgment_year")
    )

    extraction_statistics = list(
        LegalCase.objects.values("extraction_status")
        .annotate(total=Count("id"))
        .order_by("extraction_status")
    )

    recent_searches = SearchHistory.objects.all()[:10]
    latest_training = ModelTrainingRun.objects.filter(
        status=ModelTrainingRun.Status.COMPLETED
    ).first()

    return render(
        request,
        "cases/dashboard.html",
        {
            "statistics": statistics,
            "cases_by_year": cases_by_year,
            "extraction_statistics": extraction_statistics,
            "recent_searches": recent_searches,
            "latest_training": latest_training,
        },
    )


def _is_admin_user(request_or_user):
    user = getattr(request_or_user, "user", request_or_user)
    return bool(user.is_authenticated and (user.is_staff or user.is_superuser))


def _check_admin_access(request):
    user = getattr(request, "user", None)
    if not user or not user.is_authenticated:
        login_url = reverse("cases:admin_login")
        return redirect(f"{login_url}?next={request.path}")
    if not (user.is_staff or user.is_superuser):
        from django.contrib import messages
        messages.error(request, "Access restricted. Administrator privileges required.")
        return redirect("cases:home")
    return None


def admin_login(request):
    """Custom Admin Login View with CSRF, error handling, and redirection."""
    from django.contrib import messages
    user = getattr(request, "user", None)
    if user and user.is_authenticated and (user.is_staff or user.is_superuser):
        return redirect("cases:admin_dashboard")

    next_url = request.GET.get("next") or request.POST.get("next") or reverse("cases:admin_dashboard")
    error = None

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")

        if not username or not password:
            error = "Please enter both username and password."
        else:
            user = authenticate(request, username=username, password=password)
            if user is not None:
                if user.is_staff or user.is_superuser:
                    login(request, user)
                    messages.success(request, f"Welcome back, Administrator {user.username}!")
                    return redirect(next_url)
                else:
                    error = "Access denied. Your account lacks administrator privileges."
            else:
                error = "Invalid username or password. Please try again."

    return render(
        request,
        "cases/admin_login.html",
        {
            "error": error,
            "next": next_url,
        },
    )


def admin_logout(request):
    """Secure Admin Logout View with resilient SQLite lock and session handling."""
    from django.db import OperationalError, connection
    import time

    # Attempt standard session flush and logout with retry on lock
    logged_out = False
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logout(request)
            logged_out = True
            break
        except OperationalError as exc:
            logger.warning(
                "SQLite lock encountered during admin_logout (attempt %s/%s): %s",
                attempt + 1,
                max_retries,
                exc,
            )
            try:
                connection.close()
            except Exception:
                pass
            if attempt < max_retries - 1:
                time.sleep(0.2 * (attempt + 1))
        except Exception as exc:
            logger.error("Unexpected error during admin_logout: %s", exc)
            break

    if not logged_out:
        # Fallback: Invalidate session in-memory and drop cookie to protect security
        try:
            request.session.flush()
        except Exception:
            try:
                request.session.clear()
            except Exception:
                pass
        try:
            from django.contrib.auth import logout as auth_logout
            # Ensure user attribute is AnonymousUser
            from django.contrib.auth.models import AnonymousUser
            request.user = AnonymousUser()
        except Exception:
            pass

    messages.info(request, "You have been logged out securely.")
    response = redirect("cases:home")
    # Guarantee session cookie deletion even if DB row delete was locked
    if settings.SESSION_COOKIE_NAME in request.COOKIES:
        response.delete_cookie(
            settings.SESSION_COOKIE_NAME,
            path=settings.SESSION_COOKIE_PATH,
            domain=settings.SESSION_COOKIE_DOMAIN,
            samesite=settings.SESSION_COOKIE_SAMESITE,
        )
    return response


def research_console(request):
    """Expose transparent model and retrieval configuration for project review in Admin area."""
    access_denied = _check_admin_access(request)
    if access_denied:
        return access_denied

    statistics = get_statistics()
    model_status = {
        "available": False,
        "index_version": "Not loaded",
        "rows": 0,
        "features": 0,
    }
    try:
        _, matrix, _, case_ids, index_version = load_similarity_model()
        model_status.update({
            "available": True,
            "index_version": index_version,
            "rows": int(matrix.shape[0]),
            "features": int(matrix.shape[1]),
            "case_id_count": len(case_ids),
        })
    except Exception as error:
        model_status["error"] = str(error)

    pipeline_steps = [
        ("01", "Query Preprocessing", "Clean and normalize natural language user case description."),
        ("02", "Domain Expansion", "Map common language (e.g. detained, fired, property) to legal terminology."),
        ("03", "TF-IDF Query Vectorization", "Transform query into the 120,000-dimensional sparse feature space."),
        ("04", "Document-Level Candidate Selection", "Retrieve top candidate Supreme Court judgments using cosine similarity."),
        ("05", "Passage-Level Reranking", "Locate strongest matching paragraph excerpt within candidate judgments."),
        ("06", "Legal-Reference Overlap", "Match legal Articles, Sections, and Acts between query and case."),
        ("07", "Hybrid Reranking", "Combine Lexical, Passage, and Reference signals (0.50/0.35/0.15)."),
        ("08", "Evidence & Explanation Output", "Present percentage relevance score, score breakdown, and match reasoning."),
    ]

    return render(
        request,
        "cases/research_console.html",
        {
            "statistics": statistics,
            "model_status": model_status,
            "pipeline_steps": pipeline_steps,
            "tn_directory_stats": get_directory_stats(),
        },
    )


def admin_dashboard(request):
    """Admin Console Overview page."""
    access_denied = _check_admin_access(request)
    if access_denied:
        return access_denied

    statistics = get_statistics()
    recent_searches = SearchHistory.objects.all()[:10]
    
    return render(
        request,
        "cases/admin_dashboard.html",
        {
            "statistics": statistics,
            "recent_searches": recent_searches,
            "tn_directory_stats": get_directory_stats(),
        },
    )


def admin_corpus(request):
    """Admin Corpus Management page placeholder."""
    access_denied = _check_admin_access(request)
    if access_denied:
        return access_denied

    statistics = get_statistics()
    return render(
        request,
        "cases/admin_corpus.html",
        {
            "statistics": statistics,
        },
    )


def admin_add_case(request):
    """
    Upload Supreme Court Judgment PDF Page & Processing Endpoint.
    Handles PDF validation, text extraction, metadata parsing, classification, and duplicate checking.
    """
    access_denied = _check_admin_access(request)
    if access_denied:
        return access_denied

    if request.method == "POST" and request.FILES.get("pdf_file"):
        uploaded_file = request.FILES["pdf_file"]
        
        if not uploaded_file.name.lower().endswith(".pdf"):
            messages.error(request, "Invalid file format. Only Supreme Court PDF documents (.pdf) are supported.")
            return render(request, "cases/admin_add_case.html")

        if uploaded_file.size > 25 * 1024 * 1024:  # 25 MB
            messages.error(request, "File size exceeds maximum allowable limit (25 MB).")
            return render(request, "cases/admin_add_case.html")

        try:
            import os
            import re
            from django.db import transaction
            from .services.pdf_ingestion import extract_pdf_text, calculate_file_checksum, extract_judgment_metadata
            from .services.judgment_classifier import classify_judgment
            from .services.duplicate_detector import check_duplicate_judgment
            from .services.preprocessing import extract_legal_references

            # Extract PDF Text & Checksum
            extracted = extract_pdf_text(uploaded_file)
            checksum = calculate_file_checksum(extracted["file_bytes"])

            if extracted["requires_ocr"] or len(extracted["raw_text"]) < 50:
                messages.warning(request, "Text could not be extracted from this PDF. The document may be scanned. OCR support is not currently enabled.")
                return render(request, "cases/admin_add_case.html")

            # Extract Metadata & Provisions & Categories
            metadata = extract_judgment_metadata(extracted["raw_text"], filename=uploaded_file.name)
            detected_refs = extract_legal_references(extracted["raw_text"], maximum=15)

            acts_str = ", ".join([r for r in detected_refs if "Act" in r or "Code" in r or "Constitution" in r])
            sections_str = ", ".join([r for r in detected_refs if "Section" in r or "Article" in r])

            classification = classify_judgment(extracted["raw_text"], acts=acts_str, sections=sections_str)

            # Check Duplicates
            dup_check = check_duplicate_judgment(
                title=metadata["title"],
                citations=metadata["citations"],
                checksum=checksum,
                raw_text=extracted["raw_text"],
            )

            # Store PDF file securely in storage directory
            upload_dir = Path(settings.BASE_DIR) / "cases" / "data" / "uploaded_judgments"
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            safe_filename = re.sub(r"[^\w\.-]", "_", uploaded_file.name)
            dest_file_path = upload_dir / safe_filename
            
            with open(dest_file_path, "wb") as f_out:
                f_out.write(extracted["file_bytes"])

            # Derive readiness status and reasons
            readiness_reasons = []
            if dup_check["is_duplicate"]:
                readiness_reasons.append("Ingestion blocked because this judgment already exists in the Legal Lens database.")
            if not metadata["title"] or metadata["title_info"]["confidence"] in ("LOW", "NOT_FOUND"):
                readiness_reasons.append("Case Title is missing or has low confidence.")
            if not metadata["judgment_year"] or metadata["date_info"]["confidence"] in ("LOW", "NOT_FOUND"):
                readiness_reasons.append("Judgment Year is unverified.")
            if metadata["bench_info"]["confidence"] in ("LOW", "NOT_FOUND"):
                readiness_reasons.append("Bench / Judges not confidently detected (Optional).")
            if metadata["citation_info"]["confidence"] in ("LOW", "NOT_FOUND"):
                readiness_reasons.append("Citation not confidently detected (Optional).")

            if dup_check["is_duplicate"]:
                readiness_status = "DUPLICATE_BLOCKED"
            elif (not metadata["title"] or metadata["title_info"]["confidence"] in ("LOW", "NOT_FOUND") or not metadata["judgment_year"]):
                readiness_status = "REVIEW_REQUIRED"
            else:
                readiness_status = "VALIDATION_PASSED"

            # Store Pending Data in Session for Preview
            request.session["pending_judgment"] = {
                "filename": safe_filename,
                "file_path": str(dest_file_path),
                "checksum": checksum,
                "title": metadata["title"],
                "judgment_year": metadata["judgment_year"],
                "judgment_date": metadata["judgment_date"] if isinstance(metadata["judgment_date"], str) else (metadata["judgment_date"].isoformat() if metadata["judgment_date"] else None),
                "citations": metadata["citations"],
                "bench": metadata["bench"],
                "petitioner": metadata["petitioner"],
                "respondent": metadata["respondent"],
                "court": metadata["court"],
                "case_number": metadata["case_number"],
                "acts": acts_str,
                "legal_sections": sections_str,
                "raw_text": extracted["raw_text"],
                "cleaned_text": " ".join(extracted["raw_text"].split()),
                "page_count": extracted["page_count"],
                "word_count": extracted["word_count"],
                "char_count": extracted["char_count"],
                "primary_category": classification["primary_category"],
                "primary_confidence_pct": classification["primary_confidence_pct"],
                "primary_confidence_label": classification["primary_confidence_label"],
                "classification_evidence": classification.get("classification_evidence", []),
                "secondary_categories": classification["secondary_categories"],
                
                # Structured confidence & evidence fields
                "title_confidence": metadata["title_info"]["confidence"],
                "title_confidence_score": metadata["title_info"].get("confidence_score", 0.0),
                "title_evidence": metadata["title_info"].get("evidence", ""),
                
                "petitioner_confidence": metadata["title_info"].get("petitioner_confidence", "NOT_FOUND"),
                "petitioner_evidence": metadata["title_info"].get("evidence", ""),
                
                "respondent_confidence": metadata["title_info"].get("respondent_confidence", "NOT_FOUND"),
                "respondent_evidence": metadata["title_info"].get("evidence", ""),
                
                "citation_confidence": metadata["citation_info"]["confidence"],
                "citation_confidence_score": metadata["citation_info"].get("confidence_score", 0.0),
                "citation_evidence": metadata["citation_info"].get("evidence", ""),
                
                "date_confidence": metadata["date_info"]["confidence"],
                "date_confidence_score": metadata["date_info"].get("confidence_score", 0.0),
                "date_evidence": metadata["date_info"].get("evidence", ""),
                
                "bench_confidence": metadata["bench_info"]["confidence"],
                "bench_confidence_score": metadata["bench_info"].get("confidence_score", 0.0),
                "bench_evidence": metadata["bench_info"].get("evidence", ""),
                
                "case_number_confidence": metadata["case_number_info"]["confidence"],
                "case_number_evidence": metadata["case_number_info"].get("evidence", ""),

                "critical_confidence_low": metadata["critical_confidence_low"],
                "readiness_status": readiness_status,
                "readiness_reasons": readiness_reasons,

                # Duplicate detection tier
                "is_duplicate": dup_check["is_duplicate"],
                "duplicate_tier": dup_check.get("tier", "NO_DUPLICATE"),
                "duplicate_tier_label": dup_check.get("tier_label", "✓ NO DUPLICATE FOUND"),
                "duplicate_reason": dup_check["reason"],
                "existing_case_id": dup_check["existing_case"].id if dup_check["existing_case"] else None,
                "existing_case_title": dup_check["existing_case"].title if dup_check["existing_case"] else None,
            }

            return redirect("cases:admin_preview_judgment")

        except Exception as e:
            logger.exception("PDF Processing error")
            messages.error(request, f"Could not process uploaded PDF: {str(e)}")
            return render(request, "cases/admin_add_case.html")

    return render(request, "cases/admin_add_case.html")


def admin_preview_judgment(request):
    """
    Renders preview screen for processed judgment before saving to database.
    Allows editing missing metadata fields and confirming ingestion.
    """
    access_denied = _check_admin_access(request)
    if access_denied:
        return access_denied

    pending = request.session.get("pending_judgment")
    if not pending:
        messages.warning(request, "No pending judgment preview found. Please upload a judgment PDF.")
        return redirect("cases:admin_add_case")

    return render(
        request,
        "cases/admin_preview_judgment.html",
        {
            "pending": pending,
            "text_snippet": pending["raw_text"][:2000],
        },
    )


def admin_confirm_ingestion(request):
    """
    Permanently ingests confirmed judgment into LegalCase database within a safe transaction.
    Enforces strict backend duplicate prevention to reject ingestion if duplicate exists.
    """
    access_denied = _check_admin_access(request)
    if access_denied:
        return access_denied

    if request.method != "POST":
        return redirect("cases:admin_preview_judgment")

    pending = request.session.get("pending_judgment")
    if not pending:
        messages.error(request, "No pending judgment found to confirm.")
        return redirect("cases:admin_add_case")

    # STRICT BACKEND DUPLICATE GUARD
    checksum = pending.get("checksum")
    is_duplicate_db = False
    if checksum and LegalCase.objects.filter(file_checksum=checksum).exists():
        is_duplicate_db = True

    if pending.get("is_duplicate") or is_duplicate_db:
        logger.warning(f"Ingestion attempt for duplicate checksum {checksum} blocked by backend guard.")
        messages.error(
            request,
            "This judgment already exists in the Legal Lens database. Ingestion has been blocked to prevent duplicate records."
        )
        return redirect("cases:admin_preview_judgment")

    # Read updated metadata from POST form if edited by admin
    title = request.POST.get("title") or pending["title"]
    judgment_year = int(request.POST.get("judgment_year") or pending["judgment_year"])
    citations = request.POST.get("citations") or pending["citations"]
    bench = request.POST.get("bench") or pending["bench"]
    petitioner = request.POST.get("petitioner") or pending["petitioner"]
    respondent = request.POST.get("respondent") or pending["respondent"]
    acts = request.POST.get("acts") or pending["acts"]
    legal_sections = request.POST.get("legal_sections") or pending["legal_sections"]

    try:
        from django.db import transaction
        with transaction.atomic():
            base_filename = pending["filename"]
            final_filename = base_filename
            counter = 1
            while LegalCase.objects.filter(filename=final_filename).exists():
                final_filename = f"{base_filename.rsplit('.', 1)[0]}_{counter}.pdf"
                counter += 1

            new_case = LegalCase.objects.create(
                title=title[:500],
                filename=final_filename,
                file_path=pending["file_path"],
                judgment_year=judgment_year,
                citations=citations,
                bench=bench,
                petitioner=petitioner[:500],
                respondent=respondent[:500],
                acts=acts,
                legal_sections=legal_sections,
                raw_text=pending["raw_text"],
                cleaned_text=pending["cleaned_text"],
                page_count=pending["page_count"],
                word_count=pending["word_count"],
                extraction_status=LegalCase.ExtractionStatus.COMPLETE,
                extraction_quality=1.0,
                file_checksum=pending["checksum"] if not LegalCase.objects.filter(file_checksum=pending["checksum"]).exists() else None,
            )

        from .services.indexing import auto_index_legal_case
        index_result = auto_index_legal_case(new_case)

        del request.session["pending_judgment"]

        if index_result["is_searchable"]:
            messages.success(
                request,
                f"✓ Judgment '{new_case.title}' successfully ingested and AUTOMATICALLY INDEXED (Case ID #{new_case.id}). "
                f"Corpus size: {index_result['corpus_size']} cases | Status: SEARCHABLE ✓ (Index update time: {index_result['elapsed_seconds']}s)"
            )
        elif index_result["is_indexed"]:
            messages.warning(
                request,
                f"✓ Judgment '{new_case.title}' saved and indexed (Case ID #{new_case.id}), but post-index validation is pending."
            )
        else:
            messages.error(
                request,
                f"✓ Judgment saved to database (Case ID #{new_case.id}), but automatic search index update failed: {index_result.get('error')}. "
                f"Click '⚡ Rebuild Search Index' to retry indexing."
            )

        return redirect("cases:admin_ingestion_history")

    except Exception as error:
        logger.exception("Ingestion transaction error")
        messages.error(request, f"Failed to ingest judgment: {str(error)}")
        return redirect("cases:admin_preview_judgment")


def admin_rebuild_index(request):
    """
    Safely triggers search index / TF-IDF vector space matrix rebuild for LegalLens.
    """
    access_denied = _check_admin_access(request)
    if access_denied:
        return access_denied

    try:
        from .services.similarity import load_similarity_model
        load_similarity_model.cache_clear()
        vectorizer, matrix, matrix_T, case_ids, version = load_similarity_model()
        messages.success(
            request,
            f"✓ Search index rebuild complete! Loaded TF-IDF vector space matrix with {len(case_ids)} candidate cases."
        )
    except Exception as e:
        logger.exception("Index rebuild error")
        messages.error(request, f"Search index rebuild error: {str(e)}")

    return redirect("cases:admin_ingestion_history")


def admin_ingestion_history(request):
    """Admin Ingestion History page showing actual ingested cases log and index status."""
    access_denied = _check_admin_access(request)
    if access_denied:
        return access_denied

    ingested_cases = list(LegalCase.objects.all().order_by("-imported_at", "-id")[:50])

    try:
        from .services.similarity import load_similarity_model
        vectorizer, case_matrix, case_matrix_T, case_ids, version = load_similarity_model()
        case_ids_set = set(case_ids)
        matrix_rows = case_matrix.shape[0]
    except Exception:
        case_ids_set = set()
        matrix_rows = 0

    for c in ingested_cases:
        c.is_in_index = c.id in case_ids_set
        c.is_searchable = c.is_in_index

    return render(
        request,
        "cases/admin_ingestion_history.html",
        {
            "ingested_cases": ingested_cases,
            "corpus_count": matrix_rows,
            "db_count": LegalCase.objects.count(),
        },
    )


def case_compare(request):
    """
    Side-by-side Judgment Comparison Workstation.
    Allows comparing two Supreme Court precedents with genuine similarity calculations.
    """
    case_a_id = request.GET.get("case_a") or request.GET.get("case1")
    case_b_id = request.GET.get("case_b") or request.GET.get("case2")
    query_text = request.GET.get("query", "")

    comparison = None
    error = None
    case_a_obj = None
    case_b_obj = None

    if case_a_id:
        try:
            case_a_obj = LegalCase.objects.get(id=int(case_a_id))
        except (LegalCase.DoesNotExist, ValueError):
            case_a_obj = None

    if case_b_id:
        try:
            case_b_obj = LegalCase.objects.get(id=int(case_b_id))
        except (LegalCase.DoesNotExist, ValueError):
            case_b_obj = None

    if case_a_obj and case_b_obj:
        if case_a_obj.id == case_b_obj.id:
            alt_case = LegalCase.objects.exclude(id=case_a_obj.id).first()
            if alt_case:
                case_b_obj = alt_case
                case_b_id = str(alt_case.id)

        try:
            comparison = compare_cases(case_a_obj.id, case_b_obj.id, query_text=query_text)
        except Exception as e:
            logger.exception("Comparison error")
            error = f"Could not compare cases: {str(e)}"

    recent_cases = LegalCase.objects.all()[:50]

    return render(
        request,
        "cases/compare.html",
        {
            "comparison": comparison,
            "error": error,
            "case_a_id": case_a_id,
            "case_b_id": case_b_id,
            "case_a_obj": case_a_obj,
            "case_b_obj": case_b_obj,
            "recent_cases": recent_cases,
            "query_text": query_text,
        },
    )


def export_report(request):
    """
    Generates printable/PDF Legal Research Report for a given query and search mode.
    Includes mandatory legal disclaimer, filter-aware prediction, and genuine retrieved results.
    """
    query_text = request.GET.get("q", "").strip()
    search_mode = request.GET.get("search_mode", "all")
    top_k = int(request.GET.get("result_count", 10))

    start_year = request.GET.get("start_year")
    end_year = request.GET.get("end_year")
    legal_provision = request.GET.get("legal_provision")
    court_bench = request.GET.get("court_bench")

    try:
        start_year = int(start_year) if start_year else None
    except ValueError:
        start_year = None

    try:
        end_year = int(end_year) if end_year else None
    except ValueError:
        end_year = None

    applied_filters = {
        "start_year": start_year,
        "end_year": end_year,
        "legal_provision": legal_provision,
        "court_bench": court_bench,
        "search_mode": search_mode,
    }

    results = []
    diagnostics = {}
    outcome_prediction = None
    error_message = None

    if query_text:
        try:
            results = find_similar_cases(
                query_text=query_text,
                top_k=top_k,
                search_mode=search_mode,
                start_year=start_year,
                end_year=end_year,
                legal_provision=legal_provision,
                court_bench=court_bench,
            )
            # Pure Precedent Retrieval & Explainable Similarity Search (No Outcome Prediction)
            outcome_prediction = None
            from .services.metadata_sanitizer import sanitize_citations
            for item in results:
                item["sanitized_citations"] = sanitize_citations(item["case"].citations)
        except (ValueError, FileNotFoundError) as error:
            error_message = str(error)
            messages.warning(request, error_message)
        except Exception:
            logger.exception("Unexpected export report search error.")
            error_message = "An unexpected error occurred while generating the research report."
            messages.error(request, error_message)

    generation_time = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")

    return render(
        request,
        "cases/export_report.html",
        {
            "query_text": query_text,
            "search_mode": search_mode,
            "result_count": len(results),
            "results": results,
            "diagnostics": diagnostics,
            "outcome_prediction": outcome_prediction,
            "applied_filters": applied_filters,
            "error_message": error_message,
            "generation_time": generation_time,
        },
    )




def evaluation(request):
    eval_results = evaluate_retrieval_benchmark()
    return render(
        request,
        "cases/evaluation.html",
        {
            "eval_results": eval_results,
        },
    )


def saved_cases(request):
    session_key = _get_session_key(request)
    from django.db.models import Q
    if _is_authenticated(request):
        saved = SavedCase.objects.filter(
            Q(user=request.user) | Q(session_key=session_key)
        ).select_related("legal_case").distinct()
    else:
        saved = SavedCase.objects.filter(session_key=session_key).select_related("legal_case")
    return render(
        request,
        "cases/saved_cases.html",
        {
            "saved_cases": saved,
        },
    )


def toggle_save_case(request, case_id):
    from django.db.models import Q
    legal_case = get_object_or_404(LegalCase, id=case_id)
    session_key = _get_session_key(request)

    if _is_authenticated(request):
        existing = SavedCase.objects.filter(
            Q(user=request.user) | Q(session_key=session_key),
            legal_case=legal_case,
        ).first()
    else:
        existing = SavedCase.objects.filter(
            session_key=session_key,
            legal_case=legal_case,
        ).first()

    if existing:
        existing.delete()
        messages.info(request, f"Removed '{legal_case.title}' from saved cases.")
    else:
        SavedCase.objects.create(
            session_key=session_key,
            legal_case=legal_case,
            user=_get_request_user(request),
        )
        messages.success(request, f"Saved '{legal_case.title}' to your research collection.")

    next_url = request.META.get("HTTP_REFERER") or reverse("cases:saved_cases")
    return redirect(next_url)


def plain_language_assistant(request):
    form = PlainLanguageAssistantForm(request.POST or None)
    assistance = None

    if request.method == "POST" and form.is_valid():
        issue = form.cleaned_data["issue"]
        try:
            assistance = build_plain_language_assistance(issue)
        except (ValueError, FileNotFoundError) as error:
            form.add_error(None, str(error))
        except Exception:
            logger.exception("Unexpected plain-language assistant error.")
            form.add_error(None, "The assistant could not process this issue.")

    return render(
        request,
        "cases/assistant.html",
        {
            "form": form,
            "assistance": assistance,
        },
    )


def lawyer_finder(request):
    form = LawyerFinderForm(request.GET or None)
    directory_results = None

    if form.is_bound and form.is_valid():
        directory_results = find_tamil_nadu_help(
            district=form.cleaned_data["district"],
            specialization=form.cleaned_data.get("specialization") or "",
        )

    return render(
        request,
        "cases/lawyer_finder.html",
        {"form": form, "directory_results": directory_results},
    )


def legal_awareness(request):
    return render(
        request,
        "cases/legal_awareness.html",
    )


def about(request):
    return render(
        request,
        "cases/about.html",
    )


def annotate_benchmark(request):
    """Research-only view for human domain annotation of relevance benchmark candidate items."""
    benchmark_path = Path(settings.BASE_DIR) / "cases" / "data" / "relevance_benchmark.json"
    if not benchmark_path.exists():
        return render(
            request,
            "cases/annotate.html",
            {"error": "Benchmark file not found at cases/data/relevance_benchmark.json"},
        )

    with open(benchmark_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    queries = bench_data.get("queries", [])
    
    total_candidate_items = 0
    annotated_count = 0
    flattened_items = []

    for q_idx, q in enumerate(queries):
        for c_idx, cand in enumerate(q.get("candidates", [])):
            total_candidate_items += 1
            if cand.get("relevance") is not None:
                annotated_count += 1
            flattened_items.append({
                "item_index": len(flattened_items),
                "query_idx": q_idx,
                "candidate_idx": c_idx,
                "query_id": q.get("query_id"),
                "query_text": q.get("query"),
                "legal_category": q.get("legal_category"),
                "split_assignment": q.get("split_assignment", "train"),
                "candidate": cand,
            })

    item_idx = int(request.GET.get("item", 0))
    item_idx = max(0, min(item_idx, max(0, len(flattened_items) - 1)))

    current_item = flattened_items[item_idx] if flattened_items else None
    case_details = None

    if current_item:
        cid = current_item["candidate"]["case_id"]
        try:
            case_details = LegalCase.objects.get(id=cid)
        except LegalCase.DoesNotExist:
            case_details = None

    progress_pct = round((annotated_count / total_candidate_items) * 100, 1) if total_candidate_items > 0 else 0.0

    context = {
        "current_item": current_item,
        "case_details": case_details,
        "item_idx": item_idx,
        "total_items": len(flattened_items),
        "annotated_count": annotated_count,
        "total_candidate_items": total_candidate_items,
        "progress_pct": progress_pct,
        "annotation_status": bench_data.get("annotation_status") if annotated_count > 0 else "Human annotation pending",
        "prev_idx": max(0, item_idx - 1),
        "next_idx": min(len(flattened_items) - 1, item_idx + 1),
    }
    return render(request, "cases/annotate.html", context)


@require_POST
def save_annotation(request):
    """Save human annotation for a specific query candidate item."""
    query_id = request.POST.get("query_id")
    case_id = int(request.POST.get("case_id", 0))
    relevance_val = request.POST.get("relevance")

    if relevance_val is None or relevance_val == "":
        return JsonResponse({"status": "error", "message": "Relevance value required"}, status=400)

    relevance_val = int(relevance_val)
    reviewer_id = request.user.username if request.user.is_authenticated else "human_expert"

    benchmark_path = Path(settings.BASE_DIR) / "cases" / "data" / "relevance_benchmark.json"
    if not benchmark_path.exists():
        return JsonResponse({"status": "error", "message": "Benchmark file missing"}, status=404)

    with open(benchmark_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    updated = False
    for q in bench_data.get("queries", []):
        if q.get("query_id") == query_id:
            for cand in q.get("candidates", []):
                if cand.get("case_id") == case_id:
                    cand["relevance"] = relevance_val
                    cand["annotated_by"] = reviewer_id
                    cand["timestamp"] = datetime.datetime.now().isoformat()
                    cand["annotation_status"] = "Annotated"
                    updated = True
                    break

    if updated:
        total_cand = sum(len(q.get("candidates", [])) for q in bench_data.get("queries", []))
        ann_cand = sum(1 for q in bench_data.get("queries", []) for c in q.get("candidates", []) if c.get("relevance") is not None)

        if ann_cand == total_cand and total_cand > 0:
            bench_data["annotation_status"] = "Human verified"
        elif ann_cand > 0:
            bench_data["annotation_status"] = "In progress"
        else:
            bench_data["annotation_status"] = "Human annotation pending"

        with open(benchmark_path, "w", encoding="utf-8") as f:
            json.dump(bench_data, f, indent=2)

        return JsonResponse({
            "status": "success",
            "relevance": relevance_val,
            "annotated_count": ann_cand,
            "total_candidate_items": total_cand,
            "annotation_status": bench_data["annotation_status"],
        })

    return JsonResponse({"status": "error", "message": "Query or candidate item not found"}, status=404)


def research_status(request):
    """Research dashboard displaying dataset integrity, label counts, model status, and evaluation state."""
    benchmark_path = Path(settings.BASE_DIR) / "cases" / "data" / "relevance_benchmark.json"
    bench_data = {}
    pseudo_count = 0
    human_count = 0
    test_candidate_count = 0

    if benchmark_path.exists():
        with open(benchmark_path, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

        for q in bench_data.get("queries", []):
            is_test = q.get("split_assignment") == "test"
            cands = q.get("candidates", [])
            if is_test:
                test_candidate_count += len(cands)
            for c in cands:
                if c.get("human_label") is not None:
                    human_count += 1
                elif c.get("pseudo_label") is not None or c.get("relevance") is not None:
                    pseudo_count += 1

    model_status_info = get_supervised_model_status()
    from .services.evaluation import evaluate_retrieval_benchmark
    eval_info = evaluate_retrieval_benchmark()

    context = {
        "benchmark_data": bench_data,
        "total_queries": bench_data.get("total_queries", 50),
        "split_counts": bench_data.get("split_counts", {"train": 30, "val": 10, "test": 10}),
        "pseudo_count": pseudo_count,
        "human_count": human_count,
        "test_candidate_count": test_candidate_count,
        "model_status_info": model_status_info,
        "eval_info": eval_info,
    }
    return render(request, "cases/research_status.html", context)


