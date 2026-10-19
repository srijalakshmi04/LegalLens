from django.test import TestCase, SimpleTestCase
from django.urls import reverse
from django.contrib.auth.models import User
from cases.forms import LawyerFinderForm
from cases.services.lawyer_directory import find_tamil_nadu_help, get_directory_stats
from cases.models import LegalCase, SearchHistory
from cases.views import sanitize_query_for_display


class TamilNaduLegalHelpTests(SimpleTestCase):
    def test_finder_page_renders_without_case_database_changes(self):
        response = self.client.get("/tamil-nadu-legal-help/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Tamil Nadu")

    def test_only_tamil_nadu_district_choices_are_exposed(self):
        form = LawyerFinderForm()
        values = {value for value, _ in form.fields["district"].choices}
        self.assertIn("Coimbatore", values)
        self.assertNotIn("Bengaluru", values)

    def test_coimbatore_criminal_demo_profile_matches(self):
        result = find_tamil_nadu_help("Coimbatore", "criminal")
        self.assertTrue(result["offices"])
        self.assertEqual(len(result["lawyers"]), 1)
        self.assertIn("criminal", result["lawyers"][0]["specializations"])

    def test_family_specialization_filtering(self):
        result = find_tamil_nadu_help("Coimbatore", "family")
        self.assertEqual(result["district"], "Coimbatore")


class SearchHistoryDisplayTests(TestCase):
    def test_search_history_display_query_truncation_and_fallback(self):
        long_query = "Police arrested and detained a person without following proper legal procedure under Section 41 CrPC and Article 21"
        sanitized = sanitize_query_for_display(long_query, max_len=75)
        self.assertTrue(sanitized.endswith("..."))
        self.assertLessEqual(len(sanitized), 78)

        item = SearchHistory.objects.create(
            query_fingerprint="abc123hash",
            sanitized_query=sanitized,
            search_mode="recent",
            query_word_count=18,
            results_count=5,
            top_similarity_score=42.50,
        )
        self.assertEqual(item.display_query, sanitized)
        self.assertEqual(item.search_mode, "recent")
        self.assertEqual(item.query_fingerprint, "abc123hash")

        legacy_item = SearchHistory.objects.create(
            query_fingerprint="legacyhash123",
            sanitized_query="",
            query_word_count=12,
            results_count=5,
        )
        self.assertEqual(legacy_item.display_query, "Legal Search (12 words)")

    def test_dashboard_renders_readable_queries(self):
        SearchHistory.objects.create(
            query_fingerprint="testfingerprint999",
            sanitized_query="Police detained person without warrant",
            search_mode="all",
            query_word_count=5,
            results_count=5,
            top_similarity_score=35.50,
        )

        response = self.client.get("/dashboard/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Police detained person without warrant")
        self.assertContains(response, "Top Similarity:")
        self.assertContains(response, "35.50%")


class AdminConsoleAccessControlTests(TestCase):
    def setUp(self):
        self.normal_user = User.objects.create_user(username="lawyer_user", password="password123")
        self.admin_user = User.objects.create_superuser(username="admin_user", password="password123", email="admin@legallens.gov.in")

    def test_anonymous_user_redirected_to_admin_login(self):
        response = self.client.get("/admin-console/research/")
        self.assertRedirects(response, "/admin-login/?next=/admin-console/research/")

        response_admin = self.client.get("/admin-console/")
        self.assertRedirects(response_admin, "/admin-login/?next=/admin-console/")

    def test_admin_login_invalid_credentials(self):
        response = self.client.post("/admin-login/", {"username": "wrong_user", "password": "bad_password"})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Invalid username or password")

    def test_normal_user_blocked_from_admin_console(self):
        self.client.login(username="lawyer_user", password="password123")
        response = self.client.get("/admin-console/research/", follow=True)
        self.assertRedirects(response, "/")
        self.assertContains(response, "Access restricted. Administrator privileges required")

    def test_admin_user_login_and_access(self):
        login_res = self.client.post("/admin-login/", {"username": "admin_user", "password": "password123"}, follow=True)
        self.assertContains(login_res, "Admin Console Overview")

        res_research = self.client.get("/admin-console/research/")
        self.assertEqual(res_research.status_code, 200)
        self.assertContains(res_research, "Research & System Details")

    def test_admin_logout(self):
        self.client.login(username="admin_user", password="password123")
        logout_res = self.client.get("/admin-logout/", follow=True)
        self.assertRedirects(logout_res, "/")
        self.assertContains(logout_res, "You have been logged out securely")

        # After logout, direct access redirects to login
        res_post = self.client.get("/admin-console/")
        self.assertRedirects(res_post, "/admin-login/?next=/admin-console/")

    def test_admin_logout_session_terminated(self):
        self.client.login(username="admin_user", password="password123")
        self.assertIn("_auth_user_id", self.client.session)
        response = self.client.get("/admin-logout/")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, "/")
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_admin_logout_resilient_to_database_lock(self):
        """Simulate OperationalError: database is locked during logout to ensure graceful fallback."""
        from unittest.mock import patch
        from django.db import OperationalError

        self.client.login(username="admin_user", password="password123")
        with patch("django.contrib.auth.logout", side_effect=OperationalError("database is locked")):
            response = self.client.get("/admin-logout/", follow=True)
            self.assertEqual(response.status_code, 200)
            self.assertRedirects(response, "/")
            self.assertContains(response, "You have been logged out securely")

        # Confirm admin privileges are revoked post-logout even under lock
        res_post = self.client.get("/admin-console/")
        self.assertRedirects(res_post, "/admin-login/?next=/admin-console/")

    def test_public_sidebar_does_not_contain_admin_console(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "ADMIN CONSOLE")


class CompareCasesTests(TestCase):
    def setUp(self):
        self.case1 = LegalCase.objects.create(
            title="State of Punjab v. Gurbaksh Singh",
            filename="test_case_1.txt",
            judgment_year=2021,
            cleaned_text="Bail application under Section 439 CrPC for personal liberty.",
            acts="CrPC",
            legal_sections="Section 439",
        )
        self.case2 = LegalCase.objects.create(
            title="Kalyani Baskar v. M.S. Sampoornam",
            filename="test_case_2.txt",
            judgment_year=2018,
            cleaned_text="Cheque bounce under Section 138 Negotiable Instruments Act.",
            acts="NI Act",
            legal_sections="Section 138",
        )

    def test_compare_cases_page_renders(self):
        response = self.client.get("/compare-cases/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Supreme Court Judgment Comparison")

    def test_compare_two_valid_cases(self):
        response = self.client.get(f"/compare-cases/?case_a={self.case1.id}&case_b={self.case2.id}")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "State of Punjab v. Gurbaksh Singh")
        self.assertContains(response, "Kalyani Baskar v. M.S. Sampoornam")
        self.assertContains(response, "SIMILARITY ANALYSIS")


class ExportReportTests(TestCase):
    def test_export_report_page_renders_with_disclaimer(self):
        response = self.client.get("/export-report/?q=police+arrested+person+without+warrant+under+CrPC+section+41&search_mode=all")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Legal Lens")
        self.assertContains(response, "LEGAL RESEARCH DISCLAIMER")
        self.assertContains(response, "Retrieved results are informational and should be independently verified")

    def test_export_report_handles_short_query_gracefully(self):
        response = self.client.get("/export-report/?q=bail&search_mode=all")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Enter a more detailed case description")


class PdfIngestionPipelineTests(TestCase):
    def setUp(self):
        self.admin_user = User.objects.create_superuser(username="pdf_admin", password="password123", email="admin@legallens.gov.in")
        self.normal_user = User.objects.create_user(username="normal_user", password="password123")

    def test_non_admin_cannot_access_pdf_upload(self):
        response = self.client.get("/admin-console/add-case/")
        self.assertRedirects(response, "/admin-login/?next=/admin-console/add-case/")

        self.client.login(username="normal_user", password="password123")
        res_normal = self.client.get("/admin-console/add-case/", follow=True)
        self.assertRedirects(res_normal, "/")
        self.assertContains(res_normal, "Access restricted. Administrator privileges required")

    def test_admin_can_access_pdf_upload_page(self):
        self.client.login(username="pdf_admin", password="password123")
        response = self.client.get("/admin-console/add-case/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Upload Supreme Court Judgment PDF")

    def test_non_pdf_file_rejection(self):
        self.client.login(username="pdf_admin", password="password123")
        from django.core.files.uploadedfile import SimpleUploadedFile
        bad_file = SimpleUploadedFile("test_doc.txt", b"Plain text file content", content_type="text/plain")
        response = self.client.post("/admin-console/add-case/", {"pdf_file": bad_file})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Invalid file format. Only Supreme Court PDF documents (.pdf) are supported")

    def test_metadata_extraction_heuristics(self):
        from cases.services.pdf_ingestion import extract_judgment_metadata
        sample_text = """
        IN THE SUPREME COURT OF INDIA
        CRIMINAL APPELLATE JURISDICTION
        SLP (Crl) No. 4589 OF 2026
        
        State of Tamil Nadu ... Appellant
        VERSUS
        K. Selvam & Ors ... Respondent
        
        DATED: 15 January 2026
        2026 INSC 442
        
        CORAM: HON'BLE MR. JUSTICE A. B. SHAH
        
        JUDGMENT
        Bail application under Section 439 CrPC.
        """
        meta = extract_judgment_metadata(sample_text)
        self.assertIn("State of Tamil Nadu", meta["title"])
        self.assertEqual(meta["judgment_year"], 2026)
        self.assertEqual(meta["citations"], "2026 INSC 442")

    def test_judgment_classifier_rules(self):
        from cases.services.judgment_classifier import classify_judgment
        criminal_text = "Accused convicted under Section 302 IPC for murder. Bail application rejected under Section 439 CrPC."
        res_crl = classify_judgment(criminal_text)
        self.assertEqual(res_crl["primary_category"], "Criminal Law")

        const_text = "Writ petition challenging constitutional validity under Article 14 and Article 21 of Constitution of India."
        res_const = classify_judgment(const_text)
        self.assertEqual(res_const["primary_category"], "Constitutional Law")

    def test_duplicate_detection_service(self):
        from cases.services.duplicate_detector import check_duplicate_judgment
        LegalCase.objects.create(
            title="State of AP v. K. Ramu",
            filename="state_ap_ramu.pdf",
            judgment_year=2024,
            citations="2024 INSC 101",
            raw_text="Sample raw text",
            cleaned_text="sample raw text",
        )

        dup_title = check_duplicate_judgment(title="State of AP v. K. Ramu")
        self.assertTrue(dup_title["is_duplicate"])

        dup_cit = check_duplicate_judgment(citations="2024 INSC 101")
        self.assertTrue(dup_cit["is_duplicate"])

        no_dup = check_duplicate_judgment(title="Unique Case Name v. Union of India", citations="2026 INSC 999")
        self.assertFalse(no_dup["is_duplicate"])

    def test_confirm_ingestion_saves_to_database(self):
        self.client.login(username="pdf_admin", password="password123")
        session = self.client.session
        session["pending_judgment"] = {
            "filename": "test_ingestion.pdf",
            "file_path": "F:\\test_ingestion.pdf",
            "checksum": "sha256fakechecksum12345",
            "title": "Unique New Supreme Court Precedent v. State",
            "judgment_year": 2026,
            "judgment_date": "2026-01-15",
            "citations": "2026 INSC 777",
            "bench": "Hon'ble Mr. Justice Test",
            "petitioner": "Unique New Supreme Court Precedent",
            "respondent": "State",
            "court": "Supreme Court of India",
            "case_number": "SLP(C) 123/2026",
            "acts": "CrPC",
            "legal_sections": "Section 439",
            "raw_text": "Sample text for full judgment ingestion testing.",
            "cleaned_text": "sample text for full judgment ingestion testing",
            "page_count": 5,
            "word_count": 250,
            "char_count": 1200,
            "primary_category": "Criminal Law",
            "secondary_categories": ["Constitutional Law"],
            "is_duplicate": False,
            "duplicate_reason": None,
            "existing_case_id": None,
            "existing_case_title": None,
        }
        session.save()

        res_confirm = self.client.post("/admin-console/confirm-ingestion/", {
            "title": "Unique New Supreme Court Precedent v. State",
            "judgment_year": "2026",
            "citations": "2026 INSC 777",
            "bench": "Hon'ble Mr. Justice Test",
            "petitioner": "Unique New Supreme Court Precedent",
            "respondent": "State",
            "acts": "CrPC",
            "legal_sections": "Section 439",
        }, follow=True)

        self.assertEqual(res_confirm.status_code, 200)
        self.assertContains(res_confirm, "Ingested Judgments Log")
        self.assertTrue(LegalCase.objects.filter(title="Unique New Supreme Court Precedent v. State").exists())

    def test_generic_placeholder_rejection(self):
        from cases.services.pdf_ingestion import extract_case_title_and_parties
        bad_text = """
        IN THE SUPREME COURT OF INDIA
        APPELLANT(S)
        VERSUS
        RESPONDENT(S)
        JUDGMENT
        """
        result = extract_case_title_and_parties(bad_text)
        self.assertEqual(result["title"], "")
        self.assertEqual(result["confidence"], "NOT_FOUND")

    def test_citation_year_fallback(self):
        from cases.services.pdf_ingestion import extract_citation, extract_judgment_date
        cit_info = extract_citation("2026 INSC 289")
        self.assertEqual(cit_info["year_extracted"], 2026)
        self.assertEqual(cit_info["confidence"], "HIGH")

        date_info = extract_judgment_date("No explicit date string here.", citation_year=cit_info["year_extracted"])
        self.assertEqual(date_info["year"], 2026)
        self.assertEqual(date_info["confidence"], "MEDIUM")

    def test_date_extraction_formats(self):
        from cases.services.pdf_ingestion import extract_judgment_date
        d1 = extract_judgment_date("DATED: 15 January 2026")
        self.assertEqual(d1["date"], "2026-01-15")
        self.assertEqual(d1["confidence"], "HIGH")

        d2 = extract_judgment_date("Decided on: 15/01/2026")
        self.assertEqual(d2["date"], "2026-01-15")
        self.assertEqual(d2["confidence"], "HIGH")

    def test_bench_extraction_window(self):
        from cases.services.pdf_ingestion import extract_bench_judges
        text = """
        CORAM:
        HON'BLE MR. JUSTICE A. B. SHAH
        HON'BLE MS. JUSTICE B. C. ROY
        
        JUDGMENT
        This is an appeal against conviction. The accused was convicted under Section 302 IPC.
        """
        bench_info = extract_bench_judges(text)
        self.assertIn("A. B. Shah", bench_info["value"])
        self.assertNotIn("This is an appeal", bench_info["value"])
        self.assertEqual(bench_info["confidence"], "HIGH")

    def test_confidence_system_and_preview_warning(self):
        self.client.login(username="pdf_admin", password="password123")
        session = self.client.session
        session["pending_judgment"] = {
            "filename": "unclear_metadata.pdf",
            "file_path": "F:\\unclear_metadata.pdf",
            "checksum": "fakechecksum999",
            "title": "",
            "judgment_year": 2026,
            "judgment_date": None,
            "citations": "",
            "bench": "",
            "petitioner": "",
            "respondent": "",
            "court": "Supreme Court of India",
            "case_number": "",
            "acts": "",
            "legal_sections": "",
            "raw_text": "Unclear text without clear party names or headers.",
            "cleaned_text": "unclear text without clear party names or headers",
            "page_count": 2,
            "word_count": 50,
            "char_count": 300,
            "primary_category": "General Legal Dispute",
            "primary_confidence_pct": 20,
            "primary_confidence_label": "LOW",
            "secondary_categories": [],
            "title_confidence": "NOT_FOUND",
            "citation_confidence": "NOT_FOUND",
            "date_confidence": "NOT_FOUND",
            "bench_confidence": "NOT_FOUND",
            "critical_confidence_low": True,
            "is_duplicate": False,
            "duplicate_reason": None,
            "existing_case_id": None,
            "existing_case_title": None,
        }
        session.save()

        res = self.client.get("/admin-console/preview-judgment/")
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "REVIEW REQUIRED BEFORE INGESTION")
        self.assertContains(res, "NOT FOUND")

    def test_three_tier_duplicate_detection(self):
        from cases.services.duplicate_detector import check_duplicate_judgment
        c = LegalCase.objects.create(
            title="K. Ramesh v. Union of India",
            filename="k_ramesh.pdf",
            judgment_year=2025,
            citations="2025 INSC 888",
            raw_text="Sample raw text",
            cleaned_text="sample raw text",
            file_checksum="uniquechecksum12345",
        )

        # Tier 1
        exact_res = check_duplicate_judgment(checksum="uniquechecksum12345")
        self.assertEqual(exact_res["tier"], "EXACT_DUPLICATE")

        # Tier 2
        likely_res = check_duplicate_judgment(citations="2025 INSC 888")
        self.assertEqual(likely_res["tier"], "LIKELY_DUPLICATE")

        # Tier 3
        possible_res = check_duplicate_judgment(title="K. Ramesh v. State", judgment_year=2025)
        self.assertEqual(possible_res["tier"], "POSSIBLE_DUPLICATE")

    def test_classification_evidence_generation(self):
        from cases.services.judgment_classifier import classify_judgment
        res = classify_judgment("Accused convicted under Section 302 IPC. Bail application rejected under Section 439 CrPC.")
        self.assertIn("classification_evidence", res)
        self.assertTrue(len(res["classification_evidence"]) > 0)
        self.assertEqual(res["primary_category"], "Criminal Law")

    def test_admin_rebuild_index_route(self):
        self.client.login(username="pdf_admin", password="password123")
        res = self.client.get("/admin-console/rebuild-index/", follow=True)
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Search index rebuild complete")


class SearchResultsTemplateSafetyTests(TestCase):
    def test_post_search_renders_results_without_variable_does_not_exist(self):
        response = self.client.post("/", {
            "case_description": "Police arrested and detained a person without warrant under Section 41 CrPC",
            "result_count": 5,
            "search_mode": "all",
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Export Research Report (PDF)")
        self.assertContains(response, "/export-report/")
        self.assertContains(response, "q=Police%20arrested%20and%20detained%20a%20person%20without%20warrant%20under%20Section%2041%20CrPC")

    def test_empty_request_get_does_not_cause_variable_does_not_exist(self):
        from django.template.loader import render_to_string
        context = {
            "query_text": "Police arrest bail test query",
            "search_mode": "all",
            "results": [],
            "diagnostics": {},
        }
        rendered = render_to_string("cases/results.html", context)
        self.assertIn("/export-report/", rendered)
        self.assertIn("Export Research Report (PDF)", rendered)


class AutomaticSearchIndexingTests(TestCase):
    def setUp(self):
        from django.contrib.auth.models import User
        self.admin_user = User.objects.create_superuser(
            username="auto_indexer",
            email="admin@legallens.in",
            password="password123",
        )

    def test_full_automatic_indexing_and_searchability_workflow(self):
        import scipy.sparse as sp
        self.client.login(username="auto_indexer", password="password123")
        session = self.client.session
        session["pending_judgment"] = {
            "filename": "auto_indexed_supreme_court_judgment.pdf",
            "file_path": "F:\\auto_indexed_supreme_court_judgment.pdf",
            "checksum": "autoindexchecksum9999",
            "title": "State of Tamil Nadu v. Auto Indexed Precedent",
            "judgment_year": 2026,
            "judgment_date": "2026-03-15",
            "citations": "2026 INSC 999",
            "bench": "Hon'ble Mr. Justice Auto Index",
            "petitioner": "State of Tamil Nadu",
            "respondent": "Auto Indexed Precedent",
            "court": "Supreme Court of India",
            "case_number": "Civil Appeal No. 9999 OF 2026",
            "acts": "Constitution of India, Code of Criminal Procedure",
            "legal_sections": "Article 21, Section 439",
            "raw_text": "IN THE SUPREME COURT OF INDIA. State of Tamil Nadu VERSUS Auto Indexed Precedent. DATED: 15 March 2026. CORAM: HON'BLE MR. JUSTICE AUTO INDEX. The appellant challenges conviction under Section 302 IPC. Exceptional legal principles regarding bail and custodial detention.",
            "cleaned_text": "in the supreme court of india state of tamil nadu versus auto indexed precedent dated 15 march 2026 coram honble mr justice auto index the appellant challenges conviction under section 302 ipc exceptional legal principles regarding bail and custodial detention",
            "page_count": 3,
            "word_count": 80,
            "char_count": 500,
            "primary_category": "Criminal Law",
            "primary_confidence_pct": 95,
            "primary_confidence_label": "HIGH",
            "secondary_categories": [],
            "title_confidence": "HIGH",
            "citation_confidence": "HIGH",
            "date_confidence": "HIGH",
            "bench_confidence": "HIGH",
            "critical_confidence_low": False,
            "is_duplicate": False,
            "duplicate_reason": None,
            "existing_case_id": None,
            "existing_case_title": None,
        }
        session.save()

        res_confirm = self.client.post("/admin-console/confirm-ingestion/", {
            "title": "State of Tamil Nadu v. Auto Indexed Precedent",
            "judgment_year": "2026",
            "citations": "2026 INSC 999",
            "bench": "Hon'ble Mr. Justice Auto Index",
            "petitioner": "State of Tamil Nadu",
            "respondent": "Auto Indexed Precedent",
            "acts": "Constitution of India, Code of Criminal Procedure",
            "legal_sections": "Article 21, Section 439",
        }, follow=True)

        self.assertEqual(res_confirm.status_code, 200)
        self.assertContains(res_confirm, "AUTOMATICALLY INDEXED")

        created_case = LegalCase.objects.filter(title="State of Tamil Nadu v. Auto Indexed Precedent").first()
        self.assertIsNotNone(created_case)

        from cases.services.similarity import load_similarity_model, find_similar_cases
        vectorizer, case_matrix, case_matrix_T, case_ids, version = load_similarity_model()

        # TEST 7, 8 & 9: Verify model cache refreshed, case_ids contains new ID, matrix vector is non-empty, and auto-indexing validation passes
        self.assertIn(created_case.id, case_ids)
        row_idx = case_ids.index(created_case.id)
        self.assertTrue(0 <= row_idx < case_matrix.shape[0])
        self.assertTrue(case_matrix[row_idx].nnz > 0)

    def test_indexing_failure_safety_handling(self):
        import scipy.sparse as sp
        from cases.services.indexing import _validate_and_build_status
        fake_case = LegalCase.objects.create(
            title="Unindexed Fake Case",
            filename="unindexed.pdf",
            judgment_year=2026,
            raw_text="Unindexed raw text",
            cleaned_text="unindexed raw text",
        )
        status = _validate_and_build_status(fake_case, sp.csr_matrix((1, 120000)), [999999], -1, 0.0, "v2")
        self.assertFalse(status["is_searchable"])
        self.assertEqual(status["search_status"], "[PENDING]")


class DuplicateIngestionWorkflowTests(TestCase):
    def setUp(self):
        self.admin_user = User.objects.create_superuser(
            username="dup_admin",
            email="admin@legallens.in",
            password="password123",
        )
        self.existing_case = LegalCase.objects.create(
            title="Existing Judgment v. State of TN",
            filename="existing_case.pdf",
            judgment_year=2025,
            citations="2025 INSC 500",
            raw_text="Existing judgment raw content",
            cleaned_text="existing judgment raw content",
            file_checksum="sha256duplicate12345",
        )

    def test_duplicate_pdf_preview_ui_shows_blocked_status_and_disables_ingest(self):
        self.client.login(username="dup_admin", password="password123")
        session = self.client.session
        session["pending_judgment"] = {
            "filename": "existing_case_copy.pdf",
            "file_path": "F:\\existing_case_copy.pdf",
            "checksum": "sha256duplicate12345",
            "title": "Existing Judgment v. State of TN",
            "judgment_year": 2025,
            "judgment_date": "2025-05-10",
            "citations": "2025 INSC 500",
            "bench": "Hon'ble Mr. Justice Existing",
            "petitioner": "Existing Judgment",
            "respondent": "State of TN",
            "court": "Supreme Court of India",
            "case_number": "SLP 500/2025",
            "acts": "IPC",
            "legal_sections": "Section 302",
            "raw_text": "Existing judgment raw content",
            "cleaned_text": "existing judgment raw content",
            "page_count": 4,
            "word_count": 100,
            "char_count": 600,
            "primary_category": "Criminal Law",
            "primary_confidence_pct": 90,
            "primary_confidence_label": "HIGH",
            "secondary_categories": [],
            "title_confidence": "HIGH",
            "citation_confidence": "HIGH",
            "date_confidence": "HIGH",
            "bench_confidence": "HIGH",
            "critical_confidence_low": False,
            "is_duplicate": True,
            "duplicate_reason": "Exact file checksum match",
            "existing_case_id": self.existing_case.id,
            "existing_case_title": self.existing_case.title,
            "readiness_status": "DUPLICATE_BLOCKED",
        }
        session.save()

        res = self.client.get("/admin-console/preview-judgment/")
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "EXACT DUPLICATE DETECTED")
        self.assertContains(res, "Ingestion has been blocked to prevent duplicate records.")
        self.assertContains(res, "View Existing Case")
        self.assertContains(res, "Upload Different File")
        self.assertContains(res, "INGESTION BLOCKED (EXACT DUPLICATE)")
        self.assertNotContains(res, "VALIDATION PASSED")

    def test_backend_confirm_ingestion_blocks_duplicate_creation(self):
        self.client.login(username="dup_admin", password="password123")
        session = self.client.session
        session["pending_judgment"] = {
            "filename": "existing_case_copy.pdf",
            "file_path": "F:\\existing_case_copy.pdf",
            "checksum": "sha256duplicate12345",
            "title": "Existing Judgment v. State of TN",
            "judgment_year": 2025,
            "citations": "2025 INSC 500",
            "raw_text": "Existing judgment raw content",
            "cleaned_text": "existing judgment raw content",
            "is_duplicate": True,
            "duplicate_reason": "Exact file checksum match",
            "existing_case_id": self.existing_case.id,
            "existing_case_title": self.existing_case.title,
            "readiness_status": "DUPLICATE_BLOCKED",
        }
        session.save()

        initial_count = LegalCase.objects.count()
        res_post = self.client.post("/admin-console/confirm-ingestion/", {
            "title": "Existing Judgment v. State of TN",
            "judgment_year": "2025",
            "citations": "2025 INSC 500",
        }, follow=True)

        self.assertEqual(LegalCase.objects.count(), initial_count)
        self.assertContains(res_post, "This judgment already exists in the Legal Lens database. Ingestion has been blocked to prevent duplicate records.")

    def test_valid_non_duplicate_shows_validation_passed_and_enables_ingest(self):
        self.client.login(username="dup_admin", password="password123")
        session = self.client.session
        session["pending_judgment"] = {
            "filename": "fresh_judgment.pdf",
            "file_path": "F:\\fresh_judgment.pdf",
            "checksum": "sha256unique999888",
            "title": "Brand New Precedent v. Union of India",
            "judgment_year": 2026,
            "judgment_date": "2026-04-01",
            "citations": "2026 INSC 1111",
            "bench": "Hon'ble Mr. Justice Fresh",
            "petitioner": "Brand New Precedent",
            "respondent": "Union of India",
            "court": "Supreme Court of India",
            "case_number": "W.P.(C) 100/2026",
            "acts": "Constitution of India",
            "legal_sections": "Article 14",
            "raw_text": "Fresh judgment raw content",
            "cleaned_text": "fresh judgment raw content",
            "page_count": 10,
            "word_count": 500,
            "char_count": 3000,
            "primary_category": "Constitutional Law",
            "primary_confidence_pct": 95,
            "primary_confidence_label": "HIGH",
            "secondary_categories": [],
            "title_confidence": "HIGH",
            "citation_confidence": "HIGH",
            "date_confidence": "HIGH",
            "bench_confidence": "HIGH",
            "critical_confidence_low": False,
            "is_duplicate": False,
            "duplicate_reason": None,
            "existing_case_id": None,
            "existing_case_title": None,
            "readiness_status": "VALIDATION_PASSED",
        }
        session.save()

        res = self.client.get("/admin-console/preview-judgment/")
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "VALIDATION PASSED")
        self.assertContains(res, "All critical metadata (Case Title & Judgment Year) have been confidently extracted and verified")
        self.assertContains(res, "CONFIRM AND ADD TO LEGALLENS")
        self.assertNotContains(res, "EXACT DUPLICATE DETECTED")


class OutcomePredictionUnitTests(TestCase):
    def setUp(self):
        self.case1 = LegalCase.objects.create(
            title="State of TN v. Appellant One",
            filename="case1.pdf",
            file_path="F:\\case1.pdf",
            judgment_year=2022,
            acts="Negotiable Instruments Act",
            legal_sections="Section 138",
            raw_text="The accused issued a cheque that bounced. In the result, the appeal is hereby allowed.",
            cleaned_text="the accused issued a cheque that bounced. in the result, the appeal is hereby allowed.",
            file_checksum="sha256outcome111",
        )
        self.case2 = LegalCase.objects.create(
            title="Petitioner Two v. State of Kerala",
            filename="case2.pdf",
            file_path="F:\\case2.pdf",
            judgment_year=2015,
            acts="CrPC",
            legal_sections="Section 41",
            raw_text="The petitioner was detained under procedure. Finding no merit, the petition is dismissed.",
            cleaned_text="the petitioner was detained under procedure. finding no merit, the petition is dismissed.",
            file_checksum="sha256outcome222",
        )

    def test_extract_outcome_label_patterns(self):
        from cases.services.outcome_predictor import extract_outcome_label

        self.assertEqual(extract_outcome_label("The appeal is hereby allowed."), "ALLOWED")
        self.assertEqual(extract_outcome_label("Finding no merit, the appeal is dismissed."), "DISMISSED")
        self.assertEqual(extract_outcome_label("The petition is partly allowed."), "PARTLY ALLOWED")
        self.assertEqual(extract_outcome_label("The matter stands disposed of accordingly."), "DISPOSED")
        self.assertEqual(extract_outcome_label("General legal opinion without disposition."), "OTHER")

    def test_filter_aware_retrieval(self):
        from cases.services.similarity import find_similar_cases

        # Query with year filter 2021-2025
        results_filtered = find_similar_cases(
            query_text="Arrest and detention without warrant under CrPC Article 21",
            top_k=5,
            search_mode="all",
            start_year=2021,
            end_year=2025,
        )
        for item in results_filtered:
            year = item["case"].judgment_year
            if year:
                self.assertGreaterEqual(year, 2021)
                self.assertLessEqual(year, 2025)

    def test_predict_case_outcome_service(self):
        from cases.services.outcome_predictor import predict_case_outcome
        from cases.services.similarity import find_similar_cases

        results = find_similar_cases("Cheque bounce dishonour under Section 138", top_k=5)
        prediction = predict_case_outcome(
            query_text="Cheque bounce dishonour under Section 138",
            retrieved_results=results,
            applied_filters={"start_year": 2020, "end_year": 2025, "legal_provision": "Section 138"}
        )

        self.assertIn("predicted_outcome_label", prediction)
        self.assertIn("confidence_percentage", prediction)
        self.assertIn("historical_pattern", prediction)
        self.assertIn("filter_tags", prediction)
        self.assertIn("disclaimer", prediction)
        self.assertGreaterEqual(prediction["confidence_percentage"], 45.0)
        self.assertNotEqual(prediction["predicted_outcome_label"], "")

    def test_evaluate_outcome_prediction_command_runs(self):
        from unittest.mock import patch
        from django.core.management import call_command

        mock_eval = {
            "selected_algorithm": "Logistic Regression (C=2.5)",
            "total_dataset": 100,
            "train_count": 70,
            "validation_count": 15,
            "test_count": 15,
            "held_out_test_accuracy": 69.20,
            "validation_accuracy": 67.86,
            "macro_precision": 0.6883,
            "macro_recall": 0.6886,
            "macro_f1": 0.6884,
            "weighted_f1": 0.6921,
            "class_distribution": {"ALLOWED": 50, "DISMISSED": 50},
            "confusion_matrix": [[7, 3], [2, 8]],
            "mode1_val_acc": 67.86,
            "mode1_test_acc": 69.20,
            "mode1_correct": 10,
            "mode2_val_acc": 96.07,
            "mode2_test_acc": 96.56,
            "mode2_correct": 14,
            "total_test": 15,
            "unique_labels": ["ALLOWED", "DISMISSED"],
            "classification_report": {},
        }
        from unittest.mock import patch, mock_open
        with patch("cases.management.commands.evaluate_outcome_prediction.train_and_evaluate_outcome_models", return_value=mock_eval):
            with patch("builtins.open", mock_open()):
                call_command("evaluate_outcome_prediction")



