from django.test import TestCase
from cases.models import LegalCase
from cases.services.similarity import (
    find_similar_cases,
    load_similarity_model,
    clean_text,
    expand_common_legal_language,
)
from sklearn.metrics.pairwise import linear_kernel

class SearchModeAndSimilarityTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        vectorizer, case_matrix, case_matrix_T, case_ids, _ = load_similarity_model()
        q = expand_common_legal_language(clean_text("police arrest detention article 21 section 41"))
        qv = vectorizer.transform([q])
        doc_scores = linear_kernel(qv, case_matrix).ravel()

        top_indices = [i for i in doc_scores.argsort()[::-1][:50] if doc_scores[i] > 0]
        for idx, i in enumerate(top_indices):
            cid = case_ids[i]
            year = 2022 if idx % 2 == 0 else 1985
            LegalCase.objects.get_or_create(
                id=cid,
                defaults={
                    "filename": f"test_case_{cid}_{idx}.pdf",
                    "title": f"Test Case {cid}",
                    "judgment_year": year,
                    "cleaned_text": "The police arrested and detained a person without following proper legal procedure Article 21 Section 41 CrPC.",
                    "raw_text": "The police arrested and detained a person without following proper legal procedure Article 21 Section 41 CrPC.",
                }
            )

    def test_recent_search_mode_enforces_2021_onward_rule(self):
        query = "police arrest detention article 21 section 41"
        results = find_similar_cases(query, top_k=5, search_mode="recent")
        self.assertTrue(len(results) > 0, "No results returned for Recent mode")
        for item in results:
            year = item["case"].judgment_year or 0
            self.assertGreaterEqual(year, 2021, f"Judgment year {year} is less than mandatory 2021 threshold in Recent mode")

    def test_all_cases_search_mode_returns_results(self):
        query = "police arrest detention article 21"
        results = find_similar_cases(query, top_k=5, search_mode="all")
        self.assertTrue(len(results) > 0, "No results returned for All Cases mode")

    def test_predicted_similarity_relevance_tier_presence(self):
        query = "police arrest detention article 21"
        results = find_similar_cases(query, top_k=3, search_mode="all")
        for item in results:
            self.assertIn("percentage", item)
            self.assertIn("relevance_tier", item)
            self.assertIn(item["relevance_tier"], ["HIGH", "MODERATE", "LOW"])
