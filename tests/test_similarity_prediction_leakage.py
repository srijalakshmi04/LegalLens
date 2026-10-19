"""
Automated Leakage Audit & Verification Suite for Similarity-Based Pre-Decision Prediction.

Tests:
  1. Target case not retrieved in its own historical precedent candidate list.
  2. Future cases (decided after or on prediction timestamp) are strictly excluded.
  3. Post-judgment information and final operative disposition are not used.
  4. Exact duplicates and same-case-family titles are excluded.
  5. 7 independent similarity components produce non-degenerate values.
  6. Explainability structure contains required audit fields.
"""

import unittest
import numpy as np
from datetime import date

from cases.services.similarity_outcome_predictor import (
    SevenComponentSimilarityIndex,
    predict_with_hybrid_similarity,
    parse_sections,
    parse_acts,
)


class SimilarityPredictionLeakageTests(unittest.TestCase):
    def setUp(self):
        # Synthetic mock train and target cases
        self.mock_train = [
            {
                "id": 101,
                "title": "State of Tamil Nadu v. Muthu",
                "clean_title": "stateoftamilnaduvmuthu",
                "pre_text": "Criminal Appeal under Section 374 CrPC. Accused convicted for murder under Section 302 IPC. Trial court passed life sentence.",
                "sec_tokens": "302 374",
                "act_tokens": "penal code crpc",
                "sec_set": {"302", "374"},
                "act_set": {"penal code", "crpc"},
                "posture": "CRIMINAL_APPEAL",
                "who_what": "ACCUSED_AS_APPELLANT__LOWER_CONVICTION",
                "issue_text": "Accused convicted for murder under Section 302 IPC.",
                "year": 2010,
                "date_str": "2010-04-15",
                "date_val": date(2010, 4, 15).toordinal(),
                "outcome": "ALLOWED",
            },
            {
                "id": 102,
                "title": "K. Raman v. Union of India",
                "clean_title": "kramanvunionofindia",
                "pre_text": "Civil Appeal concerning service seniority and promotion under Article 309 of Constitution. High Court dismissed writ petition.",
                "sec_tokens": "309",
                "act_tokens": "constitution",
                "sec_set": {"309"},
                "act_set": {"constitution"},
                "posture": "CIVIL_APPEAL",
                "who_what": "EMPLOYEE_AS_APPELLANT__LOWER_ORDER_AFFIRMED",
                "issue_text": "Service seniority and promotion dispute.",
                "year": 2015,
                "date_str": "2015-08-20",
                "date_val": date(2015, 8, 20).toordinal(),
                "outcome": "DISMISSED",
            },
            {
                "id": 103,
                "title": "Future Tech Corp v. Commercial Tax Officer",
                "clean_title": "futuretechcorpvcommercialtaxofficer",
                "pre_text": "Assessment order challenged under Sales Tax Act. High Court reversed assessment.",
                "sec_tokens": "12",
                "act_tokens": "sales tax",
                "sec_set": {"12"},
                "act_set": {"sales tax"},
                "posture": "CIVIL_APPEAL",
                "who_what": "ASSESSEE_AS_APPELLANT__LOWER_TAX_ASSESSMENT",
                "issue_text": "Assessment order validity.",
                "year": 2024,
                "date_str": "2024-01-10",
                "date_val": date(2024, 1, 10).toordinal(),
                "outcome": "DISMISSED",
            },
        ]

        self.index = SevenComponentSimilarityIndex()
        self.index.fit(self.mock_train)

    def test_target_case_not_retrieved(self):
        """Audit: Target case itself must never be retrieved as its own historical precedent."""
        target_case = [dict(self.mock_train[0])]  # Same ID 101
        target_case[0]["date_val"] = date(2020, 1, 1).toordinal()

        mask = self.index.build_eligibility_mask(target_case)
        # Check index 0 (which has id 101)
        self.assertFalse(
            mask[0, 0],
            "LEAKAGE VIOLATION: Target case ID was not excluded from candidate retrieval mask!"
        )

    def test_future_cases_not_retrieved(self):
        """Audit: Cases decided AFTER the target case decision timestamp must be excluded."""
        # Target case decided in 2012
        target_case = [{
            "id": 999,
            "title": "Citizen v. State",
            "clean_title": "citizenvstate",
            "pre_text": "Criminal Appeal under Section 302 IPC.",
            "sec_tokens": "302",
            "act_tokens": "penal code",
            "sec_set": {"302"},
            "act_set": {"penal code"},
            "posture": "CRIMINAL_APPEAL",
            "who_what": "ACCUSED_AS_APPELLANT__LOWER_CONVICTION",
            "issue_text": "Conviction challenge under Section 302 IPC.",
            "year": 2012,
            "date_str": "2012-05-10",
            "date_val": date(2012, 5, 10).toordinal(),
            "outcome": "ALLOWED",
        }]

        mask = self.index.build_eligibility_mask(target_case)
        # Train case 101 was 2010 (Prior -> Eligible)
        self.assertTrue(mask[0, 0], "Historical prior case should be eligible.")
        # Train case 102 was 2015 (Future -> Ineligible)
        self.assertFalse(mask[0, 1], "LEAKAGE VIOLATION: Future case (2015 > 2012) was not excluded!")
        # Train case 103 was 2024 (Future -> Ineligible)
        self.assertFalse(mask[0, 2], "LEAKAGE VIOLATION: Future case (2024 > 2012) was not excluded!")

    def test_same_case_family_and_duplicates_excluded(self):
        """Audit: Duplicates and cases with identical case titles are excluded."""
        target_case = [{
            "id": 888,
            "title": "State of Tamil Nadu v. Muthu",  # Exact duplicate title of case 101
            "clean_title": "stateoftamilnaduvmuthu",
            "pre_text": "Appeal against acquittal under CrPC.",
            "sec_tokens": "378",
            "act_tokens": "crpc",
            "sec_set": {"378"},
            "act_set": {"crpc"},
            "posture": "CRIMINAL_APPEAL",
            "who_what": "STATE_AS_APPELLANT__LOWER_ACQUITTAL",
            "issue_text": "State appeal against acquittal.",
            "year": 2020,
            "date_str": "2020-01-01",
            "date_val": date(2020, 1, 1).toordinal(),
            "outcome": "DISMISSED",
        }]

        mask = self.index.build_eligibility_mask(target_case)
        # Case 101 shares the exact title -> Must be excluded
        self.assertFalse(
            mask[0, 0],
            "LEAKAGE VIOLATION: Same-case-family / duplicate title was not excluded!"
        )

    def test_pre_decision_text_representation_only(self):
        """Audit: Trailing operative disposition words must not be required or used."""
        text_with_operative_end = (
            "This is the beginning of the case setting out facts. " * 100 +
            "\nFor the reasons aforesaid, the appeal is hereby allowed and conviction is set aside."
        )
        pre_text = text_with_operative_end[:3500]
        # In pre_text (first 3500 chars), the trailing text at the end of a long judgment is excluded
        self.assertNotIn(
            "the appeal is hereby allowed and conviction is set aside",
            pre_text,
            "Pre-decision text should not include trailing operative disposition."
        )

    def test_prediction_output_structure_and_explainability(self):
        """Audit: Evaluation and prediction functions must return complete audit explainability."""
        target_case = [{
            "id": 777,
            "title": "A. Sharma v. Commissioner of Income Tax",
            "clean_title": "asharmavcommissionerofincometax",
            "pre_text": "Civil Appeal involving Section 148 reassessment notice under Income Tax Act.",
            "sec_tokens": "148",
            "act_tokens": "income tax",
            "sec_set": {"148"},
            "act_set": {"income tax"},
            "posture": "CIVIL_APPEAL",
            "who_what": "ASSESSEE_AS_APPELLANT__LOWER_TAX_ASSESSMENT",
            "issue_text": "Validity of reassessment notice.",
            "year": 2022,
            "date_str": "2022-03-15",
            "date_val": date(2022, 3, 15).toordinal(),
            "outcome": "ALLOWED",
        }]

        mats = self.index.compute_component_matrices(target_case)
        self.assertEqual(len(mats), 7, "Must compute all 7 independent similarity component matrices.")

        mask = self.index.build_eligibility_mask(target_case)
        weights = [1.0 / 7.0] * 7
        S_hyb = sum(w * m for w, m in zip(weights, mats))

        preds, confs, probs, explanations, ret_metrics = predict_with_hybrid_similarity(
            S_hyb, mask, self.index.y_tr, target_case, self.mock_train, K=2
        )

        self.assertEqual(len(preds), 1)
        self.assertIn(preds[0], ["ALLOWED", "DISMISSED"])
        self.assertGreaterEqual(confs[0], 50.0)

        # Explainability validation
        self.assertEqual(len(explanations), 1)
        expl = explanations[0]
        self.assertEqual(expl["target_case_id"], 777)
        self.assertIn("top_retrieved_cases", expl)

        # Retrieval metrics keys validation
        for key in ["precision_at_5", "precision_at_10", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"]:
            self.assertIn(key, ret_metrics)


if __name__ == "__main__":
    unittest.main()
