import os
import sys
import logging
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from cases.services.similarity_outcome_predictor import run_similarity_prediction_evaluation

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Evaluate the current-case pre-decision similarity-based outcome prediction system."

    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING("\n========================================================"))
        self.stdout.write(self.style.MIGRATE_HEADING("   LEGAL LENS — CURRENT-CASE SIMILARITY PREDICTION ENGINE"))
        self.stdout.write(self.style.MIGRATE_HEADING("========================================================\n"))

        self.stdout.write("Running 7-component similarity retrieval & outcome voting pipeline...")
        self.stdout.write("Stratified Split: 70% Train (11,513), 15% Validation (2,468), 15% Test (2,468) [Seed=42]")
        self.stdout.write("Representation: STRICTLY PRE-DECISION (chars 0-3500; no operative/disposition text)")
        self.stdout.write("Temporal Eligibility: strictly historical cases decided BEFORE target case timestamp\n")

        res = run_similarity_prediction_evaluation()

        b = res["benchmark_summary"]
        r = res["retrieval_metrics"]
        c = res["selected_configuration"]
        w = c["weights"]

        self.stdout.write(self.style.SUCCESS("\n========================================================"))
        self.stdout.write(self.style.SUCCESS(f"   {res['headline']}"))
        self.stdout.write(self.style.SUCCESS("========================================================"))
        
        status_style = self.style.SUCCESS if res["target_95_status"] == "ACHIEVED" else self.style.WARNING
        self.stdout.write(status_style(f"95% TARGET: {res['target_95_status']}"))
        self.stdout.write(f"REQUIRED FOR 95%: 2345 / {b['total_test_cases']} correct")
        self.stdout.write(f"ACTUAL VERIFIED:  {b['correct']} / {b['total_test_cases']} correct ({b['accuracy']}%)")
        self.stdout.write(f"INCORRECT:        {b['incorrect']} / {b['total_test_cases']}")
        self.stdout.write(f"BALANCED ACCURACY:{b['balanced_accuracy']}%")
        self.stdout.write(f"MACRO F1:         {b['macro_f1']}")
        self.stdout.write(f"WEIGHTED F1:      {b['f1']}")
        self.stdout.write(f"COVERAGE:         {b['coverage_pct']}%")
        self.stdout.write(f"ABSTENTION RATE:  {b['abstention_rate_pct']}%")
        self.stdout.write(f"BRIER SCORE:      {b['brier_score']}")
        self.stdout.write(f"CONFUSION MATRIX: {b['confusion_matrix']['matrix']} (Labels: {b['confusion_matrix']['labels']})\n")

        self.stdout.write(self.style.MIGRATE_LABEL("--- SEPARATE PRECEDENT RETRIEVAL METRICS ---"))
        self.stdout.write(f"Precision@5:  {r['precision_at_5']}")
        self.stdout.write(f"Precision@10: {r['precision_at_10']}")
        self.stdout.write(f"Recall@5:     {r['recall_at_5']}")
        self.stdout.write(f"Recall@10:    {r['recall_at_10']}")
        self.stdout.write(f"MRR:          {r['mrr']}")
        self.stdout.write(f"NDCG@10:      {r['ndcg_at_10']}\n")

        self.stdout.write(self.style.MIGRATE_LABEL("--- PHASE 9: COMPREHENSIVE BASELINE COMPARISON (8 ARCHITECTURES) ---"))
        self.stdout.write(f"{'Code':<6} {'Model Architecture':<48} {'Val Acc':<10} {'Test Acc':<10} {'Test Correct'}")
        self.stdout.write("-" * 85)
        for m in res.get("baseline_comparison", []):
            self.stdout.write(f"{m['code']:<6} {m['name']:<48} {m['val_acc']}%{'':<3} {m['test_acc']}%{'':<3} {m['test_correct']}/2468")
        self.stdout.write("")

        self.stdout.write(self.style.MIGRATE_LABEL("--- SELECTED HYBRID CONFIGURATION (VALIDATION ONLY) ---"))
        self.stdout.write(f"Profile:             {c['profile_name']}")
        self.stdout.write(f"Selected Top-K:      {c['top_k']}")
        self.stdout.write(f"CBR Stacking Alpha:  {c.get('cbr_alpha', 0.80)}")
        self.stdout.write(f"Validation Accuracy: {c['validation_accuracy']}%\n")

        res_path = Path(settings.BASE_DIR) / "results" / "similarity_v2_metrics.json"
        rep_path = Path(settings.BASE_DIR) / "results" / "similarity_v2_report.md"

        self.stdout.write(self.style.SUCCESS(f"V2 Metrics saved to: {res_path}"))
        self.stdout.write(self.style.SUCCESS(f"V2 Report saved to:  {rep_path}\n"))
