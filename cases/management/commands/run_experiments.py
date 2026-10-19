import json
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings
from cases.services.evaluation import evaluate_retrieval_benchmark
from cases.services.similarity import get_supervised_model_status

class Command(BaseCommand):
    help = "Run Legal Lens evaluation experiments protocol across 6 baselines and 5 feature ablations"

    def handle(self, *args, **options):
        self.stdout.write("Running Legal Lens evaluation protocol...")
        
        model_status = get_supervised_model_status()
        self.stdout.write(f"Model Status: {model_status['status']}")
        self.stdout.write(f"Active Engine: {model_status['active_engine']}")

        eval_data = evaluate_retrieval_benchmark()
        self.stdout.write(f"Evaluation Status: {eval_data['status']}")

        results_path = Path(settings.BASE_DIR) / "cases" / "data" / "evaluation_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=2)

        self.stdout.write(self.style.SUCCESS(f"Saved evaluation protocol record to {results_path}"))
