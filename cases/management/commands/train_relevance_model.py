import os
import json
import joblib
import numpy as np
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from cases.models import LegalCase
from cases.services.feature_extraction import extract_features_for_pair, FEATURE_NAMES

class Command(BaseCommand):
    help = "Train and evaluate 2-stage Supervised Relevance Reranking Model with strict query-level splitting"

    def handle(self, *args, **options):
        self.stdout.write("====================================================")
        self.stdout.write("   LEGAL LENS SUPERVISED RELEVANCE EXPERIMENT (STEP 4) ")
        self.stdout.write("====================================================")

        benchmark_path = Path(settings.BASE_DIR) / "cases" / "data" / "relevance_benchmark.json"
        if not benchmark_path.exists():
            self.stderr.write(f"Error: Benchmark file not found at {benchmark_path}")
            return

        with open(benchmark_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries = data.get("queries", [])
        train_queries = [q for q in queries if q.get("split_assignment") == "train"]
        val_queries = [q for q in queries if q.get("split_assignment") == "val"]
        test_queries = [q for q in queries if q.get("split_assignment") == "test"]

        self.stdout.write(f"Query-level splits: Train ({len(train_queries)}), Val ({len(val_queries)}), Test ({len(test_queries)})")

        def extract_dataset_for_queries(q_list):
            X_data, y_data, metadata = [], [], []
            for q_item in q_list:
                qtext = q_item["query"]
                qid = q_item["query_id"]
                for candidate in q_item.get("candidates", []):
                    cid = candidate["case_id"]
                    # Prefer human_label over pseudo_label if present
                    rel = candidate.get("human_label")
                    if rel is None:
                        rel = candidate.get("pseudo_label", candidate.get("relevance", 0))
                    
                    try:
                        legal_case = LegalCase.objects.get(id=cid)
                    except LegalCase.DoesNotExist:
                        continue

                    feat_vec, _ = extract_features_for_pair(qtext, legal_case)
                    X_data.append(feat_vec)
                    y_data.append(1 if int(rel or 0) >= 2 else 0)
                    metadata.append({"query_id": qid, "case_id": cid, "title": legal_case.title})

            return np.array(X_data, dtype=np.float64), np.array(y_data, dtype=np.int32), metadata

        X_train, y_train, meta_train = extract_dataset_for_queries(train_queries)
        X_val, y_val, meta_val = extract_dataset_for_queries(val_queries)

        self.stdout.write(f"Extracted Train samples: {len(X_train)} (Positive: {sum(y_train)})")
        self.stdout.write(f"Extracted Val samples  : {len(X_val)} (Positive: {sum(y_val)})")

        # 1. Primary Model: Logistic Regression
        lr_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))
        ])
        lr_pipeline.fit(X_train, y_train)

        lr_val_preds = lr_pipeline.predict_proba(X_val)[:, 1]
        lr_val_acc = accuracy_score(y_val, (lr_val_preds >= 0.5).astype(int))
        lr_val_auc = roc_auc_score(y_val, lr_val_preds) if len(set(y_val)) > 1 else 0.5
        lr_brier = brier_score_loss(y_val, lr_val_preds)

        self.stdout.write(f"\n--- MODEL COMPARISON (VAL SPLIT) ---")
        self.stdout.write(f"Logistic Regression  -> Accuracy: {lr_val_acc:.4f}, ROC-AUC: {lr_val_auc:.4f}, Brier Score: {lr_brier:.4f}")

        # 2. Alternative Models: Random Forest & Gradient Boosting
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_val_preds = rf_model.predict_proba(X_val)[:, 1]
        rf_val_acc = accuracy_score(y_val, (rf_val_preds >= 0.5).astype(int))
        rf_val_auc = roc_auc_score(y_val, rf_val_preds) if len(set(y_val)) > 1 else 0.5

        gb_model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        gb_model.fit(X_train, y_train)
        gb_val_preds = gb_model.predict_proba(X_val)[:, 1]
        gb_val_acc = accuracy_score(y_val, (gb_val_preds >= 0.5).astype(int))
        gb_val_auc = roc_auc_score(y_val, gb_val_preds) if len(set(y_val)) > 1 else 0.5

        self.stdout.write(f"Random Forest        -> Accuracy: {rf_val_acc:.4f}, ROC-AUC: {rf_val_auc:.4f}")
        self.stdout.write(f"Gradient Boosting    -> Accuracy: {gb_val_acc:.4f}, ROC-AUC: {gb_val_auc:.4f}")

        # 3. Feature Importance Analysis (Logistic Regression Coefficients)
        coefs = lr_pipeline.named_steps["model"].coef_[0]
        self.stdout.write(f"\n--- FEATURE COEFFICIENTS (LOGISTIC REGRESSION) ---")
        for fname, coef in zip(FEATURE_NAMES, coefs):
            self.stdout.write(f"  {fname:<30}: {coef:+.4f}")

        # 4. Error Analysis on Val Split
        false_positives = []
        false_negatives = []
        for meta, pred, actual in zip(meta_val, lr_val_preds, y_val):
            is_pred_pos = pred >= 0.5
            if is_pred_pos and actual == 0:
                false_positives.append({**meta, "pred_score": round(float(pred), 4)})
            elif not is_pred_pos and actual == 1:
                false_negatives.append({**meta, "pred_score": round(float(pred), 4)})

        self.stdout.write(f"\n--- ERROR ANALYSIS ---")
        self.stdout.write(f"False Positives: {len(false_positives)} | False Negatives: {len(false_negatives)}")
        if false_positives:
            self.stdout.write(f"Sample False Positive: {false_positives[0]}")
        if false_negatives:
            self.stdout.write(f"Sample False Negative: {false_negatives[0]}")

        # Save Model Artifact
        models_dir = Path(settings.BASE_DIR) / "ml_models"
        models_dir.mkdir(exist_ok=True)
        artifact_path = models_dir / "supervised_relevance_v1.joblib"

        artifact = {
            "pipeline": lr_pipeline,
            "feature_names": FEATURE_NAMES,
            "feature_dim": len(FEATURE_NAMES),
            "coefficients": dict(zip(FEATURE_NAMES, [float(c) for c in coefs])),
            "model_type": "Logistic Regression with StandardScaler",
            "training_data_composition": f"Weak Supervision ({len(X_train)} samples across {len(train_queries)} train queries)",
            "metrics": {
                "val_accuracy": float(lr_val_acc),
                "val_roc_auc": float(lr_val_auc),
                "val_brier_score": float(lr_brier),
            },
            "calibration_status": "Uncalibrated - Label as Predicted Relevance Score",
            "error_summary": {
                "val_false_positives_count": len(false_positives),
                "val_false_negatives_count": len(false_negatives),
                "sample_false_positive": false_positives[0] if false_positives else None,
                "sample_false_negative": false_negatives[0] if false_negatives else None,
            }
        }

        joblib.dump(artifact, artifact_path)
        self.stdout.write(self.style.SUCCESS(f"\nSaved Supervised Relevance Artifact to {artifact_path}"))
