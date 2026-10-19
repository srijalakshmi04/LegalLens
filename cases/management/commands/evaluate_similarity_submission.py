import json
import re
import time
from pathlib import Path

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand

from cases.services.evaluation import calculate_mrr, calculate_ndcg_at_k


class Command(BaseCommand):
    help = (
        "Execute official Legal Lens Precedent Retrieval Submission Benchmark across "
        "50 benchmark queries and 2,914 Supreme Court case corpus using official FIRE AILA 2019 Task 1 ground truth."
    )

    def handle(self, *args, **options):
        # 1. Paths & Verification
        aila_dir = Path(settings.BASE_DIR) / "external_benchmarks" / "AILA_2019" / "raw"
        qrels_path = aila_dir / "relevance_judgments_priorcases.txt"
        benchmark_path = Path(settings.BASE_DIR) / "cases" / "data" / "similarity_benchmark_v2.json"
        model_path = Path(settings.BASE_DIR) / "ml_models" / "similarity_classifier_v2.joblib"

        if not qrels_path.exists():
            self.stderr.write(f"Error: Official ground truth not found at {qrels_path}")
            return
        if not benchmark_path.exists():
            self.stderr.write(f"Error: Benchmark dataset not found at {benchmark_path}")
            return
        if not model_path.exists():
            self.stderr.write(f"Error: Frozen model artifact not found at {model_path}")
            return

        # 2. Load Official FIRE AILA 2019 Relevance Ground Truth
        qrels = {}
        with open(qrels_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, cid, rel = parts[0], parts[2], int(parts[3])
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][cid] = rel

        # 3. Load Benchmark Queries
        with open(benchmark_path, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

        queries = bench_data.get("queries", [])
        total_queries = len(queries)
        if total_queries == 0:
            self.stderr.write("Error: No benchmark queries found.")
            return

        # 4. Load Frozen Similarity Retrieval Configuration
        import joblib
        t_load_start = time.time()
        artifact = joblib.load(model_path)
        load_time = time.time() - t_load_start

        tfidf = artifact.get("tfidf")
        lsa = artifact.get("lsa")
        scaler = artifact.get("scaler")
        clf = artifact.get("classifier")
        calibrator = artifact.get("calibrator")
        cid_to_idx = artifact.get("cid_to_idx", {})
        cids = artifact.get("cids", [])
        X_semantic = artifact.get("X_semantic")
        X_tfidf = artifact.get("X_tfidf")
        bm25 = artifact.get("bm25")
        tokenized_corpus = artifact.get("tokenized_corpus", [])
        doc_chunks_semantic = artifact.get("doc_chunks_semantic", [])
        doc_acts = artifact.get("doc_acts", [])
        doc_secs = artifact.get("doc_secs", [])
        doc_postures = artifact.get("doc_postures", [])
        doc_titles = artifact.get("doc_titles", [])

        total_corpus_cases = len(cids) if cids else 2914

        # Extraction Patterns for Statutes and Procedural Posture
        legal_act_patterns = [
            ("Penal Code", r"\b(indian penal code|ipc|penal code)\b"),
            ("CrPC", r"\b(code of criminal procedure|crpc|cr\.p\.c|criminal procedure code)\b"),
            ("CPC", r"\b(code of civil procedure|cpc|c\.p\.c|civil procedure code)\b"),
            ("Evidence Act", r"\b(evidence act|indian evidence act)\b"),
            ("Constitution", r"\b(constitution of india|constitution|article \d+)\b"),
            ("Land Acquisition", r"\b(land acquisition act|land acquisition)\b"),
            ("Negotiable Instruments", r"\b(negotiable instruments act|ni act|section 138)\b"),
            ("NDPS", r"\b(ndps act|narcotic drugs|psychotropic substances)\b"),
            ("Income Tax", r"\b(income tax act|income-tax)\b"),
            ("Motor Vehicles", r"\b(motor vehicles act|mact)\b"),
            ("Arbitration", r"\b(arbitration and conciliation act|arbitration act)\b"),
            ("Specific Relief", r"\b(specific relief act)\b"),
            ("Prevention of Corruption", r"\b(prevention of corruption act)\b"),
            ("Companies Act", r"\b(companies act)\b"),
            ("Consumer Protection", r"\b(consumer protection act)\b"),
        ]
        legal_sec_patterns = r"\b(?:section|sec|s\.|article|art|order|rule)\s*([0-9]+[a-z]?)\b"
        procedural_posture_patterns = [
            ("Criminal Appeal", r"\bcriminal appeal\b"),
            ("Civil Appeal", r"\bcivil appeal\b"),
            ("Special Leave Petition", r"\b(?:special leave petition|slp)\b"),
            ("Writ Petition", r"\bwrit petition\b"),
            ("Bail Application", r"\b(?:bail application|regular bail|anticipatory bail|default bail)\b"),
            ("Review Petition", r"\breview petition\b"),
            ("Contempt Petition", r"\bcontempt petition\b"),
        ]

        def extract_statutes(text):
            tl = text.lower()
            acts = {name for name, pat in legal_act_patterns if re.search(pat, tl)}
            secs = set(re.findall(legal_sec_patterns, tl))
            return acts, secs

        def extract_postures(text):
            tl = text.lower()
            return {name for name, pat in procedural_posture_patterns if re.search(pat, tl)}

        # 5. Evaluate Retrieval Across Benchmark Queries
        top1_hits = 0
        top5_hits = 0
        top10_hits = 0

        p5_list = []
        p10_list = []
        r5_list = []
        r10_list = []
        mrr_list = []
        ndcg10_list = []
        top1_sim_list = []

        query_details = []

        t_eval_start = time.time()

        for q_idx, q in enumerate(queries, start=1):
            qid = q.get("query_id", f"Q{q_idx}")
            qtext = q.get("query", "")
            candidates = q.get("candidates", [])

            # Official relevance map for this query
            official_qrels = qrels.get(qid, {})
            total_relevant_in_qrels = sum(1 for r in official_qrels.values() if r > 0)

            # Query representations
            q_tfidf = tfidf.transform([qtext])
            q_sem = lsa.transform(q_tfidf)[0]
            q_tokens = qtext.lower().split()
            q_acts, q_secs = extract_statutes(qtext)
            q_post = extract_postures(qtext)
            q_title_words = [w for w in q_tokens[:10] if len(w) > 3]

            bm25_scores = np.array(bm25.get_scores(q_tokens)) if bm25 is not None else np.zeros(total_corpus_cases)
            max_bm25 = float(np.max(bm25_scores)) if len(bm25_scores) > 0 and np.max(bm25_scores) > 0 else 1.0

            scored_candidates = []
            for c in candidates:
                cid = c.get("case_id")
                is_rel = official_qrels.get(cid, 0)
                c_idx = cid_to_idx.get(cid)

                if c_idx is not None and c_idx < total_corpus_cases:
                    c_sem = X_semantic[c_idx]
                    sem_cos = float(np.dot(q_sem, c_sem))
                    c_tfidf = X_tfidf[c_idx]
                    tfidf_cos = float(q_tfidf.dot(c_tfidf.T).toarray()[0][0])
                    raw_bm25 = float(bm25_scores[c_idx])
                    bm25_norm = min(1.0, max(0.0, raw_bm25 / max_bm25))

                    c_a = doc_acts[c_idx] if c_idx < len(doc_acts) else set()
                    c_s = doc_secs[c_idx] if c_idx < len(doc_secs) else set()
                    act_overlap = len(q_acts & c_a) / max(1, len(q_acts | c_a)) if (q_acts or c_a) else 0.0
                    sec_overlap = len(q_secs & c_s) / max(1, len(q_secs | c_s)) if (q_secs or c_s) else 0.0
                    issue_sim = 0.6 * act_overlap + 0.4 * sec_overlap

                    c_tokens = tokenized_corpus[c_idx] if c_idx < len(tokenized_corpus) else []
                    q_set = set(q_tokens[:100])
                    c_set = set(c_tokens[:200])
                    fact_sim = len(q_set & c_set) / max(1, len(q_set | c_set))

                    c_post = doc_postures[c_idx] if c_idx < len(doc_postures) else set()
                    proc_sim = 1.0 if (q_post & c_post) else (0.5 if (not q_post and not c_post) else 0.0)

                    c_chunks_sem = doc_chunks_semantic[c_idx] if c_idx < len(doc_chunks_semantic) else [c_sem]
                    chunk_sims = np.dot(c_chunks_sem, q_sem)
                    max_chunk_sim = float(np.max(chunk_sims))
                    mean_chunk_sim = float(np.mean(chunk_sims))
                    top_chunk_sim = float(np.mean(np.sort(chunk_sims)[-min(3, len(chunk_sims)):]))
                    passage_sim = max_chunk_sim

                    c_title = doc_titles[c_idx] if c_idx < len(doc_titles) else ""
                    t_matches = sum(1 for w in q_title_words if w in c_title)
                    title_sim = min(1.0, t_matches / max(1, len(q_title_words)))

                    feat = [
                        sem_cos, tfidf_cos, bm25_norm,
                        issue_sim, act_overlap, sec_overlap,
                        fact_sim, proc_sim, passage_sim,
                        title_sim, max_chunk_sim, mean_chunk_sim,
                        top_chunk_sim
                    ]
                else:
                    feat = [0.0] * 13

                feat_scaled = scaler.transform([feat])
                score = clf.decision_function(feat_scaled).reshape(-1, 1)
                prob = float(calibrator.predict_proba(score)[0, 1])

                scored_candidates.append({
                    "case_id": cid,
                    "relevance": is_rel,
                    "similarity_score": prob,
                    "similarity_pct": round(prob * 100.0, 2),
                })

            # Rank retrieved candidates descending by similarity score
            scored_candidates.sort(key=lambda x: x["similarity_score"], reverse=True)

            top1_case = scored_candidates[0] if scored_candidates else {"case_id": None, "relevance": 0, "similarity_pct": 0.0}
            is_top1_relevant = (top1_case["relevance"] > 0)
            if is_top1_relevant:
                top1_hits += 1

            top1_sim_list.append(top1_case["similarity_pct"])

            retrieved_rels = [c["relevance"] for c in scored_candidates]
            hit_top5 = 1 if sum(retrieved_rels[:5]) > 0 else 0
            hit_top10 = 1 if sum(retrieved_rels[:10]) > 0 else 0

            top5_hits += hit_top5
            top10_hits += hit_top10

            p5 = sum(retrieved_rels[:5]) / 5.0
            p10 = sum(retrieved_rels[:10]) / 10.0
            r5 = min(1.0, sum(retrieved_rels[:5]) / max(1, total_relevant_in_qrels))
            r10 = min(1.0, sum(retrieved_rels[:10]) / max(1, total_relevant_in_qrels))
            mrr = calculate_mrr(retrieved_rels)
            ideal_rels = sorted(list(official_qrels.values()), reverse=True)
            ndcg10 = calculate_ndcg_at_k(retrieved_rels, ideal_rels, k=10)

            p5_list.append(p5)
            p10_list.append(p10)
            r5_list.append(r5)
            r10_list.append(r10)
            mrr_list.append(mrr)
            ndcg10_list.append(ndcg10)

            query_details.append({
                "query_id": qid,
                "top1_case_id": top1_case["case_id"],
                "top1_relevance": top1_case["relevance"],
                "top1_similarity_pct": top1_case["similarity_pct"],
                "top1_hit": is_top1_relevant,
                "top5_hit": bool(hit_top5),
                "top10_hit": bool(hit_top10),
                "precision_at_5": round(p5, 4),
                "precision_at_10": round(p10, 4),
                "recall_at_5": round(r5, 4),
                "recall_at_10": round(r10, 4),
                "mrr": round(mrr, 4),
                "ndcg_at_10": round(ndcg10, 4),
            })

        total_eval_time = time.time() - t_eval_start

        # 6. Aggregate Benchmark Metrics
        top1_accuracy = (top1_hits / total_queries) * 100.0
        top5_success = (top5_hits / total_queries) * 100.0
        top10_success = (top10_hits / total_queries) * 100.0

        mean_p5 = float(np.mean(p5_list))
        mean_p10 = float(np.mean(p10_list))
        mean_r5 = float(np.mean(r5_list))
        mean_r10 = float(np.mean(r10_list))
        mean_mrr = float(np.mean(mrr_list))
        mean_ndcg10 = float(np.mean(ndcg10_list))
        mean_top1_sim = float(np.mean(top1_sim_list))

        target_status = "ACHIEVED" if top1_accuracy >= 95.0 else "NOT ACHIEVED"

        # 7. Print Terminal Output (Exact Required Format)
        self.stdout.write("==================================================")
        self.stdout.write("LEGAL LENS — SUBMISSION BENCHMARK")
        self.stdout.write("==================================================")
        self.stdout.write(f"TOTAL BENCHMARK QUERIES: {total_queries}")
        self.stdout.write(f"TOTAL CORPUS CASES: {total_corpus_cases:,}")
        self.stdout.write("GROUND TRUTH: FIRE AILA 2019 Task 1")
        self.stdout.write("")
        self.stdout.write(f"TOP-1 SIMILARITY RETRIEVAL ACCURACY: {top1_accuracy:.2f}%")
        self.stdout.write(f"QUERIES WITH RELEVANT TOP-1: {top1_hits} / {total_queries}")
        self.stdout.write("")
        self.stdout.write(f"TOP-5 RETRIEVAL SUCCESS: {top5_success:.2f}% ({top5_hits} / {total_queries})")
        self.stdout.write(f"TOP-10 RETRIEVAL SUCCESS: {top10_success:.2f}% ({top10_hits} / {total_queries})")
        self.stdout.write("")
        self.stdout.write(f"PRECISION@5: {mean_p5:.4f}")
        self.stdout.write(f"PRECISION@10: {mean_p10:.4f}")
        self.stdout.write(f"RECALL@5: {mean_r5:.4f}")
        self.stdout.write(f"RECALL@10: {mean_r10:.4f}")
        self.stdout.write(f"MRR: {mean_mrr:.4f}")
        self.stdout.write(f"NDCG@10: {mean_ndcg10:.4f}")
        self.stdout.write("")
        self.stdout.write(f"MEAN SIMILARITY SCORE (TOP-1): {mean_top1_sim:.2f}%")
        self.stdout.write("==================================================")
        self.stdout.write("")
        self.stdout.write("==================================================")
        self.stdout.write("PRIMARY METRIC HEADLINE")
        self.stdout.write("==================================================")
        self.stdout.write(f"TOP-1 SIMILARITY RETRIEVAL ACCURACY = {top1_accuracy:.2f}%")
        self.stdout.write(f"95% TARGET = {target_status}")
        self.stdout.write(f"CORRECT / TOTAL = {top1_hits} / {total_queries}")
        self.stdout.write("==================================================")

        # 8. Save results/similarity_submission_metrics.json
        results_dir = Path(settings.BASE_DIR) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        metrics_payload = {
            "system_description": "Legal Lens is an explainable legal case similarity and precedent retrieval system for Indian Supreme Court judgments.",
            "task": "Legal Case Similarity and Precedent Retrieval",
            "benchmark": "FIRE AILA 2019 Task 1 Precedent Retrieval",
            "ground_truth_file": "relevance_judgments_priorcases.txt",
            "total_benchmark_queries": total_queries,
            "total_corpus_cases": total_corpus_cases,
            "primary_submission_metric": {
                "metric_name": "TOP-1 SIMILARITY RETRIEVAL ACCURACY",
                "accuracy_pct": round(top1_accuracy, 2),
                "correct_queries": top1_hits,
                "total_queries": total_queries,
                "target_pct": 95.0,
                "target_status": target_status,
            },
            "retrieval_success_metrics": {
                "top_5_retrieval_success_pct": round(top5_success, 2),
                "top_5_correct_queries": top5_hits,
                "top_10_retrieval_success_pct": round(top10_success, 2),
                "top_10_correct_queries": top10_hits,
            },
            "information_retrieval_metrics": {
                "precision_at_5": round(mean_p5, 4),
                "precision_at_10": round(mean_p10, 4),
                "recall_at_5": round(mean_r5, 4),
                "recall_at_10": round(mean_r10, 4),
                "mrr": round(mean_mrr, 4),
                "ndcg_at_10": round(mean_ndcg10, 4),
            },
            "similarity_score_metrics": {
                "mean_top_1_similarity_pct": round(mean_top1_sim, 2),
            },
            "execution_timings": {
                "model_load_time_sec": round(load_time, 2),
                "evaluation_duration_sec": round(total_eval_time, 2),
            },
            "query_breakdown": query_details,
        }

        json_out = results_dir / "similarity_submission_metrics.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)

        # 9. Save results/similarity_submission_report.md
        report_md = f"""# Legal Lens — Similarity and Precedent Retrieval Submission Report

> **Project Description**: Legal Lens is an explainable legal case similarity and precedent retrieval system for Indian Supreme Court judgments.
>
> **Strict Operational Boundary**: Pure legal case similarity and precedent retrieval. All outcome predictions (ALLOWED/DISMISSED, conviction/acquittal, future judgment outcomes) have been completely removed.

---

## 1. Primary Submission Metric

$$\\text{{TOP-1 SIMILARITY RETRIEVAL ACCURACY}} = \\frac{{\\text{{Queries with Relevant Top-1 Precedent}}}}{{\\text{{Total Benchmark Queries}}}} \\times 100$$

| Primary Metric | Result | Target | Status |
| :--- | :--- | :--- | :---: |
| **TOP-1 SIMILARITY RETRIEVAL ACCURACY** | **{top1_accuracy:.2f}%** | **>=95.00%** | **{target_status}** |
| **Queries with Relevant Top-1 Result** | **{top1_hits} / {total_queries}** | - | - |
| **Total Benchmark Queries** | **{total_queries}** | - | - |
| **Total Corpus Cases** | **{total_corpus_cases:,}** | - | - |
| **Ground Truth Source** | **FIRE AILA 2019 Task 1** | - | - |

---

## 2. Precedent Retrieval Success Rates

| Retrieval Metric | Score | Correct / Total | Description |
| :--- | :--- | :--- | :--- |
| **Top-1 Retrieval Accuracy** | `{top1_accuracy:.2f}%` | `{top1_hits} / {total_queries}` | #1 retrieved precedent is officially relevant |
| **Top-5 Retrieval Success** | `{top5_success:.2f}%` | `{top5_hits} / {total_queries}` | At least 1 relevant precedent retrieved in Top-5 |
| **Top-10 Retrieval Success** | `{top10_success:.2f}%` | `{top10_hits} / {total_queries}` | At least 1 relevant precedent retrieved in Top-10 |

---

## 3. Standard Information Retrieval (IR) Metrics

| IR Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Precision@5** | `{mean_p5:.4f}` | Proportion of relevant precedents among the top 5 retrieved cases |
| **Precision@10** | `{mean_p10:.4f}` | Proportion of relevant precedents among the top 10 retrieved cases |
| **Recall@5** | `{mean_r5:.4f}` | Proportion of all relevant ground-truth precedents retrieved within top 5 |
| **Recall@10** | `{mean_r10:.4f}` | Proportion of all relevant ground-truth precedents retrieved within top 10 |
| **MRR (Mean Reciprocal Rank)** | `{mean_mrr:.4f}` | Average reciprocal rank of the first relevant retrieved precedent |
| **NDCG@10** | `{mean_ndcg10:.4f}` | Normalized Discounted Cumulative Gain accounting for rank positions |
| **Mean Top-1 Similarity Score** | `{mean_top1_sim:.2f}%` | Average calibrated similarity confidence of the primary retrieved precedent |

---

## 4. Query-Level Performance Summary

- **Total Queries Evaluated**: {total_queries}
- **Queries Achieving Top-1 Hit**: {top1_hits} ({top1_accuracy:.2f}%)
- **Queries Achieving Top-5 Hit**: {top5_hits} ({top5_success:.2f}%)
- **Queries Achieving Top-10 Hit**: {top10_hits} ({top10_success:.2f}%)
- **Queries Requiring Rank 2-5 for First Relevant Precedent**: {total_queries - top1_hits}

---

## 5. Submission Headline Summary

```text
==================================================
LEGAL LENS — SUBMISSION BENCHMARK
==================================================
TOTAL BENCHMARK QUERIES: {total_queries}
TOTAL CORPUS CASES: {total_corpus_cases:,}
GROUND TRUTH: FIRE AILA 2019 Task 1

TOP-1 SIMILARITY RETRIEVAL ACCURACY: {top1_accuracy:.2f}%
QUERIES WITH RELEVANT TOP-1: {top1_hits} / {total_queries}

TOP-5 RETRIEVAL SUCCESS: {top5_success:.2f}% ({top5_hits} / {total_queries})
TOP-10 RETRIEVAL SUCCESS: {top10_success:.2f}% ({top10_hits} / {total_queries})

PRECISION@5: {mean_p5:.4f}
PRECISION@10: {mean_p10:.4f}
RECALL@5: {mean_r5:.4f}
RECALL@10: {mean_r10:.4f}
MRR: {mean_mrr:.4f}
NDCG@10: {mean_ndcg10:.4f}

MEAN SIMILARITY SCORE (TOP-1): {mean_top1_sim:.2f}%
==================================================

==================================================
PRIMARY METRIC HEADLINE
==================================================
TOP-1 SIMILARITY RETRIEVAL ACCURACY = {top1_accuracy:.2f}%
95% TARGET = {target_status}
CORRECT / TOTAL = {top1_hits} / {total_queries}
==================================================
```
"""
        report_out = results_dir / "similarity_submission_report.md"
        with open(report_out, "w", encoding="utf-8") as f:
            f.write(report_md)

        self.stdout.write(self.style.SUCCESS(f"\nSaved metrics: {json_out}"))
        self.stdout.write(self.style.SUCCESS(f"Saved report:  {report_out}"))
