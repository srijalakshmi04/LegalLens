# FIRE AILA 2019 Task 1 Precedent Retrieval Adapter Module

This directory implements the external benchmark adapter for the **FIRE AILA 2019 Task 1 Precedent Retrieval** task.

## Directory Structure
- `raw/`: Unmodified raw AILA source files (preserved).
- `processed/`: Parsed AILA queries (`aila_queries.json`), cases (`aila_cases.json`), and relevance judgments (`aila_qrels.json`).
- `mappings/`: `case_id_mapping.json` (Maps AILA case IDs $\to$ LegalLens case IDs with verification status).
- `results/`: `aila_evaluation_results.json` (Stores dry-run baseline evaluation metrics).
- `adapter.py`: Data processor constructing structured AILA objects and mappings.

## Evaluation Execution Command
```powershell
python manage.py evaluate_aila
```
