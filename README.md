# Embla Hybrid Classifier

Hybrid ML + LLM classification for mapping vendor items to ISO 9999 assistive product codes.

**Pattern:** ML classifier handles 85% of items (fast, free). AI_QUERY handles the uncertain 15% (accurate, structured output). Human corrections feed back into retraining.

## Quick Start

```bash
# Deploy to Databricks
cd projects/embla-hybrid-classifier
databricks bundle deploy -t dev
databricks bundle run hybrid_classifier_pipeline -t dev
```

Or run notebooks individually in the workspace.

## Architecture

```
New Items → ML Classifier → confidence ≥ 0.85 → Accept
                           → confidence < 0.85 → AI_QUERY (LLM) → Accept
                                                                      ↓
                                             Human Corrections ← classified_items
                                                    ↓
                                               Retrain ML
```

## Notebooks

| # | Notebook | Purpose |
|---|----------|---------|
| 0 | `00_setup_data.py` | Create tables, generate synthetic MedTech data |
| 1 | `01_train_classifier.py` | Train sklearn model, register in UC with Champion alias |
| 2 | `02_hybrid_score_pipeline.py` | ML scoring → confidence gate → AI_QUERY fallback |
| 3 | `03_monitor_and_retrain.py` | Monitor drift, retrain with corrections, promote |

## Key Technologies

- **MLflow** — experiment tracking, UC model registry, Champion/Challenger aliases
- **AI_QUERY** — Foundation Model SQL function with `responseFormat` for structured JSON
- **ai_classify** — zero-training classification for broad categories
- **Lakehouse Monitor** — automated drift detection on inference tables
- **Unity Catalog** — governance, lineage, permissions on models and tables

## Documentation

See [SOLUTION_GUIDE.md](SOLUTION_GUIDE.md) for the full technical walkthrough with code snippets.

## Context

Built for Embla Medical (formerly Össur) — MedTech company classifying vendor price-list items into ISO 9999 codes for assistive products (prostheses, orthoses, compression therapy, wound care, mobility aids).
