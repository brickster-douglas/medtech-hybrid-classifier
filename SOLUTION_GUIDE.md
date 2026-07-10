# Hybrid ML + LLM Classification on Databricks

A production pattern for classifying vendor items into ISO 9999 codes using a two-tier approach: a fast ML classifier for high-confidence predictions, with an LLM fallback for uncertain items.

## The Problem

MedTech companies receive thousands of vendor price-list items that need to be mapped to standardized ISO 9999 assistive product codes. Manual classification is slow and error-prone. A pure ML model handles most items well but struggles with ambiguous descriptions. A pure LLM approach is accurate but expensive at scale.

## The Solution: Best of Both Worlds

```
New vendor items
       │
       ▼
┌──────────────┐
│ ML Classifier │   sklearn model, registered in Unity Catalog
│  ~85% of items│   Fast, cheap, handles known patterns
└──────┬───────┘
       │
  confidence score
       │
  ┌────┴────┐
  │         │
  ≥ 0.85   < 0.85
  │         │
  │    ┌────┴─────┐
  │    │ AI_QUERY  │   Foundation Model via SQL
  │    │ ~15% of   │   Structured JSON response
  │    │ items     │   Full ISO code context
  │    └────┬─────┘
  │         │
  ▼         ▼
┌──────────────┐
│  Unified     │   classified_items table
│  Output      │   With method, confidence, LLM reasoning
└──────────────┘
       │
       ▼
  Human review → Corrections → Retraining
```

**Why this works:**
- The ML model is fast and free (runs on compute, no API calls)
- Only 10-20% of items route to the LLM, keeping costs low
- The LLM gets the full ISO code reference as context, so it's highly accurate
- Human corrections feed back into the ML model, reducing LLM dependency over time

---

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- Serverless compute (required for AI_QUERY)
- A Foundation Model endpoint (e.g., `databricks-meta-llama-3-3-70b-instruct`)
- MLflow and the Databricks SDK

---

## Step 1: Data Model

Five tables in Unity Catalog:

| Table | Purpose |
|-------|---------|
| `iso_codes` | Reference table — ISO 9999 codes with descriptions |
| `labeled_items` | Training data — items with known ISO codes |
| `new_items` | Incoming items to classify |
| `classified_items` | Output — predictions with method and confidence |
| `human_corrections` | Expert overrides for the feedback loop |

### ISO Code Reference Table

```sql
CREATE TABLE iso_codes (
    iso_code        STRING NOT NULL,    -- "06 24 09"
    name            STRING NOT NULL,    -- "Trans-tibial prostheses"
    iso_code_level_2 STRING,            -- "06 24"
    part_of_iso_standard BOOLEAN,
    description     STRING              -- Free-text description
);
```

### Vendor Items Table

```sql
CREATE TABLE labeled_items (
    item_id             STRING NOT NULL,
    vendor_name         STRING NOT NULL,
    product_description STRING NOT NULL,
    unit_price          FLOAT,
    currency            STRING,
    vendor_country      STRING,
    created_date        DATE,
    iso_code            STRING          -- Known label (training data)
);
```

> **See:** [`src/00_setup_data.py`](src/00_setup_data.py) for the full synthetic data generator with 15 vendors and 55 ISO codes.

---

## Step 2: Train the ML Classifier

The classifier uses three feature types combined into a single feature matrix:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.sparse import hstack

# 1. Text features: TF-IDF on product descriptions
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english")
X_text = tfidf.fit_transform(df["product_description"])

# 2. Categorical features: vendor one-hot encoding
X_vendor = pd.get_dummies(df["vendor_name"], prefix="vendor")

# 3. Numeric features: price normalized to EUR
scaler = StandardScaler()
X_price = scaler.fit_transform(df[["price_eur"]])

# Combined feature matrix
X = hstack([X_text, X_vendor.values, X_price])
```

Three models are compared via MLflow experiment tracking:

```python
models = {
    "LogisticRegression": LogisticRegression(C=1.0, max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=30),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=150, max_depth=6),
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("test_accuracy", accuracy)
```

The best model is registered in Unity Catalog with the `Champion` alias:

```python
mlflow.sklearn.log_model(model, "model", registered_model_name="catalog.schema.item_classifier")
client.set_registered_model_alias("catalog.schema.item_classifier", "Champion", version)
```

> **See:** [`src/01_train_classifier.py`](src/01_train_classifier.py) for the full training notebook.

---

## Step 3: The Hybrid Scoring Pipeline

This is the core pattern. Three phases in a single notebook.

### Phase 1: ML Batch Scoring

```python
# Load Champion model from UC registry
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@Champion")

# Score all new items
y_pred = model.predict(X)
y_proba = model.predict_proba(X)
max_confidence = np.max(y_proba, axis=1)

df_new["ml_predicted_iso"] = label_encoder.inverse_transform(y_pred)
df_new["ml_confidence"] = np.round(max_confidence, 4)
```

### Phase 2: Confidence Gate

```python
CONFIDENCE_THRESHOLD = 0.85

high_conf = df_new[df_new["ml_confidence"] >= CONFIDENCE_THRESHOLD]  # → Accept
low_conf  = df_new[df_new["ml_confidence"] < CONFIDENCE_THRESHOLD]   # → LLM
```

Typical split: 80-90% accepted by ML, 10-20% routed to LLM.

### Phase 3: LLM Fallback via AI_QUERY

The key SQL pattern. `AI_QUERY` calls a Foundation Model endpoint with:
- The item details (description, vendor, price)
- The ML model's uncertain prediction
- The full ISO code reference table
- A `responseFormat` schema for structured JSON output

```sql
-- Store ISO reference as a single-row view (avoids SQL injection from string interpolation)
-- In Python: build iso_ref from the iso_codes table, then:
--   spark.sql("SELECT ... AS iso_ref").createOrReplaceTempView("iso_reference")

SELECT
    l.item_id,
    l.product_description,
    AI_QUERY(
        'databricks-meta-llama-3-3-70b-instruct',
        CONCAT(
            'You are a medical device classification expert. ',
            'Classify this item into the correct ISO 9999 code.\n\n',
            'ITEM: ', l.product_description, '\n',
            'VENDOR: ', l.vendor_name, '\n',
            'PRICE: ', CAST(l.unit_price AS STRING), ' ', l.currency, '\n',
            'ML SUGGESTION: ', l.ml_predicted_iso,
            ' (confidence: ', CAST(ROUND(l.ml_confidence, 2) AS STRING), ')\n\n',
            'AVAILABLE ISO CODES:\n',
            r.iso_ref, '\n\n',  -- Full code list from temp view (safe, no interpolation)
            'Return the best matching ISO code and reason.'
        ),
        responseFormat => '{"type": "json_schema", "json_schema": {"name": "iso_classification", "schema": {"type": "object", "properties": {"iso_code": {"type": "string"}, "reason": {"type": "string"}, "agrees_with_ml": {"type": "boolean"}}, "required": ["iso_code", "reason", "agrees_with_ml"]}, "strict": true}}',
    ) AS llm_response
FROM low_confidence_items l
CROSS JOIN iso_reference r
```

Parse the structured response:

```sql
SELECT
    item_id,
    llm_response:iso_code AS final_iso_code,
    llm_response:reason AS classification_reason,
    llm_response:agrees_with_ml AS llm_agrees_with_ml
FROM llm_classified
```

### Phase 4: Merge Results

```python
# High-conf items: use ML prediction
high_result["final_iso_code"] = high_result["ml_predicted_iso"]
high_result["classification_method"] = "ml_classifier"

# Low-conf items: use LLM prediction
low_result["final_iso_code"] = low_result["llm_iso_code"]
low_result["classification_method"] = "llm_fallback"

# Combine into unified output table
classified = pd.concat([high_result, low_result])
classified.write.saveAsTable("catalog.schema.classified_items")
```

> **See:** [`src/02_hybrid_score_pipeline.py`](src/02_hybrid_score_pipeline.py) for the full implementation.

---

## Step 4: Quick Classification with ai_classify()

For broad categorization (ISO classes, not 6-digit codes), `ai_classify()` requires zero training:

```sql
-- ai_classify takes a JSON string: array for simple labels, object for label+description
SELECT
    product_description,
    ai_classify(
        product_description,
        '{
            "Orthoses — spinal": "Spinal braces, lumbar supports, cervical collars, TLSO",
            "Orthoses — upper limb": "Hand splints, wrist braces, elbow supports",
            "Orthoses — lower limb": "AFOs, knee braces, KAFOs, foot orthoses",
            "Prostheses — upper limb": "Myoelectric hands, transradial, transhumeral",
            "Prostheses — lower limb": "BK prostheses, AK prostheses, knee units",
            "Compression therapy": "Compression stockings, arm sleeves",
            "Wound care": "Dressings, NPWT, wound closure",
            "Mobility aids": "Wheelchairs, crutches, rollators"
        }',
        MAP('version', '2.0')
    ) AS broad_category
FROM new_items
```

Use `ai_classify()` when:
- You don't have labeled training data yet
- You need broad categories, not specific codes
- You're prototyping and want quick results

Use the hybrid ML + AI_QUERY approach when:
- You need specific 6-digit ISO codes
- You have labeled training data
- Cost matters (ML is free, LLM costs per token)
- You need auditability (confidence scores, reasoning)

---

## Step 5: Monitoring & Retraining

### Monitoring Signals

| Signal | Threshold | What It Means |
|--------|-----------|---------------|
| Avg ML confidence | < 0.75 | Model is losing certainty on new data |
| Low-conf volume | > 25% | Too many items hitting LLM fallback |
| LLM override rate | > 50% | ML predictions are frequently wrong |
| Per-code weakness | < 0.70 | Specific ISO codes are problematic |

### The Feedback Loop

```
Score → Monitor → Drift? ──NO──► Done
                    │
                   YES
                    │
              Enough corrections? ──NO──► Alert: need more expert feedback
                    │
                   YES
                    │
              Retrain with augmented data
                    │
              New > Champion? ──NO──► Store as Challenger
                    │
                   YES
                    │
              Promote to Champion
```

### Correction-Based Retraining

```python
# Original labeled data minus corrected items + expert corrections
corrected_ids = set(corrections["item_id"])
df_augmented = df_labeled[~df_labeled["item_id"].isin(corrected_ids)]

for _, correction in corrections.iterrows():
    item = get_item_features(correction["item_id"])
    item["iso_code"] = correction["corrected_iso_code"]  # Expert label
    df_augmented = pd.concat([df_augmented, pd.DataFrame([item])])

# Retrain with same algorithm
model = RandomForestClassifier(n_estimators=200, max_depth=30)
model.fit(X_train_augmented, y_train_augmented)

# Promote only if better
if new_accuracy >= champion_accuracy:
    client.set_registered_model_alias(MODEL_NAME, "Champion", new_version)
```

### Lakehouse Monitor (Optional)

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
w.quality_monitors.create(
    table_name="catalog.schema.classified_items",
    inference_log=ml.InferenceLog(
        problem_type="classification",
        prediction_col="final_iso_code",
        model_id_col="model_version",
        timestamp_col="scored_at",
    ),
    schedule=ml.MonitorCronSchedule(
        quartz_cron_expression="0 0 8 * * ?",
        timezone_id="Europe/Oslo",
    ),
)
```

> **See:** [`src/03_monitor_and_retrain.py`](src/03_monitor_and_retrain.py) for the full implementation.

---

## Cost Analysis

| Component | Cost Driver | Typical Volume | Est. Cost |
|-----------|------------|----------------|-----------|
| ML scoring | Serverless compute | 500 items/run | ~$0.02 |
| AI_QUERY (LLM) | Token usage | ~75 items (15%) | ~$0.15 |
| Training | Serverless compute | 1,500 rows | ~$0.05 |
| Monitoring | Serverless compute | Daily | ~$0.03 |
| **Total per run** | | | **~$0.25** |

Compare to pure LLM: 500 items × AI_QUERY = ~$0.50/run. The hybrid approach cuts LLM costs by 85%.

---

## Production Deployment

### As a Databricks Job (DAB)

```yaml
# databricks.yml
resources:
  jobs:
    hybrid_classifier_pipeline:
      tasks:
        - task_key: setup_data
          notebook_task:
            notebook_path: src/00_setup_data.py
        - task_key: train_classifier
          depends_on: [setup_data]
          notebook_task:
            notebook_path: src/01_train_classifier.py
        - task_key: hybrid_score
          depends_on: [train_classifier]
          notebook_task:
            notebook_path: src/02_hybrid_score_pipeline.py
        - task_key: monitor_retrain
          depends_on: [hybrid_score]
          notebook_task:
            notebook_path: src/03_monitor_and_retrain.py
```

Deploy:
```bash
databricks bundle deploy -t dev
databricks bundle run hybrid_classifier_pipeline -t dev
```

### Write-Back to Master Data (Profisee)

For writing classified results back to external systems:

```python
# Option A: Delta Sharing (recommended for governed data exchange)
spark.sql("""
    CREATE SHARE IF NOT EXISTS classified_items_share;
    ALTER SHARE classified_items_share ADD TABLE catalog.schema.classified_items;
""")

# Option B: JDBC write-back
classified_df.write \
    .format("jdbc") \
    .option("url", "jdbc:sqlserver://profisee-host:1433;database=master_data") \
    .option("dbtable", "dbo.classified_items") \
    .mode("append") \
    .save()

# Option C: File export to Azure Blob / ADLS
classified_df.write.mode("overwrite").parquet("/mnt/exports/classified_items/")
```

---

## Production Tips

**Scaling AI_QUERY for larger batches:**
- Use `failOnError => false` for 1K+ rows — returns errors per row instead of failing the whole query. Note: this changes the return type to `STRUCT<response, errorMessage>`, so adjust your parsing.
- Don't manually batch — submit the full dataset in one query and let the platform handle parallelization.
- For throughput debugging: `go/batchinference/debug` dashboard. For higher limits: `go/batchlimitincrease`.

**Scaling ML scoring:**
- For larger datasets (10K+ rows), use `mlflow.pyfunc.spark_udf()` instead of pandas-based `predict()`:
  ```python
  predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri, env_manager='local')
  scored = df.withColumn("prediction", predict_udf(struct(*feature_cols)))
  ```
- Use `env_manager='local'` to avoid virtualenv overhead on serverless.
- Score against Delta tables, not views (avoids a known hanging-job bug on MLR 16.4).

**ai_classify confidence scores (coming soon):**
- As of July 2026, `ai_classify` does not return confidence scores (only `ai_extract` does).
- Engineering is building v2.1 with confidence support — no public ETA yet.
- Workaround: use `AI_QUERY` with structured output if you need LLM confidence.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| sklearn over SparkML | Simpler for ~2K training rows. SparkML adds overhead without benefit at this scale. |
| AI_QUERY over ai_classify | We need specific 6-digit codes, not broad categories. AI_QUERY with responseFormat gives structured output. |
| 0.85 confidence threshold | Balances accuracy vs cost. Lower = more LLM calls. Higher = more missed errors. Tune based on your data. |
| RandomForest as default | Consistently best accuracy on TF-IDF + categorical features. Easy to explain to stakeholders. |
| Corrections-based retraining | No regression allowed — new model must beat Champion. Prevents pushing worse models. |

---

## Repository Structure

```
medtech-hybrid-classifier/
├── databricks.yml                        # DAB bundle config
├── SOLUTION_GUIDE.md                     # This document
├── README.md                             # Quick start
├── src/
│   ├── 00_setup_data.py                  # Tables + synthetic data
│   ├── 01_train_classifier.py            # Train + register in UC
│   ├── 02_hybrid_score_pipeline.py       # ML → confidence gate → LLM
│   └── 03_monitor_and_retrain.py         # Feedback loop
└── resources/
    └── hybrid_classifier_job.yml         # Job definition
```
