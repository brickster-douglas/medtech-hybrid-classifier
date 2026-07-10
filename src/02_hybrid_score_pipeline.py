# Databricks notebook source
# MAGIC %md
# MAGIC # Step 2: Hybrid Scoring Pipeline
# MAGIC
# MAGIC The core of the solution: a two-tier classification system.
# MAGIC
# MAGIC ```
# MAGIC New Items
# MAGIC    │
# MAGIC    ▼
# MAGIC ┌─────────────────┐
# MAGIC │  ML Classifier   │  sklearn model from UC Registry
# MAGIC │  (fast, cheap)   │
# MAGIC └────────┬────────┘
# MAGIC          │
# MAGIC    ┌─────┴─────┐
# MAGIC    │           │
# MAGIC    ▼           ▼
# MAGIC conf ≥ 0.85  conf < 0.85
# MAGIC    │           │
# MAGIC    │     ┌─────┴──────┐
# MAGIC    │     │ AI_QUERY   │  LLM with ISO code context
# MAGIC    │     │ (accurate, │  structured JSON response
# MAGIC    │     │  costly)   │
# MAGIC    │     └─────┬──────┘
# MAGIC    │           │
# MAGIC    ▼           ▼
# MAGIC ┌─────────────────┐
# MAGIC │  classified_items │  Final unified output
# MAGIC └─────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

try:
    CATALOG = spark.conf.get("bundle.var.catalog")
except Exception:
    CATALOG = "serverless_stable_m3qkky_catalog"
try:
    SCHEMA = spark.conf.get("bundle.var.schema")
except Exception:
    SCHEMA = "embla_hybrid_classifier"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.item_classifier"

CONFIDENCE_THRESHOLD = 0.85
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"  # or your fine-tuned endpoint

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 1: ML Classifier Batch Scoring

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from datetime import datetime

mlflow.set_registry_uri("databricks-uc")

# Load Champion model
model_uri = f"models:/{MODEL_NAME}@Champion"
model = mlflow.sklearn.load_model(model_uri)
print(f"Loaded Champion model from {model_uri}")

# Load inference artifacts (TF-IDF, label encoder, scaler, vendor columns)
client = mlflow.MlflowClient()
champion_version = client.get_model_version_by_alias(MODEL_NAME, "Champion")
run_id = champion_version.run_id
artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="inference_artifacts.pkl")

with open(artifact_path, "rb") as f:
    artifacts = pickle.load(f)

tfidf = artifacts["tfidf"]
label_encoder = artifacts["label_encoder"]
scaler = artifacts["scaler"]
vendor_columns = artifacts["vendor_columns"]

# COMMAND ----------

# Load new items
df_new = spark.table(f"{CATALOG}.{SCHEMA}.new_items").toPandas()
print(f"Scoring {len(df_new)} new items")

# FX normalization
fx_to_eur = {"EUR": 1.0, "USD": 0.92, "ISK": 0.0065, "SEK": 0.087, "NOK": 0.086, "DKK": 0.134, "GBP": 1.17}
df_new["price_eur"] = df_new.apply(lambda r: r["unit_price"] * fx_to_eur.get(r["currency"], 1.0), axis=1)

# Feature engineering (must match training exactly)
X_text = tfidf.transform(df_new["product_description"])
X_vendor = pd.get_dummies(df_new["vendor_name"], prefix="vendor")
for col in vendor_columns:
    if col not in X_vendor.columns:
        X_vendor[col] = 0
X_vendor = X_vendor[vendor_columns]
X_price = scaler.transform(df_new[["price_eur"]])

X = hstack([X_text, X_vendor.values, X_price])

# Predict with confidence scores
y_pred = model.predict(X)
y_proba = model.predict_proba(X)
max_confidence = np.max(y_proba, axis=1)

df_new["ml_predicted_iso"] = label_encoder.inverse_transform(y_pred)
df_new["ml_confidence"] = np.round(max_confidence, 4)
df_new["scored_at"] = datetime.now()
df_new["model_version"] = champion_version.version

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 2: Split by Confidence

# COMMAND ----------

high_conf = df_new[df_new["ml_confidence"] >= CONFIDENCE_THRESHOLD].copy()
low_conf = df_new[df_new["ml_confidence"] < CONFIDENCE_THRESHOLD].copy()

print(f"High confidence (≥{CONFIDENCE_THRESHOLD}): {len(high_conf)} items ({len(high_conf)/len(df_new)*100:.1f}%)")
print(f"Low confidence  (<{CONFIDENCE_THRESHOLD}): {len(low_conf)} items ({len(low_conf)/len(df_new)*100:.1f}%) → LLM fallback")

# High-confidence items are accepted directly
high_conf["final_iso_code"] = high_conf["ml_predicted_iso"]
high_conf["classification_method"] = "ml_classifier"
high_conf["final_confidence"] = high_conf["ml_confidence"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 3: LLM Fallback via AI_QUERY
# MAGIC
# MAGIC For low-confidence items, we use `AI_QUERY` with:
# MAGIC 1. The full ISO code reference table as context
# MAGIC 2. A structured prompt with the item description, vendor, and price
# MAGIC 3. `responseFormat` for structured JSON output

# COMMAND ----------

# Write low-confidence items to a temp table for SQL-based AI_QUERY
if len(low_conf) > 0:
    low_conf_for_sql = low_conf[["item_id", "vendor_name", "product_description", "unit_price", "currency", "ml_predicted_iso", "ml_confidence"]].copy()
    spark.createDataFrame(low_conf_for_sql).createOrReplaceTempView("low_confidence_items")

    # Build the ISO code reference string for the prompt
    iso_ref = spark.table(f"{CATALOG}.{SCHEMA}.iso_codes").toPandas()
    iso_ref_str = "\n".join(
        f"- {row['iso_code']}: {row['name']} — {row['description']}"
        for _, row in iso_ref.iterrows()
    )

    print(f"Sending {len(low_conf)} items to LLM for reclassification...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### AI_QUERY with Structured Output
# MAGIC
# MAGIC This is the key pattern: we ask the LLM to return a JSON object with the ISO code and its reasoning.
# MAGIC The `responseFormat` parameter enforces the schema so we always get parseable output.

# COMMAND ----------

if len(low_conf) > 0:
    # Store ISO reference in a temp view to avoid SQL injection from string interpolation
    iso_ref_rows = [
        (f"- {row['iso_code']}: {row['name']} — {row['description']}",)
        for _, row in iso_ref.iterrows()
    ]
    spark.createDataFrame(iso_ref_rows, ["line"]).createOrReplaceTempView("iso_reference_lines")
    spark.sql("SELECT CONCAT_WS('\\n', COLLECT_LIST(line)) AS iso_ref FROM iso_reference_lines").createOrReplaceTempView("iso_reference")

    # Build the AI_QUERY SQL — all string values come from columns, not Python interpolation
    response_format = '{"type": "json_schema", "json_schema": {"name": "iso_classification", "schema": {"type": "object", "properties": {"iso_code": {"type": "string", "description": "The 6-digit ISO 9999 code (e.g. 06 24 09)"}, "reason": {"type": "string", "description": "Brief explanation for the classification"}, "agrees_with_ml": {"type": "boolean", "description": "Whether this matches the ML prediction"}}, "required": ["iso_code", "reason", "agrees_with_ml"]}, "strict": true}}'

    llm_results = spark.sql(f"""
        SELECT
            l.item_id,
            l.vendor_name,
            l.product_description,
            l.unit_price,
            l.currency,
            l.ml_predicted_iso,
            l.ml_confidence,
            AI_QUERY(
                '{LLM_ENDPOINT}',
                CONCAT(
                    'You are a medical device classification expert. Classify the following vendor item into the correct ISO 9999 code.\\n\\n',
                    'ITEM DETAILS:\\n',
                    '- Description: ', l.product_description, '\\n',
                    '- Vendor: ', l.vendor_name, '\\n',
                    '- Price: ', CAST(l.unit_price AS STRING), ' ', l.currency, '\\n',
                    '- ML model suggested: ', l.ml_predicted_iso, ' (confidence: ', CAST(ROUND(l.ml_confidence, 2) AS STRING), ')\\n\\n',
                    'AVAILABLE ISO 9999 CODES:\\n',
                    r.iso_ref, '\\n\\n',
                    'Return ONLY the best matching ISO code and a brief reason.'
                ),
                responseFormat => '{response_format}'
            ) AS llm_response
        FROM low_confidence_items l
        CROSS JOIN iso_reference r
    """)

    llm_results.createOrReplaceTempView("llm_classified")
    print("LLM classification complete")
    display(llm_results.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parse LLM Results

# COMMAND ----------

if len(low_conf) > 0:
    llm_parsed = spark.sql("""
        SELECT
            item_id,
            vendor_name,
            product_description,
            unit_price,
            currency,
            ml_predicted_iso,
            ml_confidence,
            llm_response:iso_code AS llm_iso_code,
            llm_response:reason AS llm_reason,
            llm_response:agrees_with_ml AS llm_agrees
        FROM llm_classified
    """).toPandas()

    # Fall back to ML prediction if LLM response parsing failed (NULL)
    llm_parsed["final_iso_code"] = llm_parsed["llm_iso_code"].fillna(llm_parsed["ml_predicted_iso"])
    llm_parsed["classification_method"] = llm_parsed["llm_iso_code"].apply(
        lambda x: "llm_fallback" if pd.notna(x) else "ml_classifier_fallback"
    )
    llm_parsed["final_confidence"] = 0.80  # LLM predictions get a fixed confidence

    n_parsed = llm_parsed["llm_iso_code"].notna().sum()
    n_failed = llm_parsed["llm_iso_code"].isna().sum()
    if n_failed > 0:
        print(f"⚠ {n_failed} LLM responses failed to parse — falling back to ML prediction")

    llm_valid = llm_parsed[llm_parsed["llm_iso_code"].notna()]
    if len(llm_valid) > 0:
        agrees = llm_valid["llm_agrees"].astype(bool).sum()
        print(f"LLM agreed with ML on {agrees}/{len(llm_valid)} items")
        print(f"LLM overrode ML on {len(llm_valid) - agrees}/{len(llm_valid)} items")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 4: Merge Results

# COMMAND ----------

# Prepare high-conf results
high_result = high_conf[["item_id", "vendor_name", "product_description", "unit_price",
                          "currency", "ml_predicted_iso", "ml_confidence",
                          "final_iso_code", "classification_method", "final_confidence",
                          "scored_at", "model_version"]].copy()
high_result["llm_reason"] = None

if len(low_conf) > 0:
    # Prepare low-conf results
    low_result = llm_parsed[["item_id", "vendor_name", "product_description", "unit_price",
                              "currency", "ml_predicted_iso", "ml_confidence",
                              "final_iso_code", "classification_method", "final_confidence",
                              "llm_reason"]].copy()
    low_result["scored_at"] = datetime.now()
    low_result["model_version"] = champion_version.version

    # Combine
    classified = pd.concat([high_result, low_result], ignore_index=True)
else:
    classified = high_result

print(f"\nFinal classified items: {len(classified)}")
print(f"  ML classifier: {(classified['classification_method'] == 'ml_classifier').sum()}")
print(f"  LLM fallback:  {(classified['classification_method'] == 'llm_fallback').sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Final Output

# COMMAND ----------

df_classified = spark.createDataFrame(classified)
df_classified.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.classified_items")
print(f"Wrote classified_items: {len(classified)} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate Against Ground Truth (Demo Only)
# MAGIC
# MAGIC Since we generated the synthetic data with known labels, we can measure end-to-end accuracy.

# COMMAND ----------

ground_truth = spark.table(f"{CATALOG}.{SCHEMA}.new_items").select("item_id", "_ground_truth_iso").toPandas()
eval_df = classified.merge(ground_truth, on="item_id", how="left")

if "_ground_truth_iso" in eval_df.columns:
    eval_df["correct"] = eval_df["final_iso_code"] == eval_df["_ground_truth_iso"]

    total_acc = eval_df["correct"].mean()
    ml_acc = eval_df[eval_df["classification_method"] == "ml_classifier"]["correct"].mean()

    print(f"\n{'='*50}")
    print(f"END-TO-END HYBRID ACCURACY: {total_acc:.1%}")
    print(f"  ML-only accuracy:         {ml_acc:.1%}")

    llm_rows = eval_df[eval_df["classification_method"] == "llm_fallback"]
    if len(llm_rows) > 0:
        llm_acc = llm_rows["correct"].mean()
        print(f"  LLM fallback accuracy:    {llm_acc:.1%}")
        print(f"  LLM improved over ML:     {(llm_rows['final_iso_code'] != llm_rows['ml_predicted_iso']).sum()} items reclassified")
    print(f"{'='*50}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alternative: ai_classify() for Quick Categorization
# MAGIC
# MAGIC If you only need to classify into broad ISO **classes** (not 6-digit codes),
# MAGIC `ai_classify()` is simpler and requires no training:

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Quick classification into ISO classes using ai_classify (no training needed)
# MAGIC -- This is useful for triage or when you don't have labeled data yet
# MAGIC -- ai_classify takes a JSON string: array for simple labels, object for label+description
# MAGIC SELECT
# MAGIC   item_id,
# MAGIC   product_description,
# MAGIC   ai_classify(
# MAGIC     product_description,
# MAGIC     '{
# MAGIC       "Orthoses — spinal": "Spinal braces, lumbar supports, cervical collars, TLSO",
# MAGIC       "Orthoses — upper limb": "Hand splints, wrist braces, elbow supports, shoulder immobilizers",
# MAGIC       "Orthoses — lower limb": "AFOs, knee braces, KAFOs, foot orthoses, hip braces",
# MAGIC       "Prostheses — upper limb": "Hand prostheses, transradial, transhumeral, myoelectric arms",
# MAGIC       "Prostheses — lower limb": "Transtibial, transfemoral, knee units, prosthetic feet, liners",
# MAGIC       "Prostheses — non-limb": "Breast forms, ocular, auricular, nasal prostheses",
# MAGIC       "Compression therapy": "Compression stockings, arm sleeves, anti-embolism",
# MAGIC       "Wound care": "Dressings, wound closure, NPWT, irrigation",
# MAGIC       "Mobility aids": "Wheelchairs, crutches, rollators, walking sticks",
# MAGIC       "Therapeutic footwear": "Orthopedic shoes, insoles, therapeutic footwear"
# MAGIC     }',
# MAGIC     MAP('version', '2.0')
# MAGIC   ) AS broad_category
# MAGIC FROM new_items
# MAGIC LIMIT 10
