# Databricks notebook source
# MAGIC %md
# MAGIC # Step 3: Monitor & Retrain
# MAGIC
# MAGIC Closes the feedback loop:
# MAGIC 1. **Monitor** confidence degradation and drift signals
# MAGIC 2. **Incorporate** human corrections into training data
# MAGIC 3. **Retrain** model and promote if it beats the Champion
# MAGIC
# MAGIC ```
# MAGIC classified_items ──► Monitor ──► Drift detected?
# MAGIC                                       │
# MAGIC                                  YES  │  NO
# MAGIC                                       │   └─► Done
# MAGIC                                       ▼
# MAGIC                               Enough corrections?
# MAGIC                                       │
# MAGIC                                  YES  │  NO
# MAGIC                                       │   └─► Alert (need more feedback)
# MAGIC                                       ▼
# MAGIC                               Retrain with corrections
# MAGIC                                       │
# MAGIC                               New > Champion?
# MAGIC                                       │
# MAGIC                                  YES  │  NO
# MAGIC                                       │   └─► Keep as Challenger
# MAGIC                                       ▼
# MAGIC                               Promote to Champion
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
    SCHEMA = "medtech_hybrid_classifier"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.item_classifier"

# Monitoring thresholds
AVG_CONFIDENCE_FLOOR = 0.75     # Alert if avg confidence drops below this
LOW_CONF_CEILING_PCT = 25.0     # Alert if >25% of items are below 0.85
MIN_CORRECTIONS_TO_RETRAIN = 10 # Need at least 10 corrections before retraining

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitor: Confidence & Drift Signals

# COMMAND ----------

import pandas as pd

classified = spark.table(f"{CATALOG}.{SCHEMA}.classified_items").toPandas()
corrections = spark.table(f"{CATALOG}.{SCHEMA}.human_corrections").toPandas()

# Compute drift signals only on ML-classified items (LLM items are low-conf by design)
ml_classified = classified[classified["classification_method"] == "ml_classifier"]

# Signal 1: Average confidence (ML items only)
avg_conf = ml_classified["ml_confidence"].mean() if len(ml_classified) > 0 else 0
conf_drift = avg_conf < AVG_CONFIDENCE_FLOOR

# Signal 2: Low confidence volume (what % of ALL items needed LLM fallback)
llm_fallback_pct = (classified["classification_method"] != "ml_classifier").mean() * 100
volume_drift = llm_fallback_pct > LOW_CONF_CEILING_PCT

# Signal 3: LLM override rate
llm_items = classified[classified["classification_method"] == "llm_fallback"]
llm_override_rate = (llm_items["final_iso_code"] != llm_items["ml_predicted_iso"]).mean() * 100 if len(llm_items) > 0 else 0

# Signal 4: Per-ISO-code weakness
per_code_conf = classified.groupby("final_iso_code")["ml_confidence"].mean()
weak_codes = per_code_conf[per_code_conf < 0.70].sort_values()

print("=" * 60)
print("MONITORING REPORT")
print("=" * 60)
print(f"Total items classified:    {len(classified)}")
print(f"ML classifier:             {(classified['classification_method'] == 'ml_classifier').sum()}")
print(f"LLM fallback:              {(classified['classification_method'] == 'llm_fallback').sum()}")
print(f"")
print(f"Avg ML confidence:         {avg_conf:.4f} {'⚠ DRIFT' if conf_drift else '✓ OK'}")
print(f"LLM fallback rate:         {llm_fallback_pct:.1f}% {'⚠ HIGH' if volume_drift else '✓ OK'}")
print(f"LLM override rate:         {llm_override_rate:.1f}%")
print(f"Human corrections:         {len(corrections)}")
print(f"")

if len(weak_codes) > 0:
    print("Weakest ISO codes (avg confidence < 0.70):")
    for code, conf in weak_codes.items():
        print(f"  {code}: {conf:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision: Should We Retrain?

# COMMAND ----------

should_retrain = (conf_drift or volume_drift) and len(corrections) >= MIN_CORRECTIONS_TO_RETRAIN

print(f"\nRetrain decision:")
print(f"  Confidence drift:  {'YES' if conf_drift else 'NO'}")
print(f"  Volume drift:      {'YES' if volume_drift else 'NO'}")
print(f"  Enough corrections: {'YES' if len(corrections) >= MIN_CORRECTIONS_TO_RETRAIN else f'NO ({len(corrections)}/{MIN_CORRECTIONS_TO_RETRAIN})'}")
print(f"  → RETRAIN: {'YES' if should_retrain else 'NO'}")

if not should_retrain:
    if conf_drift or volume_drift:
        print(f"\n⚠ Drift detected but only {len(corrections)} corrections available.")
        print(f"  Need {MIN_CORRECTIONS_TO_RETRAIN - len(corrections)} more expert corrections before retraining.")
    else:
        print("\n✓ Model is performing within acceptable thresholds.")
    dbutils.notebook.exit(f"RETRAIN:NO|AVG_CONF:{avg_conf:.4f}|LOW_PCT:{llm_fallback_pct:.1f}|CORRECTIONS:{len(corrections)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrain with Corrections
# MAGIC
# MAGIC The retraining strategy:
# MAGIC 1. Start with original labeled data
# MAGIC 2. For corrected items: replace original label with expert correction
# MAGIC 3. Train new model with same algorithm + hyperparameters
# MAGIC 4. Promote only if new model beats Champion accuracy

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

mlflow.set_registry_uri("databricks-uc")

# Load original training data
df_labeled = spark.table(f"{CATALOG}.{SCHEMA}.labeled_items").toPandas()

# Build augmented dataset: original labels + expert corrections
corrected_ids = set(corrections["item_id"])
df_augmented = df_labeled[~df_labeled["item_id"].isin(corrected_ids)].copy()

# Add corrected items (find their features from classified_items, apply expert label)
labeled_cols = ["item_id", "vendor_name", "product_description", "unit_price", "currency", "vendor_country", "created_date", "iso_code"]
for _, correction in corrections.iterrows():
    item_row = classified[classified["item_id"] == correction["item_id"]]
    if len(item_row) > 0:
        item = {col: item_row.iloc[0].get(col) for col in labeled_cols if col != "iso_code"}
        item["iso_code"] = correction["corrected_iso_code"]
        df_augmented = pd.concat([df_augmented, pd.DataFrame([item])], ignore_index=True)

print(f"Augmented training set: {len(df_augmented)} rows ({len(corrections)} corrections applied)")

# COMMAND ----------

# Feature engineering (same pipeline as training)
fx_to_eur = {"EUR": 1.0, "USD": 0.92, "ISK": 0.0065, "SEK": 0.087, "NOK": 0.086, "DKK": 0.134, "GBP": 1.17}
df_augmented["price_eur"] = df_augmented.apply(
    lambda r: r["unit_price"] * fx_to_eur.get(r["currency"], 1.0), axis=1
)

tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english")
X_text = tfidf.fit_transform(df_augmented["product_description"])

X_vendor = pd.get_dummies(df_augmented["vendor_name"], prefix="vendor")
vendor_columns = list(X_vendor.columns)

scaler = StandardScaler()
X_price = scaler.fit_transform(df_augmented[["price_eur"]])

X = hstack([X_text, X_vendor.values, X_price])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_augmented["iso_code"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train and Evaluate

# COMMAND ----------

experiment_name = f"/Users/{spark.sql('SELECT current_user()').first()[0]}/medtech-hybrid-classifier"
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="retrained_with_corrections") as run:
    model = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    new_accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_param("algorithm", "RandomForest")
    mlflow.log_param("training_rows", len(df_augmented))
    mlflow.log_param("corrections_included", len(corrections))
    mlflow.log_param("retrain_reason", "drift_detected")
    mlflow.log_metric("test_accuracy", new_accuracy)

    # Save artifacts
    artifacts = {
        "tfidf": tfidf,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "vendor_columns": vendor_columns,
        "confidence_threshold": 0.85,
    }
    with open("/tmp/inference_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    mlflow.log_artifact("/tmp/inference_artifacts.pkl")

    from mlflow.models import infer_signature
    sample_input = X_test[:5]
    sample_output = model.predict(sample_input)
    signature = infer_signature(sample_input, sample_output)

    model_info = mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        registered_model_name=MODEL_NAME,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conditional Promotion

# COMMAND ----------

client = mlflow.MlflowClient()
champion_version = client.get_model_version_by_alias(MODEL_NAME, "Champion")
champion_run = client.get_run(champion_version.run_id)
champion_accuracy = champion_run.data.metrics.get("test_accuracy", 0)

print(f"Champion accuracy: {champion_accuracy:.4f}")
print(f"New model accuracy: {new_accuracy:.4f}")

if new_accuracy >= champion_accuracy:
    # Promote new model
    latest_version = max(
        client.search_model_versions(f"name='{MODEL_NAME}'"),
        key=lambda v: int(v.version),
    )
    client.set_registered_model_alias(MODEL_NAME, "Champion", latest_version.version)
    client.update_model_version(
        MODEL_NAME,
        latest_version.version,
        description=f"Retrained on {len(df_augmented)} samples (+{len(corrections)} corrections). Accuracy: {new_accuracy:.4f}",
    )
    print(f"\n✓ NEW CHAMPION: version {latest_version.version} (accuracy: {new_accuracy:.4f})")
else:
    latest_version = max(
        client.search_model_versions(f"name='{MODEL_NAME}'"),
        key=lambda v: int(v.version),
    )
    client.set_registered_model_alias(MODEL_NAME, "Challenger", latest_version.version)
    print(f"\n✗ New model ({new_accuracy:.4f}) did not beat Champion ({champion_accuracy:.4f})")
    print(f"  Stored as Challenger (version {latest_version.version})")

dbutils.notebook.exit(f"RETRAIN:YES|OLD_ACC:{champion_accuracy:.4f}|NEW_ACC:{new_accuracy:.4f}|PROMOTED:{'YES' if new_accuracy >= champion_accuracy else 'NO'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix: Lakehouse Monitor Setup (Optional)
# MAGIC
# MAGIC Attach a Databricks Lakehouse Monitor to the classified_items table for
# MAGIC automated drift detection on a schedule.

# COMMAND ----------

# MAGIC %md
# MAGIC ```python
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC
# MAGIC w = WorkspaceClient()
# MAGIC
# MAGIC monitor = w.quality_monitors.create(
# MAGIC     table_name=f"{CATALOG}.{SCHEMA}.classified_items",
# MAGIC     assets_dir=f"/Workspace/Users/{user}/monitors/medtech_classifier",
# MAGIC     output_schema_name=f"{CATALOG}.{SCHEMA}",
# MAGIC     inference_log=ml.InferenceLog(
# MAGIC         problem_type="classification",
# MAGIC         prediction_col="final_iso_code",
# MAGIC         model_id_col="model_version",
# MAGIC         timestamp_col="scored_at",
# MAGIC     ),
# MAGIC     schedule=ml.MonitorCronSchedule(
# MAGIC         quartz_cron_expression="0 0 8 * * ?",  # Daily at 8 AM
# MAGIC         timezone_id="Europe/Oslo",
# MAGIC     ),
# MAGIC )
# MAGIC print(f"Monitor created: {monitor.monitor_name}")
# MAGIC ```
