# Databricks notebook source
# MAGIC %md
# MAGIC # Step 1: Train ML Classifier
# MAGIC
# MAGIC Trains a multi-class text classifier to predict ISO 9999 codes from vendor item descriptions.
# MAGIC
# MAGIC **Features:**
# MAGIC - TF-IDF on product description (500 features, 1-2 grams)
# MAGIC - Vendor one-hot encoding
# MAGIC - Price normalized to EUR (StandardScaler)
# MAGIC
# MAGIC **Models compared:** Logistic Regression, Random Forest, Gradient Boosting
# MAGIC
# MAGIC Best model registered to Unity Catalog with `Champion` alias.

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

CONFIDENCE_THRESHOLD = 0.85  # Items below this go to LLM fallback

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and Prepare Data

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack
import mlflow
import mlflow.sklearn

mlflow.set_registry_uri("databricks-uc")

df = spark.table(f"{CATALOG}.{SCHEMA}.labeled_items").toPandas()
print(f"Training data: {len(df)} items, {df['iso_code'].nunique()} unique ISO codes")

# FX rates to EUR for price normalization
fx_to_eur = {"EUR": 1.0, "USD": 0.92, "ISK": 0.0065, "SEK": 0.087, "NOK": 0.086, "DKK": 0.134, "GBP": 1.17}
df["price_eur"] = df.apply(lambda r: r["unit_price"] * fx_to_eur.get(r["currency"], 1.0), axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering

# COMMAND ----------

# TF-IDF on product descriptions
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english")
X_text = tfidf.fit_transform(df["product_description"])

# Vendor one-hot encoding
X_vendor = pd.get_dummies(df["vendor_name"], prefix="vendor")

# Price feature (scaled)
scaler = StandardScaler()
X_price = scaler.fit_transform(df[["price_eur"]])

# Combine features
X = hstack([X_text, X_vendor.values, X_price])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["iso_code"])

print(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
print(f"Classes: {len(label_encoder.classes_)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Test Split

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare Models with MLflow

# COMMAND ----------

experiment_name = f"/Users/{spark.sql('SELECT current_user()').first()[0]}/embla-hybrid-classifier"
mlflow.set_experiment(experiment_name)

models = {
    "LogisticRegression": LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42),
}

results = {}
for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        avg_confidence = np.mean(np.max(y_proba, axis=1))
        low_conf_pct = np.mean(np.max(y_proba, axis=1) < CONFIDENCE_THRESHOLD) * 100

        mlflow.log_param("algorithm", name)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_classes", len(label_encoder.classes_))
        mlflow.log_param("confidence_threshold", CONFIDENCE_THRESHOLD)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("avg_confidence", avg_confidence)
        mlflow.log_metric("low_confidence_pct", low_conf_pct)

        results[name] = {
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "low_conf_pct": low_conf_pct,
            "run_id": run.info.run_id,
            "model": model,
        }

        print(f"{name}: accuracy={accuracy:.4f}, avg_conf={avg_confidence:.4f}, low_conf={low_conf_pct:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Best Model to Unity Catalog

# COMMAND ----------

best_name = max(results, key=lambda k: results[k]["accuracy"])
best = results[best_name]
print(f"\nBest model: {best_name} (accuracy={best['accuracy']:.4f})")

# Log the winning model with all artifacts needed for inference
with mlflow.start_run(run_name=f"champion_{best_name}") as run:
    import pickle

    # Package everything needed for inference
    artifacts = {
        "tfidf": tfidf,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "vendor_columns": list(X_vendor.columns),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }

    mlflow.log_param("algorithm", best_name)
    mlflow.log_metric("test_accuracy", best["accuracy"])
    mlflow.log_metric("avg_confidence", best["avg_confidence"])
    mlflow.log_metric("low_confidence_pct", best["low_conf_pct"])

    # Save artifacts as pickle
    with open("/tmp/inference_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    mlflow.log_artifact("/tmp/inference_artifacts.pkl")

    # Register model
    model_info = mlflow.sklearn.log_model(
        best["model"],
        artifact_path="model",
        registered_model_name=MODEL_NAME,
    )
    print(f"Registered model: {MODEL_NAME}")

# COMMAND ----------

# Set Champion alias on latest version
from mlflow import MlflowClient

client = MlflowClient()
latest_version = max(
    client.search_model_versions(f"name='{MODEL_NAME}'"),
    key=lambda v: int(v.version),
)
client.set_registered_model_alias(MODEL_NAME, "Champion", latest_version.version)
print(f"Set Champion alias → version {latest_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confidence Distribution Analysis
# MAGIC
# MAGIC This is critical for setting the right threshold. Items below the threshold
# MAGIC will be routed to the LLM fallback — too low = missed errors, too high = expensive LLM calls.

# COMMAND ----------

y_proba_test = best["model"].predict_proba(X_test)
max_conf = np.max(y_proba_test, axis=1)

bands = {
    "High (>=0.95)": np.sum(max_conf >= 0.95),
    "Good (0.85-0.95)": np.sum((max_conf >= 0.85) & (max_conf < 0.95)),
    "Medium (0.70-0.85)": np.sum((max_conf >= 0.70) & (max_conf < 0.85)),
    "Low (0.50-0.70)": np.sum((max_conf >= 0.50) & (max_conf < 0.70)),
    "Very Low (<0.50)": np.sum(max_conf < 0.50),
}

total = len(max_conf)
print(f"\nConfidence Distribution (threshold={CONFIDENCE_THRESHOLD}):")
print(f"{'Band':<25} {'Count':>6} {'Pct':>8} {'Action':<20}")
print("-" * 65)
for band, count in bands.items():
    action = "ML accepted" if "High" in band or "Good" in band else "→ LLM fallback"
    print(f"{band:<25} {count:>6} {count/total*100:>7.1f}% {action:<20}")

print(f"\nEstimated LLM fallback rate: {np.sum(max_conf < CONFIDENCE_THRESHOLD)/total*100:.1f}%")
print(f"This means ~{np.sum(max_conf < CONFIDENCE_THRESHOLD)/total*100:.0f}% of items will use AI_QUERY")
