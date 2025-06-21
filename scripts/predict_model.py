import pandas as pd
import joblib
import os
import json
from datetime import datetime
import argparse

# ---------------- Configuration ----------------
MODEL_DIR = "/opt/airflow/datamart/gold/models"
SCORING_DIR = "/opt/airflow/datamart/gold/scoring"
PREDICTIONS_DIR = "/opt/airflow/datamart/gold/predictions"
PREDICTION_TABLE_PATH = "/opt/airflow/datamart/gold/predictions/predictions_table.parquet"
FEATURE_PREFIX = "scoring_features_"
SNAPSHOT_PRED_PREFIX = "model_predictions_"
TABLE_PRED_PREFIX = "predictions_"
TRAINING_SNAPSHOT = "2023_12_01"

# ---------------- Prediction Function ----------------
def predict_model(model_type: str, execution_date: str):
    if not execution_date:
        raise ValueError("execution_date is required (YYYY-MM-DD)")
    if not model_type:
        raise ValueError("model_type is required (e.g., 'xgboost')")

    snapshot_str = execution_date.replace("-", "_")

    feature_path = os.path.join(SCORING_DIR, f"{FEATURE_PREFIX}{snapshot_str}.parquet")
    model_path = os.path.join(MODEL_DIR, model_type, TRAINING_SNAPSHOT, "model.pkl")
    schema_path = os.path.join(MODEL_DIR, model_type, TRAINING_SNAPSHOT, "feature_columns.json")
    snapshot_pred_path = os.path.join(SCORING_DIR, f"{SNAPSHOT_PRED_PREFIX}{model_type}_{snapshot_str}.parquet")
    table_pred_path = os.path.join(PREDICTIONS_DIR, f"{TABLE_PRED_PREFIX}{snapshot_str}.parquet")

    # Load model
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model not found at {model_path}")
        return
    model = joblib.load(model_path)

    # Load feature schema
    if not os.path.exists(schema_path):
        print(f"[ERROR] Feature schema not found at {schema_path}")
        return
    with open(schema_path, "r") as f:
        schema = json.load(f)
    expected_cols = schema["features"]

    # Load features
    if not os.path.exists(feature_path):
        print(f"[ERROR] Scoring features not found at {feature_path}")
        return
    df = pd.read_parquet(feature_path)
    if df.empty:
        print("[ERROR] Scoring features are empty.")
        return

    if "customer_id" not in df.columns:
        raise ValueError("Missing 'customer_id' in features")

    customer_ids = df["customer_id"].copy()

    # Drop non-features
    drop_cols = [col for col in ["customer_id", "label", "snapshot_date"] if col in df.columns]
    features = df.drop(columns=drop_cols)

    # Align columns to match training
    features = features.reindex(columns=expected_cols)

    # Convert all values to numeric and fill missing
    features = features.apply(pd.to_numeric, errors="coerce")
    features.fillna(0, inplace=True)

    if features.empty:
        print("[ERROR] Feature matrix is empty after preprocessing.")
        return

    # Generate predictions
    preds = model.predict(features)
    pred_df = pd.DataFrame({
        "customer_id": customer_ids.values,
        "prediction": preds,
        "model_type": model_type,
        "snapshot_date": execution_date,
        "prediction_ts": datetime.utcnow()
    })

    # Save snapshot-specific prediction
    os.makedirs(os.path.dirname(snapshot_pred_path), exist_ok=True)
    pred_df.to_parquet(snapshot_pred_path, index=False)
    print(f"[INFO] Snapshot prediction saved to: {snapshot_pred_path}")

    # Save to central daily predictions directory
    os.makedirs(os.path.dirname(table_pred_path), exist_ok=True)
    pred_df.to_parquet(table_pred_path, index=False)
    print(f"[INFO] Daily predictions saved to: {table_pred_path}")

    # Append to predictions_table.parquet
    if os.path.exists(PREDICTION_TABLE_PATH):
        try:
            existing = pd.read_parquet(PREDICTION_TABLE_PATH)
        except Exception as e:
            print(f"[WARNING] Failed to load existing predictions_table. Reinitializing: {e}")
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    combined = pd.concat([existing, pred_df], ignore_index=True)

    if {"customer_id", "snapshot_date"}.issubset(combined.columns):
        combined.drop_duplicates(subset=["customer_id", "snapshot_date"], keep="last", inplace=True)
    else:
        print("[WARNING] Cannot deduplicate predictions_table — missing required columns")

    os.makedirs(os.path.dirname(PREDICTION_TABLE_PATH), exist_ok=True)
    combined.to_parquet(PREDICTION_TABLE_PATH, index=False)
    print(f"[INFO] Updated prediction table saved to: {PREDICTION_TABLE_PATH}")

# ---------------- CLI Interface ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, help="Model type (e.g., xgboost)")
    parser.add_argument("--execution_date", required=True, help="Execution date (YYYY-MM-DD)")
    args = parser.parse_args()

    predict_model(args.model_type, args.execution_date)
