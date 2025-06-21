import os
import pandas as pd
import numpy as np
import joblib
import shap
import mlflow
import json
import tempfile
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import ks_2samp
import argparse

# ---------------- Configuration ----------------
BASE_DIR = "/opt/airflow/datamart/gold"
MODEL_DIR = os.path.join(BASE_DIR, "models")
SCORING_DIR = os.path.join(BASE_DIR, "scoring")
TRAINING_FEATURE_DIR = os.path.join(BASE_DIR, "training_features")
LABEL_DIR = os.path.join(BASE_DIR, "label_store")
MONITORING_LOG_PATH = os.path.join(BASE_DIR, "monitoring_log.parquet")
ALERT_LOG_PATH = os.path.join(BASE_DIR, "monitoring_alerts.parquet")

MLFLOW_TRACKING_URI = "http://mlflow_tracking:5000"
MLFLOW_EXPERIMENT_NAME = "loan_default_experiment"

# ---------------- Main Monitoring Function ----------------
def monitor_model(model_type: str, execution_date: str):
    snapshot_str = execution_date.replace("-", "_")
    TRAINING_SNAPSHOT = "2023_12_01"

    model_path = os.path.join(MODEL_DIR, model_type, TRAINING_SNAPSHOT, "model.pkl")
    schema_path = os.path.join(MODEL_DIR, model_type, TRAINING_SNAPSHOT, "feature_columns.json")

    predictions_path = os.path.join(SCORING_DIR, f"model_predictions_{model_type}_{snapshot_str}.parquet")
    features_path = os.path.join(SCORING_DIR, f"scoring_features_{snapshot_str}.parquet")
    training_features_path = os.path.join(TRAINING_FEATURE_DIR, "training_features_2023_12_01.parquet")

    label_date = (datetime.strptime(execution_date, "%Y-%m-%d") + relativedelta(months=6)).strftime("%Y_%m_%d")
    label_path = os.path.join(LABEL_DIR, f"gold_label_store_{label_date}.parquet")

    # Validate required files
    for path in [model_path, schema_path, predictions_path, features_path, label_path]:
        if not os.path.exists(path):
            print(f"[ERROR] Missing required file: {path}")
            return

    # Load data
    print("[INFO] Loading model, schema, features, predictions, and labels...")
    model = joblib.load(model_path)
    with open(schema_path, "r") as f:
        schema = json.load(f)
    expected_cols = schema["features"]

    features_df = pd.read_parquet(features_path)
    features_df.columns = features_df.columns.str.strip()
    features_df = features_df.reindex(columns=expected_cols)
    features_df = features_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    preds_df = pd.read_parquet(predictions_path)
    preds_df.columns = preds_df.columns.str.lower().str.strip()

    labels_df = pd.read_parquet(label_path)
    labels_df.columns = labels_df.columns.str.lower().str.strip()

    if "customer_id" not in preds_df.columns or "customer_id" not in labels_df.columns:
        raise ValueError("Missing customer_id column in predictions or labels.")

    merged = preds_df.merge(labels_df[["customer_id", "label"]], on="customer_id", how="inner")
    if merged.empty:
        raise ValueError("No overlapping customer_id between predictions and labels.")

    y_pred, y_true = merged["prediction"], merged["label"]

    # Compute metrics
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "execution_date": execution_date,
        "model_type": model_type,
        "f1_score": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "support": len(y_true)
    }

    # SHAP + Drift
    if not features_df.empty:
        try:
            print("[INFO] Computing SHAP values...")
            sample = shap.utils.sample(features_df, min(100, len(features_df)), random_state=42)
            explainer = shap.Explainer(model.predict, sample)
            shap_values = explainer(features_df)
            avg_shap = np.abs(shap_values.values).mean(axis=0)
            for i, col in enumerate(expected_cols):
                metrics[f"shap_{col}"] = avg_shap[i]
        except Exception as e:
            print(f"[WARNING] SHAP failed: {e}")

        if os.path.exists(training_features_path):
            print("[INFO] Running KS drift test...")
            train_df = pd.read_parquet(training_features_path).reindex(columns=expected_cols)
            train_df = train_df.apply(pd.to_numeric, errors="coerce").fillna(0)

            for col in expected_cols:
                try:
                    ks_stat, _ = ks_2samp(train_df[col], features_df[col])
                    metrics[f"ks_{col}"] = ks_stat
                except Exception as e:
                    print(f"[WARNING] KS test failed for {col}: {e}")
        else:
            print("[WARNING] Training features not available for drift detection.")
    else:
        print("[WARNING] Features dataframe is empty after schema enforcement.")

    print("[METRICS]", {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

    # ---------------- MLflow Logging ----------------
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"{model_type}_{snapshot_str}_monitoring"):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
                else:
                    mlflow.set_tag(key, value)

            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_dir = os.path.join(tmpdir, "monitoring_artifacts")
                os.makedirs(artifact_dir, exist_ok=True)

                json_path = os.path.join(artifact_dir, "monitoring_metrics.json")
                with open(json_path, "w") as f:
                    json.dump(metrics, f, indent=2)
                mlflow.log_artifact(json_path)

                if "shap_" in "".join(metrics.keys()):
                    try:
                        shap.summary_plot(shap_values, features_df, show=False)
                        shap_path = os.path.join(artifact_dir, "shap_summary.png")
                        plt.savefig(shap_path, bbox_inches="tight")
                        plt.close()
                        mlflow.log_artifact(shap_path)
                    except Exception as e:
                        print(f"[WARNING] SHAP plot failed: {e}")

                drift_cols = {k: v for k, v in metrics.items() if k.startswith("ks_")}
                if drift_cols:
                    drift_df = pd.DataFrame([drift_cols])
                    drift_path = os.path.join(artifact_dir, "drift_stats.csv")
                    drift_df.to_csv(drift_path, index=False)
                    mlflow.log_artifact(drift_path)
    except Exception as e:
        print(f"[WARNING] MLflow logging failed: {e}")

    # ---------------- Save Monitoring Log ----------------
    df = pd.DataFrame([metrics])
    if os.path.exists(MONITORING_LOG_PATH):
        try:
            existing = pd.read_parquet(MONITORING_LOG_PATH)
            df = pd.concat([existing, df], ignore_index=True)
        except Exception as e:
            print(f"[WARNING] Failed to read existing monitoring log: {e}")

    os.makedirs(os.path.dirname(MONITORING_LOG_PATH), exist_ok=True)
    df.to_parquet(MONITORING_LOG_PATH, index=False)
    print(f"[INFO] Monitoring log updated at: {MONITORING_LOG_PATH}")

    # ---------------- Store Alerts ----------------
    drift_alerts = [
        col.replace("ks_", "") for col, val in metrics.items()
        if col.startswith("ks_") and isinstance(val, float) and val > 0.2
    ]
    alert_record = {
        "execution_date": execution_date,
        "model_type": model_type,
        "f1_score": metrics["f1_score"],
        "drift_alerts": drift_alerts,
        "timestamp": datetime.utcnow().isoformat()
    }
    alert_df = pd.DataFrame([alert_record])

    if os.path.exists(ALERT_LOG_PATH):
        try:
            existing_alerts = pd.read_parquet(ALERT_LOG_PATH)
            alert_df = pd.concat([existing_alerts, alert_df], ignore_index=True)
        except Exception as e:
            print(f"[WARNING] Failed to read existing alert log: {e}")

    os.makedirs(os.path.dirname(ALERT_LOG_PATH), exist_ok=True)
    alert_df.to_parquet(ALERT_LOG_PATH, index=False)
    print(f"[INFO] Alert log updated at: {ALERT_LOG_PATH}")


# ---------------- CLI Interface ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, help="Model type (e.g. xgboost)")
    parser.add_argument("--execution_date", required=True, help="Execution date (YYYY-MM-DD)")
    args = parser.parse_args()

    monitor_model(args.model_type, args.execution_date)
