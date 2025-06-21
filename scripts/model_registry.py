import os
import json
import shutil
import mlflow
import pandas as pd
from datetime import datetime
from mlflow.tracking import MlflowClient
import argparse

# ---------------- Configuration ----------------
MONITORING_LOG_PATH = "/opt/airflow/datamart/gold/monitoring_log.parquet"
BEST_MODEL_METADATA = "/opt/airflow/datamart/gold/models/best_model_metadata.json"
PERSIST_DIR = "/opt/airflow/models/persisted_model"
METADATA_FILE = os.path.join(PERSIST_DIR, "model_metadata.json")
MODEL_REGISTRY_NAME = "LoanDefaultModel"

# ---------------- Select Best Model ----------------
def select_best_model(execution_date: str):
    if not os.path.exists(MONITORING_LOG_PATH):
        print(f"[ERROR] Monitoring log not found at {MONITORING_LOG_PATH}")
        return

    try:
        log_df = pd.read_parquet(MONITORING_LOG_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load monitoring log: {e}")
        return

    filtered = log_df[log_df["execution_date"] == execution_date]
    if filtered.empty:
        print(f"[WARNING] No monitoring data found for execution_date: {execution_date}")
        return

    required_cols = {"f1_score", "model_type"}
    if not required_cols.issubset(filtered.columns):
        print(f"[ERROR] Required columns missing: {required_cols}")
        return

    best_row = filtered.sort_values("f1_score", ascending=False).iloc[0]
    best_model = {
        "model_type": best_row["model_type"],
        "f1_score": best_row["f1_score"],
        "execution_date": execution_date,
        "selected_at": datetime.utcnow().isoformat()
    }

    print(f"[INFO] Best model selected: {best_model['model_type']} (F1={best_model['f1_score']:.4f})")

    os.makedirs(os.path.dirname(BEST_MODEL_METADATA), exist_ok=True)
    with open(BEST_MODEL_METADATA, "w") as f:
        json.dump(best_model, f, indent=2)
    print(f"[INFO] Best model metadata saved to: {BEST_MODEL_METADATA}")


# ---------------- Persist Best Model ----------------
def persist_best_model(execution_date: str):
    if not os.path.exists(BEST_MODEL_METADATA):
        print("[ERROR] Best model metadata not found. Run select_best_model first.")
        return

    with open(BEST_MODEL_METADATA, "r") as f:
        best_model = json.load(f)

    model_type = best_model.get("model_type")
    f1_score = best_model.get("f1_score")
    selected_date = best_model.get("execution_date")

    if execution_date != selected_date:
        print(f"[WARNING] Provided execution_date ({execution_date}) != best model date ({selected_date}). Using selected_date.")

    client = MlflowClient()
    experiment = client.get_experiment_by_name("loan_default_experiment")
    if experiment is None:
        print("[ERROR] MLflow experiment not found.")
        return

    # Search for the best run for the given model_type and execution_date
    print("[INFO] Searching for best MLflow run...")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                              filter_string=f"tags.model_type = '{model_type}' and tags.execution_date = '{selected_date}'",
                              order_by=["metrics.f1_score DESC"],
                              max_results=1)

    if not runs:
        print("[ERROR] No matching run found in MLflow for selected model.")
        return

    run = runs[0]
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    print(f"[INFO] Loading model from MLflow run: {run_id}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    print(f"[INFO] Persisting model to {PERSIST_DIR}")
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    mlflow.pyfunc.save_model(model, path=PERSIST_DIR)

    print(f"[INFO] Registering model as {MODEL_REGISTRY_NAME}")
    try:
        result = mlflow.register_model(model_uri, name=MODEL_REGISTRY_NAME)
        version = result.version
    except Exception as e:
        print(f"[ERROR] Failed to register model: {e}")
        version = None

    metadata = {
        "run_id": run_id,
        "model_type": model_type,
        "f1_score": f1_score,
        "execution_date": selected_date,
        "registry_name": MODEL_REGISTRY_NAME,
        "registry_version": version,
        "persisted_at": datetime.utcnow().isoformat()
    }

    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Model and metadata saved to {PERSIST_DIR}")


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution_date", required=True, help="Execution date (YYYY-MM-DD)")
    parser.add_argument("--action", required=True, choices=["select", "persist"], help="Action to perform")

    args = parser.parse_args()

    if args.action == "select":
        select_best_model(args.execution_date)
    elif args.action == "persist":
        persist_best_model(args.execution_date)
