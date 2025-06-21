import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import os
import json
from joblib import dump

# ---------------- Configuration ----------------
PARTS_DIR = "/opt/airflow/datamart/gold/training_features"
MLFLOW_TRACKING_URI = "http://mlflow_tracking:5000"
MLFLOW_EXPERIMENT_NAME = "loan_default_experiment"
ARTIFACT_BASE = "/opt/airflow/datamart/gold/models"
SCORING_DIR = "/opt/airflow/datamart/gold/scoring"
TRAIN_EVAL_DIR = "/opt/airflow/datamart/gold/train_eval"

TRAINING_SNAPSHOT = "2023_12_01"  # Fixed training date for full model training


def clean_dataframe(df):
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.fillna(0)
    df = df.clip(lower=-1e5, upper=1e5)
    return df

def train_model(model_type: str = "xgboost", execution_date=None):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print(f"[DEBUG] execution_date: {execution_date}")

    if not execution_date:
        raise ValueError("execution_date is required")

    snapshot_str = execution_date.replace("-", "_")
    if snapshot_str != TRAINING_SNAPSHOT:
        print(f"[SKIP] Only train on {TRAINING_SNAPSHOT}. Current snapshot: {snapshot_str}")
        return

    # Load and combine training parts
    print(f"[INFO] Loading training parts from {PARTS_DIR}...")
    parts = [f for f in os.listdir(PARTS_DIR) if f.endswith(".parquet") and "training_features_" in f]
    if not parts:
        print(f"[ERROR] No training parts found in {PARTS_DIR}")
        return

    dataframes = []
    for f in sorted(parts):
        df = pd.read_parquet(os.path.join(PARTS_DIR, f))
        if "label" in df.columns:
            dataframes.append(df)
    full_df = pd.concat(dataframes, ignore_index=True)
    print(f"[INFO] Combined training data shape: {full_df.shape}")

    y = full_df["label"]
    X = full_df.drop(columns=["label", "snapshot_date", "customer_id"], errors="ignore")
    customer_ids = full_df["customer_id"]

    if X.empty or y.empty:
        print("[ERROR] Empty features or labels. Skipping.")
        return

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    tuning_history = []

    def objective_xgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "eval_metric": "logloss",
            "use_label_encoder": False
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        score = f1_score(y_val, model.predict(X_val))
        tuning_history.append({"model": "xgboost", **params, "score": score})
        return score

    def objective_lr(trial):
        params = {
            "C": trial.suggest_float("C", 0.01, 10.0),
            "penalty": trial.suggest_categorical("penalty", ["l2"]),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
            "max_iter": 100
        }
        model = LogisticRegression(**params)
        model.fit(clean_dataframe(X_train), y_train)
        score = f1_score(y_val, model.predict(clean_dataframe(X_val)))
        tuning_history.append({"model": "logistic_regression", **params, "score": score})
        return score

    print("[INFO] Starting Optuna hyperparameter tuning...")
    study = optuna.create_study(direction="maximize")

    model_type = model_type.lower()
    if model_type == "logistic_regression":
        study.optimize(objective_lr, n_trials=10)
        final_model = LogisticRegression(**study.best_params)
        X_train_final = clean_dataframe(X_train)
    elif model_type == "xgboost":
        study.optimize(objective_xgb, n_trials=10)
        final_model = XGBClassifier(**study.best_params, eval_metric="logloss", use_label_encoder=False)
        X_train_final = X_train
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    final_model.fit(X_train_final, y_train)
    y_pred = final_model.predict(clean_dataframe(X_val))
    f1 = f1_score(y_val, y_pred)
    print(f"[INFO] Final F1 Score: {f1:.4f}")

    # Save scoring features
    scoring_df = full_df.loc[X_val.index].drop(columns=["label", "snapshot_date"], errors="ignore")
    os.makedirs(SCORING_DIR, exist_ok=True)
    scoring_df.to_parquet(f"{SCORING_DIR}/scoring_features_{TRAINING_SNAPSHOT}.parquet", index=False)

    # Save predictions
    os.makedirs(TRAIN_EVAL_DIR, exist_ok=True)
    pred_df = pd.DataFrame({
        "prediction": y_pred,
        "customer_id": customer_ids.loc[X_val.index].values
    })
    pred_df.to_parquet(f"{TRAIN_EVAL_DIR}/model_predictions_{model_type}_{TRAINING_SNAPSHOT}.parquet", index=False)

    # Save model and tuning artifacts
    artifact_dir = os.path.join(ARTIFACT_BASE, model_type, TRAINING_SNAPSHOT)
    os.makedirs(artifact_dir, exist_ok=True)
    dump(final_model, os.path.join(artifact_dir, "model.pkl"))
    pd.DataFrame(tuning_history).to_parquet(os.path.join(artifact_dir, f"tuning_results_{model_type}.parquet"), index=False)

    # Save feature schema
    schema_path = os.path.join(artifact_dir, "feature_columns.json")
    with open(schema_path, "w") as f:
        json.dump({"features": X.columns.tolist()}, f, indent=2)

    # Log to MLflow
    with mlflow.start_run(run_name=f"{model_type}_{TRAINING_SNAPSHOT}_run"):
        mlflow.set_tag("execution_date", execution_date)
        mlflow.set_tag("snapshot", TRAINING_SNAPSHOT)
        mlflow.set_tag("model_type", model_type)
        mlflow.log_param("model_name", model_type)
        mlflow.log_params(study.best_params)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(final_model, artifact_path="model")

    print(f"[INFO] Training complete and artifacts saved for {model_type} @ {TRAINING_SNAPSHOT}")

# CLI trigger
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--execution_date", type=str, required=True)
    args = parser.parse_args()

    train_model(model_type=args.model_type, execution_date=args.execution_date)
