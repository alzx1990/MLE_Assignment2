import pandas as pd
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse
from prepare_training_data import prepare_training_data

# ---------------- Configuration ----------------
CUST_FIN_DIR = "/opt/airflow/datamart/gold/feature_store/cust_fin_risk"
ENGAGEMENT_DIR = "/opt/airflow/datamart/gold/feature_store/eng"
LABEL_DIR = "/opt/airflow/datamart/gold/label_store"
TRAINING_OUTPUT = "/opt/airflow/datamart/gold/training_features"
SCORING_OUTPUT = "/opt/airflow/datamart/gold/scoring"

CUST_FIN_PREFIX = "gold_ft_store_cust_fin_risk_"
ENGAGEMENT_PREFIX = "gold_ft_store_engagement_"
LABEL_PREFIX = "gold_label_store_"

def load_and_merge_features(snapshot_str: str):
    cust_path = os.path.join(CUST_FIN_DIR, f"{CUST_FIN_PREFIX}{snapshot_str}.parquet")
    eng_path = os.path.join(ENGAGEMENT_DIR, f"{ENGAGEMENT_PREFIX}{snapshot_str}.parquet")

    if not os.path.exists(cust_path):
        print(f"[ERROR] Missing customer_fin file: {cust_path}")
        return None

    cust_df = pd.read_parquet(cust_path)
    cust_df.rename(columns={"Customer_ID": "customer_id"}, inplace=True)
    cust_df.columns = cust_df.columns.str.strip()
    cust_df["customer_id"] = cust_df["customer_id"].astype(str).str.strip()
    cust_df["snapshot_date"] = pd.to_datetime(cust_df["snapshot_date"]).dt.date
    cust_df.fillna(0, inplace=True)

    if os.path.exists(eng_path):
        eng_df = pd.read_parquet(eng_path)
        eng_df.rename(columns={"Customer_ID": "customer_id"}, inplace=True)
        eng_df.columns = eng_df.columns.str.strip()
        eng_df["customer_id"] = eng_df["customer_id"].astype(str).str.strip()
        eng_df["snapshot_date"] = pd.to_datetime(eng_df["snapshot_date"]).dt.date
        cust_df = cust_df.merge(eng_df, on=["customer_id", "snapshot_date"], how="left")
        cust_df.fillna(0, inplace=True)
    else:
        print(f"[INFO] No engagement data found for {snapshot_str}. Proceeding with customer features only.")

    return cust_df

def generate_training_features(execution_date: str):
    print("[INFO] Generating training features...")
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 1)
    dfs = []

    while start_date <= end_date:
        feature_str = start_date.strftime("%Y_%m_%d")
        label_date = start_date + relativedelta(months=6)
        label_str = label_date.strftime("%Y_%m_%d")

        features_df = load_and_merge_features(feature_str)
        if features_df is None:
            start_date += relativedelta(months=1)
            continue

        label_path = os.path.join(LABEL_DIR, f"{LABEL_PREFIX}{label_str}.parquet")
        if not os.path.exists(label_path):
            print(f"[SKIP] Missing label file: {label_path}")
            start_date += relativedelta(months=1)
            continue

        label_df = pd.read_parquet(label_path)
        label_df.rename(columns={"Customer_ID": "customer_id"}, inplace=True)
        label_df.columns = label_df.columns.str.strip()
        label_df["customer_id"] = label_df["customer_id"].astype(str).str.strip()
        label_df["snapshot_date"] = pd.to_datetime(label_df["snapshot_date"]).dt.date
        label_df = label_df[label_df["snapshot_date"] == label_date.date()]

        merged = features_df.merge(label_df[["customer_id", "label"]], on="customer_id", how="inner")
        if not merged.empty:
            dfs.append(merged)
        else:
            print(f"[WARNING] No label match for {feature_str}")

        start_date += relativedelta(months=1)

    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        os.makedirs(TRAINING_OUTPUT, exist_ok=True)
        out_path = os.path.join(TRAINING_OUTPUT, "training_features_2023_12_01.parquet")
        final_df.to_parquet(out_path, index=False)
        print(f"[INFO] Saved training features: {out_path} ({final_df.shape})")
    else:
        print("[WARNING] No training data compiled.")

def generate_scoring_features(execution_date: str):
    print(f"[INFO] Generating scoring features for {execution_date}")
    snapshot_str = execution_date.replace("-", "_")
    features_df = load_and_merge_features(snapshot_str)

    if features_df is not None:
        os.makedirs(SCORING_OUTPUT, exist_ok=True)
        out_path = os.path.join(SCORING_OUTPUT, f"scoring_features_{snapshot_str}.parquet")
        features_df.to_parquet(out_path, index=False)
        print(f"[INFO] Scoring features saved: {out_path}")
    else:
        print("[ERROR] Feature generation failed.")

# Add this Airflow-compatible wrapper function:
def run_feature_pipeline(mode: str, execution_date: str):
    if mode == "train":
        prepare_training_data(execution_date)
    elif mode == "score":
        generate_scoring_features(execution_date)
    else:
        raise ValueError(f"Unknown mode: {mode}")

# CLI fallback
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "score"], required=True, help="Mode: train or score")
    parser.add_argument("--execution_date", required=True, help="Execution date in YYYY-MM-DD format")
    args = parser.parse_args()

    run_feature_pipeline(args.mode, args.execution_date)
