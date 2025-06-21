import pandas as pd
import os
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Directory paths
CUST_FIN_DIR = "/opt/airflow/datamart/gold/feature_store/cust_fin_risk"
ENGAGEMENT_DIR = "/opt/airflow/datamart/gold/feature_store/eng"
LABEL_DIR = "/opt/airflow/datamart/gold/label_store"
OUTPUT_DIR = "/opt/airflow/datamart/gold/training_features"
SCHEMA_DIR = "/opt/airflow/models/schema"

# Filename patterns
CUST_FIN_PREFIX = "gold_ft_store_cust_fin_risk_"
ENGAGEMENT_PREFIX = "gold_ft_store_engagement_"
LABEL_PREFIX = "gold_label_store_"

def prepare_training_data(execution_date: str = None):
    if not execution_date:
        raise ValueError("execution_date (YYYY-MM-DD) is required")

    snapshot_date = datetime.strptime(execution_date, "%Y-%m-%d")
    snapshot_str = snapshot_date.strftime("%Y_%m_%d")
    label_date = snapshot_date + relativedelta(months=6)
    label_str = label_date.strftime("%Y_%m_%d")

    cust_path = os.path.join(CUST_FIN_DIR, f"{CUST_FIN_PREFIX}{snapshot_str}.parquet")
    eng_path = os.path.join(ENGAGEMENT_DIR, f"{ENGAGEMENT_PREFIX}{snapshot_str}.parquet")
    label_path = os.path.join(LABEL_DIR, f"{LABEL_PREFIX}{label_str}.parquet")

    if not os.path.exists(cust_path) or not os.path.exists(label_path):
        print(f"[SKIP] Missing customer or label data for {snapshot_str}")
        return

    # Load financial features
    cust_df = pd.read_parquet(cust_path)
    cust_df.rename(columns={"Customer_ID": "customer_id"}, inplace=True)
    cust_df.columns = cust_df.columns.str.strip()
    cust_df["customer_id"] = cust_df["customer_id"].astype(str).str.strip()
    cust_df["snapshot_date"] = pd.to_datetime(cust_df["snapshot_date"]).dt.date
    cust_df.fillna(0, inplace=True)

    # Merge with engagement features
    if os.path.exists(eng_path):
        eng_df = pd.read_parquet(eng_path)
        eng_df.rename(columns={"Customer_ID": "customer_id"}, inplace=True)
        eng_df.columns = eng_df.columns.str.strip()
        eng_df["customer_id"] = eng_df["customer_id"].astype(str).str.strip()
        eng_df["snapshot_date"] = pd.to_datetime(eng_df["snapshot_date"]).dt.date
        cust_df = cust_df.merge(eng_df, on=["customer_id", "snapshot_date"], how="left")
        cust_df.fillna(0, inplace=True)
    else:
        print(f"[INFO] No engagement data for {snapshot_str}")

    # Load and filter labels
    label_df = pd.read_parquet(label_path)
    label_df.rename(columns={"Customer_ID": "customer_id"}, inplace=True)
    label_df.columns = label_df.columns.str.strip()
    label_df["customer_id"] = label_df["customer_id"].astype(str).str.strip()
    label_df["snapshot_date"] = pd.to_datetime(label_df["snapshot_date"]).dt.date
    label_df = label_df[label_df["snapshot_date"] == label_date.date()]

    # Merge features and label
    merged = cust_df.merge(label_df[["customer_id", "label"]], on="customer_id", how="inner")
    if merged.empty:
        print(f"[WARNING] No label matches found for {snapshot_str}")
        return

    # Save monthly training part
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"training_features_{snapshot_str}.parquet")
    merged.to_parquet(output_path, index=False)
    print(f"[INFO] Saved training part: {output_path} ({merged.shape})")

    # Optionally update feature schema (only for latest month)
    if snapshot_str == "2023_12_01":
        schema = {"features": [col for col in merged.columns if col not in {"label", "customer_id", "snapshot_date"}]}
        os.makedirs(SCHEMA_DIR, exist_ok=True)
        schema_path = os.path.join(SCHEMA_DIR, "feature_columns.json")
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2)
        print(f"[INFO] Saved feature schema to {schema_path}")

