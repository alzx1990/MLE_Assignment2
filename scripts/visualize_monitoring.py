import streamlit as st
import pandas as pd
import os
import altair as alt
import plotly.express as px
import mlflow
import tempfile
import socket
import glob
from datetime import datetime

# ------------------ Configuration ------------------
MONITORING_LOG = "/app/datamart/gold/monitoring_log.parquet"
ALERT_LOG = "/app/datamart/gold/monitoring_alerts.parquet"
MLFLOW_TRACKING_URI = "http://mlflow_tracking:5000"

st.set_page_config(page_title="ML Monitoring Dashboard", layout="wide")
st.title("📊 ML Model Monitoring Dashboard")

# ------------------ Helper Functions ------------------
def is_mlflow_available(uri: str) -> bool:
    try:
        host, port = uri.replace("http://", "").split(":")
        socket.create_connection((host, int(port)), timeout=2)
        return True
    except Exception:
        return False

# ------------------ Monitoring View ------------------
st.header("Model Performance Over Time")

if not os.path.exists(MONITORING_LOG):
    st.warning("Monitoring log not found. Please ensure `monitor_model.py` has been run.")
else:
    df = pd.read_parquet(MONITORING_LOG)
    if df.empty:
        st.warning("Monitoring log is empty.")
    else:
        df["execution_date"] = pd.to_datetime(df["execution_date"], errors="coerce")
        df = df.dropna(subset=["execution_date"])
        df = df.sort_values("execution_date")

        aggregate_monthly = st.checkbox("Aggregate metrics by month", value=False)
        metric_cols = ["f1_score", "accuracy"]
        col1, col2 = st.columns(2)

        for metric, col in zip(metric_cols, [col1, col2]):
            with col:
                st.subheader(f"{metric.replace('_', ' ').title()} Over Time")
                if aggregate_monthly:
                    df_monthly = (
                        df.groupby([df["execution_date"].dt.to_period("M"), "model_type"])
                        .agg({metric: "mean"})
                        .reset_index()
                    )
                    df_monthly["execution_month"] = df_monthly["execution_date"].dt.to_timestamp()
                    x_col = "execution_month:T"
                    tooltip_cols = ["execution_month", metric, "model_type"]
                    plot_df = df_monthly
                else:
                    x_col = "execution_date:T"
                    tooltip_cols = ["execution_date", metric, "model_type"]
                    plot_df = df

                chart = alt.Chart(plot_df).mark_line(point=True).encode(
                    # x=alt.X(x_col, title="Snapshot Date", axis=alt.Axis(format="%b %Y", labelAngle=90)),
                    x=alt.X(
                        "execution_date:T",
                        title="Snapshot Date",
                        axis=alt.Axis(format="%b %Y", labelAngle=0, labelOverlap=True, tickCount="month")
                    ),
                    y=alt.Y(f"{metric}:Q", title=metric.replace("_", " ").title()),
                    color="model_type:N",
                    tooltip=tooltip_cols
                )
                st.altair_chart(chart.properties(height=300), use_container_width=True)

        # ------------------ Explainability ------------------
        st.subheader("Explainability: SHAP Feature Importance")
        model_options = df["model_type"].dropna().unique().tolist()
        selected_model = st.selectbox("Select model for explanation", model_options)

        model_df = df[df["model_type"] == selected_model].sort_values("execution_date")
        shap_cols = [col for col in df.columns if col.startswith("shap_")]

        if shap_cols and not model_df[shap_cols].dropna(how="all").empty:
            latest = model_df.dropna(subset=shap_cols).iloc[-1]
            shap_df = pd.DataFrame({
                "Feature": [col.replace("shap_", "") for col in shap_cols],
                "Importance": [latest[col] for col in shap_cols]
            }).sort_values("Importance", ascending=False)

            if not shap_df.empty:
                st.bar_chart(shap_df.set_index("Feature"))
            else:
                st.info("No SHAP data found for this model.")
        else:
            st.info("SHAP values unavailable for selected model.")

        # ------------------ Drift Detection ------------------
        st.subheader("Feature Drift: KS Statistic")
        ks_cols = [col for col in df.columns if col.startswith("ks_")]

        if ks_cols and not model_df[ks_cols].dropna(how="all").empty:
            latest = model_df.dropna(subset=ks_cols).iloc[-1]
            ks_df = pd.DataFrame({
                "Feature": [col.replace("ks_", "") for col in ks_cols],
                "KS Statistic": [latest[col] for col in ks_cols]
            }).sort_values("KS Statistic", ascending=False)

            if not ks_df.empty:
                fig = px.bar(ks_df, x="Feature", y="KS Statistic", title="Feature Drift (KS Test)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid KS drift statistics available.")
        else:
            st.info("KS drift values unavailable for selected model.")

        # ------------------ Raw Monitoring Snapshot ------------------
        st.subheader("Recent Monitoring Logs")
        st.dataframe(df.tail(10).sort_values("timestamp", ascending=False))

# ------------------ Alert Summary View ------------------
if os.path.exists(ALERT_LOG):
    st.header("⚠️ Drift & Performance Alerts")
    alert_df = pd.read_parquet(ALERT_LOG).sort_values("timestamp", ascending=False)
    alert_df["Drifted Features"] = alert_df["drift_alerts"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
    st.dataframe(alert_df[["execution_date", "model_type", "f1_score", "Drifted Features", "timestamp"]])
else:
    st.info("Alert log not found. No alerts to show.")

# ------------------ Hyperparameter Tuning ------------------
st.header("📌 Hyperparameter Tuning Insights")

MODEL_DIR = "/opt/airflow/datamart/gold/models"
TRAINING_SNAPSHOT = "2023_12_01"
model_types = ["logistic_regression", "xgboost"]

options = []
for model_type in model_types:
    tuning_path = os.path.join(
        MODEL_DIR,
        model_type,
        TRAINING_SNAPSHOT,
        f"tuning_results_{model_type}.parquet"
    )
    if os.path.exists(tuning_path):
        label = f"{model_type} - {TRAINING_SNAPSHOT}"
        options.append((label, tuning_path))

if not options:
    st.info("No tuning result files found.")
else:
    selected_label, selected_path = st.selectbox("Select tuning result snapshot", options)
    tuning_df = pd.read_parquet(selected_path)

    if "score" in tuning_df.columns:
        model_type, snapshot_str = selected_label.split(" - ")
        fig = px.histogram(tuning_df, x="score", nbins=10,
                           title=f"{model_type.upper()} F1 Score Distribution ({snapshot_str})")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tuning_df.sort_values("score", ascending=False).head(5))

        if is_mlflow_available(MLFLOW_TRACKING_URI) and st.button("Log to MLflow"):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                fig.write_image(tmp.name)
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                with mlflow.start_run(run_name=f"tuning_viz_{model_type}_{snapshot_str}", nested=True):
                    mlflow.log_artifact(tmp.name, artifact_path="tuning_viz")
            st.success("Logged to MLflow.")
    else:
        st.warning("Missing 'score' column in tuning results.")

