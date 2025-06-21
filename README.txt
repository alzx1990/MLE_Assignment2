https://github.com/alzx1990/MLE_Assignment2

------------------------------------------------------------------------------------------------
# README.md

## 📊 Modular ML Pipeline for Loan Default Prediction

This project implements an end-to-end machine learning pipeline for loan default prediction using Apache Airflow, MLflow, and Streamlit. The pipeline orchestrates data preparation, model training, evaluation, persistence, inference, monitoring, and dashboard visualization.

---

## 🧱 Folder Structure
```
├── dags/                     # Airflow DAG definitions
├── scripts/                  # Modular Python pipeline scripts
├── mlruns/                  # MLflow tracking data
├── datamart/gold/           # Gold tables for features, predictions, and monitoring
├── models/                  # Persisted models and best model info
├── Dockerfile               # Container setup
├── docker-compose.yml       # Multi-service orchestration
├── requirements.txt         # Python dependencies
```

---

## 🔧 Setup Instructions
1. **Build & Start Airflow + MLflow**
```bash
docker-compose up --build
```

2. **Access Airflow UI**: [http://localhost:8080](http://localhost:8080)
   - Username: `admin`
   - Password: `admin`

3. **Trigger the ML Pipeline**
- Use the DAG: `loan_default_ml_pipeline`
- Set `catchup=True` to backfill historical runs

4. **Access MLflow UI**: [http://localhost:5000](http://localhost:5000)
   - View training metrics, parameters, and artifacts

5. **Launch Monitoring Dashboard**
```bash
streamlit run scripts/visualize_monitoring.py
```
- Then visit: [http://localhost:8501](http://localhost:8501)

---

## 🔄 Pipeline Overview
1. `prepare_training_data.py` — merges and prepares features from gold tables
2. `train_model.py` — trains and logs models using MLflow (XGBoost, MLP)
3. `select_best_model.py` — selects best model based on F1 score
4. `persist_model.py` — saves the best model locally
5. `predict_model.py` — runs batch inference on scoring dataset
6. `store_predictions.py` — stores predictions in the datamart
7. `monitor_model.py` — computes monitoring metrics using true labels
8. `store_monitoring.py` — appends metrics to a monitoring log
9. `visualize_monitoring.py` — Streamlit dashboard to view trends over time

---

## 📝 Notes
- All model metadata and artifacts are logged via MLflow
- Streamlit provides live visualization of monitoring logs
- Airflow orchestrates the full pipeline with monthly scheduling

---

## 📩 Contact
For any queries, please contact the ML engineering team.