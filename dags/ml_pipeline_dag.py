from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
from functools import partial
import os
import sys

# Add scripts folder to path
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'scripts')
sys.path.append(os.path.abspath(SCRIPT_DIR))

# Import pipeline functions
from feature_pipeline import run_feature_pipeline
from train_model import train_model
from predict_model import predict_model
from monitor_model import monitor_model
from model_registry import select_best_model, persist_best_model

# ---------------- Default args ----------------
default_args = {
    'owner': 'ml_team',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# ---------------- Control logic ----------------
def only_on_train_date(**context):
    snapshot_str = context["ds"].replace("-", "_")
    return snapshot_str == "2023_12_01"

def run_on_inference_dates(**context):
    snapshot_str = context["ds"].replace("-", "_")
    return snapshot_str >= "2024_01_01"

def model_exists(model_type: str, snapshot: str) -> bool:
    model_path = f"/opt/airflow/datamart/gold/models/{model_type}/{snapshot}/model.pkl"
    return os.path.exists(model_path)

def allow_inference_if_model_available(model_type: str, **context):
    snapshot_str = context["ds"].replace("-", "_")
    return model_exists(model_type, "2023_12_01")  # Always check if 2023_12_01 model exists


# ---------------- DAG Definition ----------------
with DAG(
    dag_id='loan_default_ml_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline with training, inference, monitoring, and model registry',
    schedule_interval='@monthly',
    start_date=datetime(2022, 12, 1),
    end_date=datetime(2024, 6, 1),
    catchup=True,
    max_active_runs=1,
    tags=['loan_default', 'ml_pipeline']
) as dag:

    # Prepare monthly training data parts (Jan–Dec 2023)
    prepare_training_part = PythonOperator(
        task_id="prepare_training_data",
        python_callable=run_feature_pipeline,
        op_kwargs={
            "execution_date": "{{ ds }}",
            "mode": "train"
        },
    )

    # Gate to allow training only on 2023-12-01
    train_gate = ShortCircuitOperator(
        task_id="train_gate",
        python_callable=only_on_train_date,
    )

    automl_start = EmptyOperator(task_id="automl_start")
    training_completed = EmptyOperator(task_id="training_completed")

    # # Inference gate (from 2024-01-01 onward)
    # infer_gate = ShortCircuitOperator(
    #     task_id="infer_gate",
    #     python_callable=run_on_inference_dates,
    #     trigger_rule=TriggerRule.ALL_DONE,  # Key fix!
    # )

    # Scoring features (only for OOT inference)
    generate_scoring = PythonOperator(
        task_id="generate_scoring_features",
        python_callable=run_feature_pipeline,
        op_kwargs={
            "execution_date": "{{ ds }}",
            "mode": "score"
        },
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # Training, Inference, Monitoring per model
    model_tasks = []
    for model in ['logistic_regression', 'xgboost']:
        # Inference gate with model-existence logic
        infer_gate = ShortCircuitOperator(
            task_id=f"{model}_infer_gate",
            python_callable=partial(allow_inference_if_model_available, model),
            provide_context=True,
        )

        train = PythonOperator(
            task_id=f'{model}_train',
            python_callable=train_model,
            op_kwargs={
                "model_type": model,
                "execution_date": "{{ ds }}"
            }
        )
        infer = PythonOperator(
            task_id=f'{model}_infer',
            python_callable=predict_model,
            op_kwargs={
                "model_type": model,
                "execution_date": "{{ ds }}"
            }
        )
        monitor = PythonOperator(
            task_id=f'{model}_monitor',
            python_callable=monitor_model,
            op_kwargs={
                "model_type": model,
                "execution_date": "{{ ds }}"
            }
        )

        # Training path
        prepare_training_part >> train_gate >> automl_start >> train >> training_completed

        # Inference + Monitoring path (only after training completes and for OOT months)
        generate_scoring >> infer_gate >> infer >> monitor
        model_tasks.append(monitor)

    # Monitoring completion and model registry steps
    monitoring_done = EmptyOperator(
    task_id='monitoring_complete',
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
)

    # Model registry gate (from 2024-01-01 onward)
    registry_gate = ShortCircuitOperator(
        task_id="model_registry_gate",
        python_callable=run_on_inference_dates,
    )

    select = PythonOperator(
        task_id='select_best_model',
        python_callable=select_best_model,
        op_kwargs={"execution_date": "{{ ds }}"},
    )

    persist = PythonOperator(
        task_id='persist_best_model',
        python_callable=persist_best_model,
        op_kwargs={"execution_date": "{{ ds }}"},
    )

    # Only select and persist models after inference + monitoring
    model_tasks >> monitoring_done >> registry_gate >> select >> persist
