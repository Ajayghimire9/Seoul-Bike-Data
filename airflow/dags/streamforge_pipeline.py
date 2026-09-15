from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="streamforge_bike_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["data-engineering", "bike-demand"],
) as dag:
    ingest_and_validate = BashOperator(
        task_id="ingest_and_validate",
        bash_command="python -m src.streamforge.pipeline",
    )

    transform = BashOperator(
        task_id="spark_transform",
        bash_command="spark-submit spark/jobs/transform_bike_demand.py",
    )

    analytics = BashOperator(
        task_id="dbt_build",
        bash_command="cd dbt && dbt build --profiles-dir .",
    )

    ingest_and_validate >> transform >> analytics
