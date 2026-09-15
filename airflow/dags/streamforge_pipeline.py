from datetime import UTC, datetime

from airflow.operators.bash import BashOperator

from airflow import DAG

with DAG(
    dag_id="streamforge_bike_pipeline",
    start_date=datetime(2025, 1, 1, tzinfo=UTC),
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

    load_warehouse = BashOperator(
        task_id="load_warehouse",
        bash_command="python -m src.streamforge.load",
    )
    ingest_and_validate >> load_warehouse >> analytics
    ingest_and_validate >> transform
