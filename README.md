# StreamForge

Urban mobility ingestion and analytics.

StreamForge prepares hourly Seoul bike-demand observations for analytics. It provides a validated local batch pipeline and a warehouse path that can be rerun without accumulating duplicate records.

## Run locally

Use Python 3.11 or newer in a virtual environment.

```bash
pip install -e ".[dev]"
python -m src.streamforge.pipeline
# Optional warehouse load after starting PostgreSQL and setting POSTGRES_DSN:
python -m src.streamforge.load
```

## Design decisions

CSV ingestion produces a typed fact table and a JSON quality report. Duplicate timestamps and invalid humidity or demand values stop publication.

The warehouse loader performs transactional upserts keyed by timestamp. SQLite tests exercise the retry behavior without requiring a database server.

The Airflow path now explicitly loads the warehouse before dbt runs. dbt references the analytics source through a declared source contract.

A Kafka producer and separate Spark transformation job remain available for streaming and distributed-processing experiments.

## Technology

Python, pandas, Parquet, SQLAlchemy, PostgreSQL, Kafka, Spark, Airflow, dbt, Docker Compose.

## Validation

Run `python -m pytest tests -q` from the repository root. CI runs the maintained test suite and lint checks. Tests use local fixtures or mocks and do not deploy cloud resources.

## Scope and limitations

The default command writes local Parquet only. PostgreSQL, Kafka, Airflow and dbt require their own running services and connection configuration. The Spark output is a separate analytical view, not the input to the PostgreSQL loader. Event delivery guarantees must be validated against a broker.
