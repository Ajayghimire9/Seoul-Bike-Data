"""Transactional, idempotent warehouse loading keyed by observation timestamp."""

import argparse
import os

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
)


def load_to_postgres(frame: pd.DataFrame, dsn: str) -> int:
    from .quality import quality_report

    if not quality_report(frame)["passed"]:
        raise ValueError("Warehouse input failed the data quality contract")
    engine = create_engine(dsn)
    schema = "analytics" if engine.dialect.name == "postgresql" else None
    metadata = MetaData(schema=schema)
    columns = []
    for name in frame.columns:
        if name == "timestamp":
            dtype = DateTime()
        elif name == "is_weekend":
            dtype = Boolean()
        elif pd.api.types.is_integer_dtype(frame[name]):
            dtype = Integer()
        elif pd.api.types.is_numeric_dtype(frame[name]):
            dtype = Float()
        else:
            dtype = String()
        columns.append(Column(name, dtype, primary_key=(name == "timestamp")))
    table = Table("bike_demand", metadata, *columns)
    clean = frame.copy()
    clean["is_weekend"] = clean["is_weekend"].astype(bool)
    records = clean.astype(object).where(clean.notna(), None).to_dict("records")
    try:
        with engine.begin() as conn:
            if schema:
                conn.exec_driver_sql("CREATE SCHEMA IF NOT EXISTS analytics")
                from sqlalchemy.dialects.postgresql import insert
            elif engine.dialect.name == "sqlite":
                from sqlalchemy.dialects.sqlite import insert
            else:
                raise ValueError("Only PostgreSQL and SQLite are supported")
            metadata.create_all(conn)
            for offset in range(0, len(records), 500):
                stmt = insert(table).values(records[offset : offset + 500])
                stmt = stmt.on_conflict_do_update(
                    index_elements=["timestamp"],
                    set_={
                        c.name: stmt.excluded[c.name]
                        for c in table.columns
                        if c.name != "timestamp"
                    },
                )
                conn.execute(stmt)
        return len(records)
    finally:
        engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/bike_demand.parquet")
    parser.add_argument("--dsn", default=os.getenv("POSTGRES_DSN"))
    args = parser.parse_args()
    if not args.dsn:
        parser.error("Set POSTGRES_DSN or pass --dsn")
    print(load_to_postgres(pd.read_parquet(args.input), args.dsn))
