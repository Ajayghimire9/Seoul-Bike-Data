import pandas as pd
from sqlalchemy import create_engine, text


def load_to_postgres(frame: pd.DataFrame, dsn: str) -> int:
    engine = create_engine(dsn)
    with engine.begin() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS analytics"))
    frame.to_sql("bike_demand", engine, schema="analytics", if_exists="append", index=False, method="multi")
    return len(frame)
