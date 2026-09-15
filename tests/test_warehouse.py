import sqlite3

from src.streamforge.ingest import load_source
from src.streamforge.load import load_to_postgres
from src.streamforge.transform import build_fact_table


def test_rerun_updates_instead_of_duplicating(tmp_path):
    frame = build_fact_table(load_source("Dataset/SeoulBikeData.csv").iloc[:5])
    path = tmp_path / "warehouse.db"
    dsn = "sqlite:///" + str(path)
    load_to_postgres(frame, dsn)
    frame.loc[frame.index[0], "rented_bike_count"] = 999
    load_to_postgres(frame, dsn)
    with sqlite3.connect(path) as conn:
        assert conn.execute("select count(*) from bike_demand").fetchone()[0] == 5
        assert conn.execute("select max(rented_bike_count) from bike_demand").fetchone()[0] == 999
