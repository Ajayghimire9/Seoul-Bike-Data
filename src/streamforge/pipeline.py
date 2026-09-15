from pathlib import Path

from .ingest import load_source, validate_source
from .transform import build_fact_table


def run(source: str = "Dataset/SeoulBikeData.csv", output: str = "artifacts/bike_demand.parquet") -> str:
    frame = validate_source(load_source(source))
    fact = build_fact_table(frame)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fact.to_parquet(output_path, index=False)
    return str(output_path)


if __name__ == "__main__":
    print(run())
