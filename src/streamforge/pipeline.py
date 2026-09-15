import json
from pathlib import Path

from .ingest import load_source, validate_source
from .quality import quality_report
from .transform import build_fact_table


def run(
    source: str = "Dataset/SeoulBikeData.csv", output: str = "artifacts/bike_demand.parquet"
) -> str:
    frame = validate_source(load_source(source))
    fact = build_fact_table(frame)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = quality_report(fact)
    output_path.with_suffix(".quality.json").write_text(json.dumps(report, indent=2))
    if not report["passed"]:
        raise ValueError("Curated data failed quality checks; see quality report")
    temporary = output_path.with_suffix(".tmp.parquet")
    fact.to_parquet(temporary, index=False)
    temporary.replace(output_path)
    return str(output_path)


if __name__ == "__main__":
    print(run())
