import pandas as pd


def quality_report(frame: pd.DataFrame) -> dict:
    checks = {
        "row_count_positive": len(frame) > 0,
        "timestamp_not_null": frame["timestamp"].notna().all(),
        "demand_non_negative": (frame["rented_bike_count"] >= 0).all(),
        "hour_range_valid": frame["hour"].between(0, 23).all(),
        "humidity_range_valid": frame["humidity_pct"].between(0, 100).all(),
    }
    return {"passed": all(checks.values()), "checks": checks}
