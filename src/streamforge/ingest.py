from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "Date",
    "Rented Bike Count",
    "Hour",
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Rainfall(mm)",
    "Snowfall (cm)",
    "Seasons",
    "Holiday",
    "Functioning Day",
}


def load_source(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {file_path}")
    frame = pd.read_csv(file_path, encoding="ISO-8859-1")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return frame


def validate_source(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Source dataset is empty")
    if frame["Rented Bike Count"].isna().any():
        raise ValueError("Rented Bike Count contains null values")
    if (frame["Rented Bike Count"] < 0).any():
        raise ValueError("Rented Bike Count cannot be negative")
    return frame
