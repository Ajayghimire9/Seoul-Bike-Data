import pandas as pd


def build_fact_table(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["timestamp"] = pd.to_datetime(df["Date"], dayfirst=True) + pd.to_timedelta(
        df["Hour"], unit="h"
    )
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["demand_bucket"] = pd.cut(
        df["Rented Bike Count"],
        bins=[-1, 100, 500, 1000, float("inf")],
        labels=["low", "medium", "high", "very_high"],
    ).astype(str)
    return df.rename(columns={"Rented Bike Count": "rented_bike_count"})[
        [
            "timestamp",
            "date",
            "Hour",
            "day_of_week",
            "is_weekend",
            "rented_bike_count",
            "Temperature(°C)",
            "Humidity(%)",
            "Wind speed (m/s)",
            "Visibility (10m)",
            "Rainfall(mm)",
            "Snowfall (cm)",
            "Seasons",
            "Holiday",
            "Functioning Day",
            "demand_bucket",
        ]
    ].rename(
        columns={
            "Hour": "hour",
            "Temperature(°C)": "temperature_c",
            "Humidity(%)": "humidity_pct",
            "Wind speed (m/s)": "wind_speed_ms",
            "Visibility (10m)": "visibility_10m",
            "Rainfall(mm)": "rainfall_mm",
            "Snowfall (cm)": "snowfall_cm",
            "Seasons": "season",
            "Holiday": "holiday",
            "Functioning Day": "functioning_day",
        }
    )
