import pandas as pd

from src.streamforge.quality import quality_report
from src.streamforge.transform import build_fact_table


def test_transform_and_quality_contract():
    source = pd.DataFrame(
        {
            "Date": ["01/12/2017"],
            "Rented Bike Count": [100],
            "Hour": [8],
            "Temperature(°C)": [5.0],
            "Humidity(%)": [60],
            "Wind speed (m/s)": [1.2],
            "Visibility (10m)": [1500],
            "Rainfall(mm)": [0.0],
            "Snowfall (cm)": [0.0],
            "Seasons": ["Winter"],
            "Holiday": ["No Holiday"],
            "Functioning Day": ["Yes"],
        }
    )
    fact = build_fact_table(source)
    report = quality_report(fact)
    assert report["passed"]
    assert fact.loc[0, "hour"] == 8
    assert fact.loc[0, "rented_bike_count"] == 100
