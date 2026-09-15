CREATE TABLE IF NOT EXISTS bike_demand (
    timestamp TIMESTAMPTZ NOT NULL,
    date DATE NOT NULL,
    hour SMALLINT NOT NULL,
    day_of_week SMALLINT NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    rented_bike_count INTEGER NOT NULL,
    temperature_c DOUBLE PRECISION,
    humidity_pct DOUBLE PRECISION,
    wind_speed_ms DOUBLE PRECISION,
    visibility_10m DOUBLE PRECISION,
    rainfall_mm DOUBLE PRECISION,
    snowfall_cm DOUBLE PRECISION,
    season TEXT,
    holiday TEXT,
    functioning_day TEXT,
    demand_bucket TEXT,
    PRIMARY KEY (timestamp)
);

CREATE INDEX IF NOT EXISTS idx_bike_demand_date ON bike_demand (date);
CREATE INDEX IF NOT EXISTS idx_bike_demand_hour ON bike_demand (hour);
