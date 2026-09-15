select
    date,
    hour,
    season,
    avg(rented_bike_count) as avg_rented_bikes,
    max(rented_bike_count) as peak_rented_bikes,
    avg(temperature_c) as avg_temperature_c,
    avg(humidity_pct) as avg_humidity_pct,
    count(*) as observations
from bike_demand
group by 1, 2, 3
