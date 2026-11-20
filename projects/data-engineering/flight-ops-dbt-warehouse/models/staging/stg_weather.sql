-- Hourly weather at each NYC airport; joins to flights on origin + scheduled hour.
select
    origin as airport_code,
    time_hour::timestamptz as hour_utc,
    temp as temp_f, dewp as dewpoint_f, humid as humidity_pct,
    wind_dir as wind_dir_deg, wind_speed as wind_speed_mph, wind_gust as wind_gust_mph,
    precip as precip_in, pressure as pressure_mb, visib as visibility_mi
from {{ source('raw', 'weather') }}
