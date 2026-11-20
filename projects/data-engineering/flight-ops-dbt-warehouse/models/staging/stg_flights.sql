-- One row per scheduled departure. Times in the source are local clock times stored
-- as integers (e.g. 517 = 05:17); convert them to timestamps.
with src as (
    select * from {{ source('raw', 'flights') }}
)
select
    md5(concat_ws('|', year, month, day, carrier, flight, origin, sched_dep_time)) as flight_id,
    make_date(year, month, day)                                         as flight_date,
    carrier                                                             as carrier_code,
    flight                                                              as flight_number,
    nullif(tailnum, '')                                                 as tail_number,
    origin                                                              as origin_code,
    dest                                                                as dest_code,
    make_timestamp(year, month, day, sched_dep_time // 100, sched_dep_time % 100, 0) as sched_dep_at,
    dep_delay                                                           as dep_delay_min,
    arr_delay                                                           as arr_delay_min,
    air_time                                                            as air_time_min,
    distance                                                            as distance_mi,
    dep_time is null                                                    as is_cancelled,
    arr_delay is null and dep_time is not null                          as is_diverted_or_missing_arrival,
    time_hour::timestamptz                                              as sched_hour_utc
from src
