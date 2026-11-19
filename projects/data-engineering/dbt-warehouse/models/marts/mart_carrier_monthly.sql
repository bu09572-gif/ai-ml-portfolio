-- On-time performance by carrier and month (DOT definition: arrival within 15 minutes).
select
    c.carrier_name,
    f.carrier_code,
    date_trunc('month', f.flight_date)                                   as month,
    count(*)                                                             as flights,
    avg(f.is_cancelled::int)                                             as cancel_rate,
    avg(case when f.arrival_status = 'on_time' then 1 when f.arrival_status = 'late' then 0 end) as on_time_rate,
    median(f.arr_delay_min)                                              as median_arr_delay_min,
    quantile_cont(f.arr_delay_min, 0.9)                                  as p90_arr_delay_min
from {{ ref('fct_flights') }} f
join {{ ref('dim_carrier') }} c using (carrier_code)
group by all
