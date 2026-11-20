select
    origin_code, dest_code,
    count(*)                                   as flights,
    avg(arr_delay_min)                         as avg_arr_delay_min,
    avg(is_cancelled::int)                     as cancel_rate,
    any_value(distance_mi)                     as distance_mi
from {{ ref('fct_flights') }}
group by origin_code, dest_code
having count(*) >= 100
