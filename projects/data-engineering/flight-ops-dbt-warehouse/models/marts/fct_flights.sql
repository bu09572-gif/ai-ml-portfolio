select
    f.*,
    w.temp_f, w.wind_speed_mph, w.precip_in, w.visibility_mi,
    case
        when f.is_cancelled then 'cancelled'
        when f.arr_delay_min is null then 'no_arrival_record'
        when f.arr_delay_min <= 15 then 'on_time'
        else 'late'
    end as arrival_status
from {{ ref('stg_flights') }} f
left join {{ ref('stg_weather') }} w
    on w.airport_code = f.origin_code and w.hour_utc = f.sched_hour_utc
