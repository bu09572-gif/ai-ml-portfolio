-- Does bad weather at the origin go with more delays and cancellations?
select
    case
        when visibility_mi < 1 then '1: visibility < 1 mi'
        when precip_in > 0.1 then '2: heavy precipitation'
        when wind_speed_mph > 25 then '3: wind > 25 mph'
        when visibility_mi is null then '9: no weather record'
        else '4: normal'
    end                                        as weather_bucket,
    count(*)                                   as flights,
    avg(is_cancelled::int)                     as cancel_rate,
    avg(dep_delay_min)                         as avg_dep_delay_min
from {{ ref('fct_flights') }}
group by 1
order by 1
