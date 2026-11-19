select tailnum as tail_number, year as year_built, type as aircraft_type, manufacturer, model, engines, seats, engine
from {{ source('raw', 'planes') }}
