select faa as airport_code, name as airport_name, lat, lon, alt as altitude_ft, tz as utc_offset, tzone as time_zone
from {{ source('raw', 'airports') }}
