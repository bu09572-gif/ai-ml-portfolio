select carrier as carrier_code, name as carrier_name from {{ source('raw', 'airlines') }}
