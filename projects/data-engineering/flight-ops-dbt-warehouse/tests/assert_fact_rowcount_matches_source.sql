-- The weather join must not duplicate or drop flights.
select 1
from (select count(*) n from {{ ref('fct_flights') }}) a,
     (select count(*) n from {{ source('raw', 'flights') }}) b
where a.n <> b.n
