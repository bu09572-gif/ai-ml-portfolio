export DBT_PROFILES_DIR := .
.PHONY: data build docs
data:  ; cd ../../.. && PYTHONPATH=src python -m flightsdata.source
build: ; mkdir -p ../../../data/warehouse && dbt build
docs:  ; dbt docs generate && dbt docs serve
