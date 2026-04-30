# Router Guide

This directory contains FastAPI router modules.

## Router Files

- `simulate_router.py`
  - `/simulate/devices`
  - `/simulate/template/{device_id}`
  - `/simulate/predict`
- `monitoring_router.py`
  - `/monitor/dashboard/{device_id}`
  - `/monitor/daily-energy/{device_id}`
- `optimize_router.py`
  - `/optimize/peak-dispatch`
- `predict_router.py`
  - `/predict`
  - `/predict/{device_id}`
  - `/predict/batch`
  - `/predict/feature/{device_id}`
- `ingest_router.py`
  - `/ingest/pems-pro`

## Responsibilities

- Validate request shape and query ranges.
- Map exceptions to HTTP status codes.
- Record API events when needed.
- Delegate business logic to service layer.

## Unit Contract Notes

- Several response fields now return values in kW/kWh.
- Some legacy key names intentionally remain unchanged for compatibility.
  - Example: `daily_energy_wh` key may carry kWh value.
  - Example: `allocation_plan[].power_w` key may carry kW value.

## Router Review Summary

- Router registration in `main_server/main.py` includes all router modules above.
- No router code fix was required for path conflicts or runtime errors during this update.
- Main update need was documentation alignment.

## Editing Checklist

- Keep endpoint path and method stable unless versioning is planned.
- If response payload shape changes, update:
  - `api/schemas`
  - `main_server/api/README.md`
  - web pages using the endpoint
- Keep router logic thin; put business rules in services.
