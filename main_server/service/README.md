# Service Layer Guide

This directory contains business logic used by API routers and scheduler jobs.

## Role

- Keeps request/response handling out of business logic.
- Centralizes prediction, simulation, optimization, ingestion, and monitoring logic.
- Converts response units where needed while preserving internal calculation units.

## Key Files

- `prediction_service.py`
  - Manual prediction and auto prediction.
  - Simulation template and simulation run.
  - Public power response values are exposed as kW.
- `dashboard_service.py`
  - Dashboard payload composition.
  - Daily energy retrieval and history construction.
  - Public dashboard power/energy response values are exposed as kW/kWh.
- `optimization_service.py`
  - Peak dispatch optimization and response shaping.
  - Public power-like response fields are exposed as kW.
- `feature_prediction_service.py`
  - Feature-wise prediction path.
- `ingest_service.py`
  - External PEMS data ingestion flow.
- `monitoring_service.py`
  - API event logging and monitoring helper functions.
- `model_store.py`, `model_constants.py`, `model_input_utils.py`
  - Model loading and model-input support utilities.
- `processing/`
  - Pipeline/config/metrics/report helpers used by prediction paths.

## Boundaries

- Services should not depend on FastAPI request objects.
- Routers should call services and only do validation, HTTP mapping, and logging.
- DB persistence and internal calculations can keep legacy internal units (W/Wh) unless migration is explicitly required.

## Editing Checklist

- Keep public API key names stable unless contract change is approved.
- If unit conversion is changed, update:
  - router docs
  - API README
  - web pages consuming the field
- Add tests or smoke checks for both normal and edge cases when changing service return payloads.
