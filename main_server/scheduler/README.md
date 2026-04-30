# Scheduler Guide

This directory contains periodic job runners.

## Structure

- `sensor/scheduler.py`
  - Sensor batch job logic.
- `EW/scheduler.py`
  - EW batch job logic.

## Runtime Integration

- Scheduler instance is created in `main_server/main.py` using `AsyncIOScheduler`.
- Job registration is currently commented out in `main_server/main.py`.
- App startup (`lifespan`) starts the scheduler; app shutdown stops it.

## Current Behavior

- Scheduler framework is active.
- Periodic jobs are disabled by code comments.
- This means no interval-triggered sensor/EW batch execution until jobs are re-enabled.

## Re-enable Flow

1. Restore `@scheduler.scheduled_job(...)` decorators in `main_server/main.py`.
2. Validate DB connection lifecycle in each scheduled run.
3. Monitor logs for execution interval and failure handling.

## Editing Checklist

- Keep job methods idempotent when possible.
- Avoid long blocking logic in scheduled callbacks.
- Ensure exceptions are logged and do not crash scheduler lifecycle.
