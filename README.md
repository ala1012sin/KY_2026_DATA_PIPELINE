# KY-2026 Main

FastAPI-based server for prediction, simulation, monitoring, and peak dispatch optimization.

## Project Layout

- `main_server/`: Main application
- `docker-compose.yaml`: Container run configuration

## App Entry And Routing

The application starts at `main_server/main.py`.

Current registered routers under `/api`:

- `simulate`
- `monitoring`
- `optimize`
- `predict`
- `ingest`

Web pages:

- `/simulate`
- `/milp`
- `/feature-dashboard`

## Public API Groups

- Simulate
  - `GET /api/simulate/devices`
  - `GET /api/simulate/template/{device_id}`
  - `POST /api/simulate/predict`
- Monitoring
  - `GET /api/monitor/dashboard/{device_id}`
  - `GET /api/monitor/daily-energy/{device_id}`
- Optimize
  - `POST /api/optimize/peak-dispatch`
- Predict
  - `POST /api/predict`
  - `GET /api/predict/{device_id}`
  - `POST /api/predict/batch`
  - `GET /api/predict/feature/{device_id}`
- Ingest
  - `POST /api/ingest/pems-pro`

Detailed request/response examples are in `main_server/api/README.md`.

## Unit Conventions (Public Response)

- Power-like response fields are exposed as `kW`.
- Daily energy response value is exposed as `kWh`.
- Some legacy key names remain unchanged for compatibility (for example, `daily_energy_wh`, `power_w`).

## Run

### Docker Compose

```bash
docker compose up --build
```

`docker-compose.yaml` currently runs only the `app` service.

### Local

```bash
cd main_server
python -m pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Important Paths

- `main_server/api/README.md`: API usage guide
- `main_server/api/routers/README.md`: Router overview
- `main_server/service/README.md`: Service layer overview
- `main_server/scheduler/README.md`: Scheduler overview
