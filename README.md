# KY-2026 Main

FastAPI 기반 설비 데이터 예측/시뮬레이션/모니터링/피크 분산 최적화 서버입니다.

## 프로젝트 구조

- `main_server/`: 메인 애플리케이션
- `docker-compose.yaml`: 컨테이너 실행 설정

## 앱 시작 및 라우터

애플리케이션은 `main_server/main.py`에서 시작됩니다.

현재 `/api` 하위에 등록된 라우터:

- `simulate`
- `monitoring`
- `optimize`
- `predict`
- `ingest`

웹 페이지 경로:

- `/simulate`
- `/milp`
- `/feature-dashboard`

## 공개 API 목록

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

상세 요청/응답 예시는 `main_server/api/README.md`를 참고하세요.

## 공개 응답 단위 규칙

- 전력 관련 응답 필드는 `kW` 단위로 반환합니다.
- 일 누적 전력량 응답 값은 `kWh` 단위로 반환합니다.
- 하위 호환을 위해 일부 레거시 키 이름은 유지합니다 (예: `daily_energy_wh`, `power_w`).

## 실행 방법

### Docker Compose

```bash
docker compose up --build
```

현재 `docker-compose.yaml`은 `app` 서비스만 실행합니다.

### 로컬 실행

```bash
cd main_server
python -m pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 주요 문서 경로

- `main_server/api/README.md`: API 사용 가이드
- `main_server/api/routers/README.md`: 라우터 개요
- `main_server/service/README.md`: 서비스 레이어 개요
- `main_server/scheduler/README.md`: 스케줄러 개요
