# KY-2026 Main

이 프로젝트는 FastAPI 기반의 설비 데이터 처리/예측 서버입니다. 장비별 AI 모델을 이용한 예측, 시뮬레이션, 운영 모니터링, MILP 기반 최적화 테스트 기능을 포함하고 있으며, 정적 웹 페이지도 함께 제공합니다.

## 구성 요약

- `fastapi/`: 메인 애플리케이션
- `postgres/`: PostgreSQL 초기화용 파일
- `docker-compose.yaml`: 앱 컨테이너 실행 설정

## 주요 기능

- 장비별 AI 예측 API
- 다중 장비 배치 예측 API
- 시뮬레이션 템플릿 및 시뮬레이션 예측 API
- 모델 메타정보 조회 및 모델 캐시 리로드 API
- API 사용량/상태 모니터링 요약 API
- MILP 기반 최적화 테스트 API
- `/simulate`, `/milp` 정적 웹 페이지 제공
- APScheduler 기반 60초 주기 센서 배치 실행

## 실행 구조

애플리케이션은 `fastapi/main.py`에서 시작됩니다.

- FastAPI 앱 생성
- SQLAlchemy 메타데이터 기준 테이블 생성
- APScheduler 시작
- `/api` 하위 라우터 등록
- `/web` 정적 파일 마운트
- 루트(`/`) 접속 시 `simulate.html` 우선 제공

현재 `docker-compose.yaml` 기준으로는 `app` 서비스만 실행되며, `fastapi/.env`를 환경변수 파일로 사용합니다. `postgres/` 디렉터리에는 별도 PostgreSQL 이미지 구성 예시와 `uuid-ossp` 확장 초기화 SQL이 포함되어 있습니다.

## API 엔드포인트

### Predict

- `POST /api/predict`
- `GET /api/predict/{device_id}`
- `POST /api/predict/batch`

### Simulate

- `GET /api/simulate/devices`
- `GET /api/simulate/template/{device_id}`
- `POST /api/simulate/predict`

### Model

- `GET /api/model-info/{device_id}`
- `POST /api/reload-models`

### Monitoring

- `GET /api/monitoring/summary`

### Optimize

- `POST /api/optimize/milp-test`
- `POST /api/optimize/peak-dispatch-test`

## 주요 디렉터리

```text
.
├── docker-compose.yaml
├── fastapi
│   ├── main.py
│   ├── requirements.txt
│   ├── .env
│   ├── api
│   │   ├── routers
│   │   └── schemas
│   ├── db
│   ├── service
│   ├── scheduler
│   ├── ai_models
│   └── web
└── postgres
    ├── Dockerfile
    └── init.sql
```

## 의존성

주요 패키지는 아래와 같습니다.

- `fastapi`
- `uvicorn`
- `apscheduler`
- `sqlalchemy`
- `psycopg`, `psycopg2-binary`
- `requests`
- `pydantic`
- `tensorflow`
- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`

## 실행 방법

### Docker Compose

```bash
docker compose up --build
```

실행 후 기본 접속 경로:

- 앱: `http://localhost:8000`
- Swagger Docs: `http://localhost:8000/docs`
- 시뮬레이션 페이지: `http://localhost:8000/simulate`
- MILP 페이지: `http://localhost:8000/milp`

### 로컬 실행

```bash
cd fastapi
python -m pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 환경변수


```env
# fastapi/.env

DB_URL=postgresql+psycopg://postgres:Manair5568.@34.47.96.225:5432/postgres?sslmode=require
DB_HOST=34.47.96.225
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=Manair5568.
DB_NAME=postgres

# API 관련 설정
KY_DEVICE_DATA_URL=https://api-wmumpxg2lq-du.a.run.app/api/device-data
KY_ERROR_DATA_URL=https://api-wmumpxg2lq-du.a.run.app/api/error-data
#KY_CUSTOMER_DATA_URL=https://api-wmumpxg2lq-du.a.run.app/api/customer-data # 테스트
KY_API_TOKEN=
KY_API_TIMEOUT=10
KY_API_RETRIES=2
KY_API_RETRY_BACKOFF=1.0
```

## 참고

- 앱 컨테이너는 Python `3.9.23` 기반 이미지로 빌드됩니다.
- 컨테이너 시간대는 `Asia/Seoul`로 설정되어 있습니다.
- 앱 시작 시 DB 테이블 생성 로직이 수행됩니다.
- 센서 스케줄러는 60초마다 최근 1분 데이터를 처리하도록 등록되어 있습니다.
