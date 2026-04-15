# KY-2026 Main

FastAPI 기반 설비 데이터 예측/시뮬레이션 서버입니다.  
현재 공개 기능은 시뮬레이션, 장비 대시보드, 일일 전력량 조회, 피크 분산 최적화와 웹 화면 제공에 맞춰 정리되어 있습니다.

## 구성 요약

- `fastapi/`: 메인 애플리케이션
- `postgres/`: PostgreSQL 초기화용 파일
- `docs/`: 구조 설계 및 작업 문서
- `docker-compose.yaml`: 앱 컨테이너 실행 설정

## 현재 공개 기능

- 시뮬레이션 대상 장비 조회
- 시뮬레이션 템플릿 조회
- 입력값 override 기반 시뮬레이션 예측
- 장비별 대시보드 데이터 조회
- 장비별 일일 누적 전력량 조회
- 회사 단위 피크 분산 최적화
- 웹 화면 제공
  - `/simulate`
  - `/milp`
  - `/feature-dashboard`

## 실행 구조

애플리케이션은 [fastapi/main.py](/home/dongjae1012/KY-2026-main/fastapi/main.py:1)에서 시작됩니다.

- FastAPI 앱 생성
- SQLAlchemy 메타데이터 기준 테이블 생성
- 라우터 등록
  - `simulate`
  - `monitoring`
  - `optimize`
- `/web` 정적 파일 마운트
- 루트(`/`) 접속 시 `simulate.html` 우선 제공

현재 `docker-compose.yaml` 기준으로는 `app` 서비스만 실행되며, `fastapi/.env`를 환경변수 파일로 사용합니다.

## 공개 API 엔드포인트

### Simulate

- `GET /api/simulate/devices`
- `GET /api/simulate/template/{device_id}`
- `POST /api/simulate/predict`

### Monitor

- `GET /api/monitor/dashboard/{device_id}`
- `GET /api/monitor/daily-energy/{device_id}`

### Optimize

- `POST /api/optimize/peak-dispatch`

상세 요청/응답 예시는 [fastapi/api/README.md](/home/dongjae1012/KY-2026-main/fastapi/api/README.md:1)를 참고하면 됩니다.

## 주요 디렉터리

```text
.
├── docker-compose.yaml
├── docs
├── fastapi
│   ├── main.py
│   ├── requirements.txt
│   ├── .env
│   ├── api
│   │   ├── routers
│   │   └── schemas
│   ├── db
│   ├── infrastructure
│   ├── scheduler
│   ├── service
│   ├── ai_models
│   │   ├── current
│   │   └── feature_model
│   └── web
└── postgres
    ├── Dockerfile
    └── init.sql
```

## 모델 디렉터리

- `fastapi/ai_models/current`
  - 전력 예측용 현재 모델
  - `classification`, `regression`, `cluster_meta.json`, `device_thresholds.json` 포함
- `fastapi/ai_models/feature_model`
  - 센서 피처 예측용 모델

## 주요 패키지

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

실행 후 주요 접속 경로:

- 앱: `http://localhost:8000`
- Swagger Docs: `http://localhost:8000/docs`
- 시뮬레이션 페이지: `http://localhost:8000/simulate`
- MILP 페이지: `http://localhost:8000/milp`
- 피처 대시보드: `http://localhost:8000/feature-dashboard`

### 로컬 실행

```bash
cd fastapi
python -m pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 참고

- 앱 컨테이너 시간대는 `Asia/Seoul`입니다.
- 앱 시작 시 DB 테이블 생성 로직이 수행됩니다.
- 현재 공개되지 않는 내부/관리용 API는 README에서 제외했습니다.
