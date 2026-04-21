# API README

현재 외부에 공개되는 사용자용 API 요약 문서입니다.

## Base URL

- 로컬: `http://localhost:8000`

## 공통 사항

- Content-Type: `application/json`
- 에러 코드
  - `400`: 입력값 검증 실패 또는 비즈니스 조건 불충족
  - `404`: 장비 또는 기준 데이터 없음
  - `500`: 내부 서버 오류

## 1) 시뮬레이션 API

### GET `/api/simulate/devices`

- 목적: 시뮬레이션 가능한 장비 목록 조회
- Response 요약
  - `total`: 장비 개수
  - `devices`: 장비 ID 목록

예시:

```bash
curl "http://localhost:8000/api/simulate/devices"
```

### GET `/api/simulate/template/{device_id}`

- 목적: 기준 시점 raw 값, 변경 가능한 입력 필드, baseline 예측 조회
- Path
  - `device_id` (string)
- Query
  - `lookback_hours` (int, optional, default=`24`)
- Response 요약
  - `device_id`
  - `base_timestamp`
  - `base_log_id`
  - `editable_fields`
  - `baseline`

예시:

```bash
curl "http://localhost:8000/api/simulate/template/{device_id}?lookback_hours=24"
```

### POST `/api/simulate/predict`

- 목적: 입력값 override 반영 후 시뮬레이션 예측 실행
- Request Body
  - `device_id` (string)
  - `overrides` (object: `{컬럼명: 값}`)
  - `lookback_hours` (int, optional, default=`24`)
  - `base_timestamp` (string, optional)
  - `base_log_id` (int, optional)
- Response 요약
  - `baseline`
  - `simulated`
  - `delta`
  - `input_influence`

예시:

```bash
curl -X POST "http://localhost:8000/api/simulate/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "YOUR_DEVICE_ID",
    "lookback_hours": 24,
    "base_timestamp": "",
    "base_log_id": null,
    "overrides": {
      "CSUSAGETIME": 120,
      "MGREFILLTIME": 10
    }
  }'
```

## 2) 모니터링 API

### GET `/api/monitor/dashboard/{device_id}`

- 목적: 장비 대시보드 표시용 통합 데이터 조회
- Query
  - `lookback_hours` (int, optional, default=`24`)
- Response 요약
  - `device_id`
  - `timestamp`
  - `daily_energy_wh`
  - `history_by_time`

예시:

```bash
curl "http://localhost:8000/api/monitor/dashboard/{device_id}?lookback_hours=24"
```

### GET `/api/monitor/daily-energy/{device_id}`

- 목적: 오늘 자정부터 현재까지의 누적 전력량 조회
- Response 요약
  - `device_id`
  - `daily_energy_wh`

예시:

```bash
curl "http://localhost:8000/api/monitor/daily-energy/{device_id}"
```

## 3) 최적화 API

### POST `/api/optimize/peak-dispatch`

- 목적: 회사별 장비 부하 분배로 15분/30분 피크를 낮추는 최적화 실행
- Request Body
  - `lookback_hours` (int, default=`24`)
  - `customer_id` (string, optional)
  - `idle_op_status_threshold` (float, default=`0.05`)
- Response 요약
  - `status`, `success`, `message`
  - `device_count`
  - `donor_device_ids`, `idle_device_ids`
  - `peak_15_reduction`, `peak_30_reduction`
  - `allocation_plan`
  - `devices`
  - `skipped_devices`

예시:

```bash
curl -X POST "http://localhost:8000/api/optimize/peak-dispatch" \
  -H "Content-Type: application/json" \
  -d '{
    "lookback_hours": 24,
    "customer_id": "c3af7a38-2e25-4159-8a5a-77fa3eebf4d4",
    "idle_op_status_threshold": 0.05
  }'
```

## 4) 외부 적재 API

### POST `/api/ingest/pems-pro`

- 목적: 원격 데스크탑/윈도우 작업 스케줄러 등 외부 프로세스가 전송한 PEMS 로그를 수신해 `TB_PEMS_PRO_LOG`에 적재
- Request Body
  - JSON 배열
  - 필수: `DEVICE_ID`, `LOG_DT`
  - 권장: `PRESSURE`, `TEMPERATURE`, `HZ`, `OP_STATUS`, `AVGVOLTAGE`, `AVGCURRENT`, `CURVOLTAGE`, `FACTOR`, `OP_TIME`, `CSUSAGETIME`, `MGREFILLTIME`
  - 선택: `SERIAL_NO`, `DEVICE_TYPE`, `DEVICE_NUM`, `CUSTOMER_ID`, `DEVICE_NAME`
- 동작
  - `DEVICE_ID`가 `TB_DEVICE`에 없으면 placeholder 장비를 자동 생성
  - 같은 `DEVICE_ID + LOG_DT`가 이미 있으면 중복으로 보고 스킵
- Response 요약
  - `total`, `inserted`, `skipped_duplicates`, `created_devices`, `failed`, `errors`

예시:

```bash
curl -X POST "http://localhost:8000/api/ingest/pems-pro" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "DEVICE_ID": "dc291c30-2a18-4199-a33c-3020a57ee4bb",
      "LOG_DT": "2026-04-21T10:00:00",
      "PRESSURE": 5.1,
      "TEMPERATURE": 32.4,
      "HZ": 29.8,
      "OP_STATUS": true,
      "AVGVOLTAGE": 381.2,
      "AVGCURRENT": 12.8,
      "CURVOLTAGE": 384100,
      "FACTOR": 0.94,
      "OP_TIME": 1180,
      "CSUSAGETIME": 220,
      "MGREFILLTIME": 80,
      "SERIAL_NO": "2412310908",
      "DEVICE_TYPE": "2",
      "DEVICE_NUM": "1",
      "DEVICE_NAME": "PEMS PRO 1"
    }
  ]'
```

## 5) 빠른 점검 순서

1. `GET /api/simulate/devices`로 장비 목록 확인
2. `GET /api/simulate/template/{device_id}`로 기준값 로드
3. `POST /api/simulate/predict`로 override 시뮬레이션 실행
4. `GET /api/monitor/dashboard/{device_id}`로 대시보드 응답 확인
5. `POST /api/optimize/peak-dispatch`로 피크 분산 최적화 실행

## 6) 웹 경로

- 시뮬레이션: `/simulate`
- MILP 대시보드: `/milp`
- 피처 대시보드: `/feature-dashboard`
