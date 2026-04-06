# API README

FastAPI 기반 시뮬레이션/최적화 API 요약 문서입니다.

## Base URL
- 로컬: `http://localhost:8000`

## 공통 사항
- Content-Type: `application/json`
- 에러 코드
  - `400`: 입력값 검증 실패 또는 비즈니스 조건 불충족
  - `500`: 내부 서버 오류

---

## 1) 시뮬레이션 API

### GET `/api/simulate/devices`
- 목적: 시뮬레이션 가능한 장비 목록 조회
- Response (요약)
  - `total`: 장비 개수
  - `devices`: 장비 ID 목록

예시:
```bash
curl "http://localhost:8000/api/simulate/devices"
```

---

### GET `/api/simulate/template/{device_id}`
- 목적: 기준 시점(base) 데이터 + baseline 예측 + 변경 가능한 입력 필드 조회
- Path
  - `device_id` (string)
- Query
  - `lookback_hours` (int, optional, default=`24`)
- Response (요약)
  - `device_id`, `base_timestamp`, `base_log_id`
  - `editable_fields`: 변경 가능한 raw 입력 컬럼
  - `baseline`: 기준 예측 결과

예시:
```bash
curl "http://localhost:8000/api/simulate/template/{device_id}?lookback_hours=24"
```

---

### POST `/api/simulate/predict`
- 목적: 입력값 override 반영 후 시뮬레이션 예측 실행
- Request Body
  - `device_id` (string)
  - `overrides` (object: `{컬럼명: 값}`)
  - `lookback_hours` (int, optional, default=`24`)
  - `base_timestamp` (string, optional)
  - `base_log_id` (int, optional)
- Response (요약)
  - `baseline`: 기준 예측
  - `simulated`: 시뮬레이션 예측
  - `delta`: 기준 대비 변화량/변화율

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

---

## 2) 최적화(MILP) API

### POST `/api/optimize/milp-test`
- 목적: 선택형(0/1) MILP 테스트
- Request Body
  - `gains_kw` (float[])
  - `costs` (float[])
  - `budget` (float)
  - `max_actions` (int, optional)
  - `mandatory_indices` (int[], optional)
- Response (요약)
  - `objective_gain_kw`, `total_cost`
  - `selected_indices`, `selected_items`
  - `raw_solution`, `success`, `status`, `message`

예시:
```bash
curl -X POST "http://localhost:8000/api/optimize/milp-test" \
  -H "Content-Type: application/json" \
  -d '{
    "gains_kw": [1.2, 0.8, 1.5],
    "costs": [10, 7, 12],
    "budget": 18,
    "max_actions": 2,
    "mandatory_indices": []
  }'
```

---

### POST `/api/optimize/peak-dispatch`
목적: 회사별 장비 부하 분배로 15/30분 피크를 낮추는 MILP 실행

Request Body
  - `lookback_hours` (int, default=`24`)
  - `customer_id` (string, optional)
  - `idle_op_status_threshold` (float, default=`0.05`)

Response (요약)
  - 피크 절감량: `peak_15_reduction`, `peak_15_reduction_pct`, `peak_30_reduction`, `peak_30_reduction_pct`
  - 장비 그룹: `donor_device_ids`, `idle_device_ids`, `skipped_devices`
  - 분배 결과: `allocation_plan`, `devices`(장비별 shift/slack 포함)
  - `objective_peak_sum`, `total_slack`, `success`, `status`, `message`

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

---

## 3) 빠른 점검 순서
1. `GET /api/simulate/devices`로 장비 목록 확인
2. `GET /api/simulate/template/{device_id}`로 기준값 로드
3. `POST /api/simulate/predict`로 override 시뮬레이션 실행
4. `POST /api/optimize/peak-dispatch`로 분배 최적화 실행

## 4) 참고
- 웹 페이지
  - 시뮬레이션: `/simulate`
  - MILP 대시보드: `/milp`
