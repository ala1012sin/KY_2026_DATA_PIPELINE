# Router 가이드

FastAPI 라우터 모듈이 위치하는 디렉터리입니다.

## 라우터 파일 목록

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

## 라우터 책임

- 요청 구조 및 쿼리 파라미터 범위 검증
- 예외를 HTTP 상태 코드로 매핑
- 필요 시 API 이벤트 기록
- 비즈니스 로직은 서비스 레이어에 위임

## 단위 계약 메모

- 공개 응답의 전력 관련 필드는 kW/kWh 단위로 반환합니다.
- 하위 호환을 위해 일부 레거시 키 이름은 의도적으로 유지합니다.
  - 예: `daily_energy_wh` 키가 실제로는 kWh 값을 담습니다.
  - 예: `allocation_plan[].power_w` 키가 실제로는 kW 값을 담습니다.

## 수정 체크리스트

- 버전 관리 계획 없이 엔드포인트 경로나 HTTP 메서드를 바꾸지 않습니다.
- 응답 페이로드 구조가 바뀌면 함께 업데이트할 항목:
  - `api/schemas/`
  - `main_server/api/README.md`
  - 해당 엔드포인트를 사용하는 웹 페이지
- 라우터 로직은 얇게 유지하고, 비즈니스 규칙은 서비스에 넣습니다.
