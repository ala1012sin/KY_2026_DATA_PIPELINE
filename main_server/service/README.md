# Service 레이어 가이드

API 라우터와 스케줄러 잡이 사용하는 비즈니스 로직이 위치하는 디렉터리입니다.

## 역할

- 요청/응답 처리를 비즈니스 로직과 분리합니다.
- 예측, 시뮬레이션, 최적화, 적재, 모니터링 로직을 한곳에서 관리합니다.
- 내부 계산 단위는 유지하면서 공개 응답 단위만 변환합니다.

## 주요 파일

- `prediction_service.py`
  - 수동 예측 및 자동 예측 처리
  - 시뮬레이션 템플릿 생성 및 시뮬레이션 실행
  - 공개 응답의 전력값은 kW 단위로 변환하여 반환
- `dashboard_service.py`
  - 대시보드 페이로드 조합
  - 일 누적 전력량 조회 및 시계열 이력 구성
  - 공개 응답의 전력/에너지 값은 kW/kWh 단위로 반환
- `optimization_service.py`
  - 피크 분산 최적화 및 응답 구성
  - 공개 응답의 전력 관련 필드는 kW 단위로 반환
- `feature_prediction_service.py`
  - 피처별 전용 모델 기반 예측 처리
- `ingest_service.py`
  - 외부 PEMS 데이터 적재 흐름 처리
- `monitoring_service.py`
  - API 이벤트 로깅 및 모니터링 헬퍼 함수
- `model_store.py`, `model_constants.py`, `model_input_utils.py`
  - 모델 로딩 및 모델 입력 지원 유틸리티
- `processing/`
  - 예측 파이프라인에서 사용하는 pipeline/config/metrics/report 헬퍼

## 경계 원칙

- 서비스 함수는 FastAPI Request 객체에 의존하지 않아야 합니다.
- 라우터는 검증, HTTP 매핑, 로깅만 담당하고 비즈니스 로직은 서비스에 위임합니다.
- DB 영속화 및 내부 계산은 명시적 마이그레이션 요청이 없으면 W/Wh 단위를 유지합니다.

## 수정 체크리스트

- 공개 API 키 이름은 계약 변경 승인 없이 바꾸지 않습니다.
- 단위 변환 로직을 수정할 경우 함께 업데이트할 항목:
  - 라우터 docstring
  - `api/README.md`
  - 해당 필드를 사용하는 웹 페이지
- 서비스 반환 페이로드 변경 시 정상/엣지 케이스 smoke test를 추가합니다.
