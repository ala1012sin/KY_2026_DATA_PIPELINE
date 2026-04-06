"""시뮬레이션 관련 API 라우터."""

import logging

from fastapi import APIRouter, HTTPException, Query

from api.schemas.simulate import SimulatePredictRequest
from api.schemas.responses import SimulationPredictResponse, SimulationTemplateResponse, SimulateDevicesResponse
from service.monitoring_service import record_api_event
from service.prediction_service import (
    build_simulation_template,
    list_model_device_ids,
    run_simulation,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/simulate/devices", response_model=SimulateDevicesResponse)
def list_simulation_devices():
    """시뮬레이션 대상 장비 ID 목록을 반환한다."""
    # 드롭다운/선택 UI용 장비 리스트
    try:
        devices = list_model_device_ids()
        result = {
            "total": len(devices),
            "devices": devices,
        }
        record_api_event(endpoint="/simulate/devices", device_id=None, status_code=200)
        return result
    except HTTPException as e:
        record_api_event(endpoint="/simulate/devices", device_id=None, status_code=e.status_code, detail=str(e.detail))
        raise
    except Exception as e:
        logger.exception("simulate devices failed")
        record_api_event(endpoint="/simulate/devices", device_id=None, status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="시뮬레이션 장비 목록 조회 중 내부 오류가 발생했습니다") from e


@router.get("/simulate/template/{device_id}", response_model=SimulationTemplateResponse)
def simulate_template(device_id: str, lookback_hours: int = Query(24, ge=1, le=24 * 31)):
    """시뮬레이션 입력용 기준(raw) 값과 baseline 예측을 반환한다."""
    # 기준행, 변경 가능 필드, baseline 예측을 한 번에 조회
    try:
        result = build_simulation_template(device_id=device_id, lookback_hours=lookback_hours)
        record_api_event(endpoint="/simulate/template/{device_id}", device_id=device_id, status_code=200)
        return result
    except HTTPException as e:
        record_api_event(
            endpoint="/simulate/template/{device_id}",
            device_id=device_id,
            status_code=e.status_code,
            detail=str(e.detail),
        )
        raise
    except Exception as e:
        logger.exception("simulate template failed: device_id=%s, lookback_hours=%s", device_id, lookback_hours)
        record_api_event(endpoint="/simulate/template/{device_id}", device_id=device_id, status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="시뮬레이션 템플릿 생성 중 내부 오류가 발생했습니다") from e


@router.post("/simulate/predict", response_model=SimulationPredictResponse)
def simulate_predict(req: SimulatePredictRequest):
    """raw 값 override를 적용한 시뮬레이션 예측을 수행한다."""
    # 실제 시뮬레이션 계산은 service로 위임
    try:
        result = run_simulation(
            device_id=req.device_id,
            overrides=req.overrides,
            lookback_hours=req.lookback_hours,
            base_timestamp=req.base_timestamp,
            base_log_id=req.base_log_id,
            save_log=True,
        )
        record_api_event(endpoint="/simulate/predict", device_id=req.device_id, status_code=200)
        return result
    except HTTPException as e:
        record_api_event(
            endpoint="/simulate/predict",
            device_id=req.device_id,
            status_code=e.status_code,
            detail=str(e.detail),
        )
        raise
    except Exception as e:
        logger.exception("simulate predict failed: device_id=%s", req.device_id)
        record_api_event(endpoint="/simulate/predict", device_id=req.device_id, status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="시뮬레이션 예측 처리 중 내부 오류가 발생했습니다") from e
