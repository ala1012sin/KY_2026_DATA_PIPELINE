"""최적화(MILP) 테스트 API 라우터."""

from fastapi import APIRouter, HTTPException

from api.schemas.optimize import MilpTestRequest, PeakDispatchRequest
from api.schemas.responses import MilpTestResponse, PeakDispatchResponse
from service.monitoring_service import record_api_event
from service.optimization_service import optimize_peak_dispatch_test, solve_test_milp

router = APIRouter()


@router.post("/optimize/milp-test", response_model=MilpTestResponse)
def milp_test(req: MilpTestRequest):
    """간단한 0/1 MILP 테스트 최적화를 수행한다."""
    try:
        result = solve_test_milp(
            gains_kw=req.gains_kw,
            costs=req.costs,
            budget=req.budget,
            max_actions=req.max_actions,
            mandatory_indices=req.mandatory_indices,
        )
        record_api_event(endpoint="/optimize/milp-test", device_id=None, status_code=200)
        return result
    except ValueError as e:
        record_api_event(endpoint="/optimize/milp-test", device_id=None, status_code=400, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        record_api_event(endpoint="/optimize/milp-test", device_id=None, status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail=f"milp_test_failed: {e}")


@router.post("/optimize/peak-dispatch", response_model=PeakDispatchResponse)
def peak_dispatch(req: PeakDispatchRequest):
    """전체 장비를 단일 그룹으로 보고 상위 사용량 장비 부하를 분배해 피크를 낮춘다."""
    try:
        result = optimize_peak_dispatch_test(
            lookback_hours=req.lookback_hours,
            customer_id=req.customer_id,
            idle_op_status_threshold=req.idle_op_status_threshold,
        )
        record_api_event(endpoint="/optimize/peak-dispatch", device_id=None, status_code=200)
        return result
    except ValueError as e:
        record_api_event(endpoint="/optimize/peak-dispatch", device_id=None, status_code=400, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        record_api_event(endpoint="/optimize/peak-dispatch", device_id=None, status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail=f"peak_dispatch_test_failed: {e}")
