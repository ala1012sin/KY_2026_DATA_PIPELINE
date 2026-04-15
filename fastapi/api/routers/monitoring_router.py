"""운영 모니터링 조회 API 라우터."""

from fastapi import APIRouter, HTTPException, Query

from service.dashboard_service import get_daily_energy_wh, get_dashboard_data

router = APIRouter()


@router.get("/monitor/dashboard/{device_id}")
def dashboard(
    device_id: str,
    lookback_hours: int = Query(24, ge=1, le=24 * 31),
):
    """대시보드용 통합 데이터 반환 (현재 센서값 + 피처 예측 + 전력 예측 + 일 누적 전력량)."""
    try:
        return get_dashboard_data(device_id, lookback_hours)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"dashboard_failed: {e}")


@router.get("/monitor/daily-energy/{device_id}")
def daily_energy(device_id: str):
    """오늘 자정부터 현재까지의 누적 전력량(Wh)을 반환한다."""
    try:
        wh = get_daily_energy_wh(device_id)
        return {"device_id": device_id, "daily_energy_wh": wh}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"daily_energy_failed: {e}")
