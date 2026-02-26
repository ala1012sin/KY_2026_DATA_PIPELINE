"""운영 모니터링 조회 API 라우터."""

from fastapi import APIRouter

from api.schemas.responses import MonitoringSummaryResponse
from service.monitoring_service import get_monitoring_summary

router = APIRouter()


@router.get("/monitoring/summary", response_model=MonitoringSummaryResponse)
def monitoring_summary(period_days: int = 7, top_n: int = 5):
    """예측/시뮬 API 운영 집계 요약을 반환한다."""
    return get_monitoring_summary(period_days=period_days, top_n=top_n)
