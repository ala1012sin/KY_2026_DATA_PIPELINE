"""예측 라우터용 요청 스키마."""

from typing import Dict, List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """수동 예측 요청 스키마."""
    device_id: str
    rows: List[Dict[str, float]] = Field(..., description="feature dict 리스트")


class MultiPredictRequest(BaseModel):
    """여러 장비 자동 예측 요청 스키마."""
    device_ids: List[str] = Field(..., description="예측할 장비 ID 목록")
    lookback_hours: int = Field(24, ge=1, le=24 * 31, description="조회 기간(시간, 1~744)")
