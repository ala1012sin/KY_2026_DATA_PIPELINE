"""시뮬레이션 라우터용 요청 스키마."""

from typing import Dict, Optional

from pydantic import BaseModel, Field


class SimulatePredictRequest(BaseModel):
    """원본(raw) 값 변경 기반 시뮬레이션 요청 스키마."""
    device_id: str
    overrides: Dict[str, float] = Field(..., description="수정할 raw 컬럼 값")
    lookback_hours: int = Field(24, description="조회 기간(시간)")
    base_timestamp: str = Field("", description="기준 시각(ISO8601, 비우면 최신)")
    base_log_id: Optional[int] = Field(None, description="기준 행 LOG_ID")
    save_log: bool = Field(True, description="시뮬레이션 로그 저장 여부")
