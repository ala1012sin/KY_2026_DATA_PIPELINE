"""최적화 라우터용 요청 스키마."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PeakDispatchRequest(BaseModel):
    """피크 분배 MILP 테스트 요청 스키마."""
    model_config = ConfigDict(extra="forbid")

    lookback_hours: int = Field(24, ge=1, le=24 * 31, description="예측/OP_STATUS 조회 기간(시간, 1~744)")
    customer_id: Optional[str] = Field(None, description="대상 회사 ID(CUSTOMER_ID)")
    idle_op_status_threshold: float = Field(0.05, ge=0.0, le=1.0, description="미가동 판단 OP_STATUS 평균 임계치")
