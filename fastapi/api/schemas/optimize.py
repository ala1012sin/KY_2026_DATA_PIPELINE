"""최적화 라우터용 요청 스키마."""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class MilpTestRequest(BaseModel):
    """MILP 테스트 요청 스키마."""
    gains_kw: List[float] = Field(..., min_length=1, description="각 선택 항목의 기대 절감량(kW)")
    costs: List[float] = Field(..., min_length=1, description="각 선택 항목의 비용")
    budget: float = Field(..., ge=0, description="총 비용 상한")
    max_actions: Optional[int] = Field(None, ge=1, description="선택 가능한 최대 항목 수")
    mandatory_indices: List[int] = Field(default_factory=list, description="반드시 선택할 항목 인덱스")


class PeakDispatchRequest(BaseModel):
    """피크 분배 MILP 테스트 요청 스키마."""
    model_config = ConfigDict(extra="forbid")

    lookback_hours: int = Field(24, ge=1, le=24 * 31, description="예측/OP_STATUS 조회 기간(시간, 1~744)")
    customer_id: Optional[str] = Field(None, description="대상 회사 ID(CUSTOMER_ID)")
    idle_op_status_threshold: float = Field(0.05, ge=0.0, le=1.0, description="미가동 판단 OP_STATUS 평균 임계치")
