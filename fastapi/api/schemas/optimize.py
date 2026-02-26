"""최적화 라우터용 요청 스키마."""

from typing import List, Optional

from pydantic import BaseModel, Field


class MilpTestRequest(BaseModel):
    """MILP 테스트 요청 스키마."""
    gains_kw: List[float] = Field(..., min_length=1, description="각 선택 항목의 기대 절감량(kW)")
    costs: List[float] = Field(..., min_length=1, description="각 선택 항목의 비용")
    budget: float = Field(..., ge=0, description="총 비용 상한")
    max_actions: Optional[int] = Field(None, ge=1, description="선택 가능한 최대 항목 수")
    mandatory_indices: List[int] = Field(default_factory=list, description="반드시 선택할 항목 인덱스")


class PeakDispatchRequest(BaseModel):
    """피크 분배 MILP 테스트 요청 스키마."""
    lookback_hours: int = Field(24, ge=1, description="예측/OP_STATUS 조회 기간(시간)")
    top_k: int = Field(2, ge=1, description="부하를 분배할 상위 사용량 장비 수")
    idle_op_status_threshold: float = Field(0.05, ge=0.0, le=1.0, description="미가동 판단 OP_STATUS 평균 임계치")
    force_exceed_demo: bool = Field(False, description="테스트용: 상위 부하 장비의 임계치 초과를 강제로 만들지 여부")
    force_exceed_margin_ratio: float = Field(0.05, ge=0.01, le=0.5, description="강제 초과 시 임계치 하향 비율(기본 5%)")
