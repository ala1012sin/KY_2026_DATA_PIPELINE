"""API 응답 스키마 모듈."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DevicePredictionResponse(BaseModel):
    """장비 단건 예측 응답 공통 스키마."""
    device_id: str
    best_model: str
    preds: List[Dict[str, Any]]
    missing_features: List[str] = Field(default_factory=list)
    base_timestamp: Optional[str] = None


class ManualPredictResponse(BaseModel):
    """수동 예측 응답 스키마."""
    device_id: str
    best_model: str
    preds: List[Dict[str, Any]]
    missing_feature_count: int
    missing_features: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class PredictBatchError(BaseModel):
    """배치 예측 에러 항목 스키마."""
    device_id: str
    status_code: int
    detail: str


class PredictBatchResponse(BaseModel):
    """배치 예측 응답 스키마."""
    total: int
    success: int
    failed: int
    results: List[DevicePredictionResponse] = Field(default_factory=list)
    errors: List[PredictBatchError] = Field(default_factory=list)


class SimulateDevicesResponse(BaseModel):
    """시뮬레이션 장비 목록 응답 스키마."""
    total: int
    devices: List[str] = Field(default_factory=list)


class SimulationTemplateResponse(BaseModel):
    """시뮬레이션 템플릿 응답 스키마."""
    device_id: str
    base_timestamp: str
    base_log_id: Optional[int] = None
    editable_fields: Dict[str, Optional[float]] = Field(default_factory=dict)
    baseline: DevicePredictionResponse


class SimulationPredictResponse(BaseModel):
    """시뮬레이션 실행 응답 스키마."""
    device_id: str
    base_timestamp: str
    overrides: Dict[str, float] = Field(default_factory=dict)
    baseline: DevicePredictionResponse
    simulated: DevicePredictionResponse
    delta: Dict[str, float] = Field(default_factory=dict)


class ModelInfoResponse(BaseModel):
    """모델 메타정보 응답 스키마."""
    device_id: str
    best_model: str
    model_type: str
    dl_seq_len: Optional[int] = None
    required_features: List[str] = Field(default_factory=list)
    model_root: str


class ReloadModelsResponse(BaseModel):
    """모델 재로딩 응답 스키마."""
    status: str
    model_root: str


class MilpSelectedItem(BaseModel):
    """MILP 선택 항목 스키마."""
    index: int
    gain_kw: float
    cost: float


class MilpTestResponse(BaseModel):
    """MILP 테스트 응답 스키마."""
    status: str
    success: bool
    message: str
    objective_gain_kw: float
    total_cost: float
    selected_indices: List[int] = Field(default_factory=list)
    selected_items: List[MilpSelectedItem] = Field(default_factory=list)
    raw_solution: List[float] = Field(default_factory=list)


class PeakDispatchDeviceResult(BaseModel):
    """장비별 피크 분배 결과 스키마."""
    device_id: str
    customer_id: Optional[str] = None
    company_name: Optional[str] = None
    is_donor: bool
    is_idle: bool
    op_status_mean: float
    threshold: float
    baseline_15: float
    baseline_30: float
    optimized_15: float
    optimized_30: float
    delta_15: float
    delta_30: float
    shift_in_15: float
    shift_in_30: float
    shift_out_15: float
    shift_out_30: float
    required_shift_15: float
    required_shift_30: float
    distributed_targets_15: List[Dict[str, Any]] = Field(default_factory=list)
    distributed_targets_30: List[Dict[str, Any]] = Field(default_factory=list)
    distribution_text: Optional[str] = None
    slack_15: float
    slack_30: float


class PeakDispatchSkippedDevice(BaseModel):
    """최적화 대상에서 제외된 장비 스키마."""
    device_id: str
    reason: str


class PeakDispatchResponse(BaseModel):
    """피크 분배 최적화 응답 스키마."""
    status: str
    success: bool
    message: str
    lookback_hours: int
    idle_op_status_threshold: float
    device_count: int
    donor_device_ids: List[str] = Field(default_factory=list)
    idle_device_ids: List[str] = Field(default_factory=list)
    peak_15_before: float
    peak_15_after: float
    peak_30_before: float
    peak_30_after: float
    objective_peak_sum: float
    total_slack: float
    allocation_plan: List[Dict[str, Any]] = Field(default_factory=list)
    devices: List[PeakDispatchDeviceResult] = Field(default_factory=list)
    skipped_devices: List[PeakDispatchSkippedDevice] = Field(default_factory=list)
    company_summaries: List[Dict[str, Any]] = Field(default_factory=list)


class MonitoringDailyCount(BaseModel):
    """일별 호출 집계 스키마."""
    date: str
    total: int
    success: int
    failed: int
    no_data: int


class MonitoringTopDevice(BaseModel):
    """장비별 호출 상위 집계 스키마."""
    device_id: str
    count: int


class MonitoringSummaryResponse(BaseModel):
    """운영 모니터링 요약 응답 스키마."""
    period_days: int
    total_requests: int
    success_requests: int
    failed_requests: int
    no_data_requests: int
    no_data_ratio_pct: float
    daily_counts: List[MonitoringDailyCount] = Field(default_factory=list)
    top_devices: List[MonitoringTopDevice] = Field(default_factory=list)
