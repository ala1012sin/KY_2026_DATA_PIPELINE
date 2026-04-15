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


class SimulationDevicePredictionResponse(BaseModel):
    """시뮬레이션용 장비 예측 응답 스키마(경량)."""
    device_id: str
    best_model: str
    preds: List[Dict[str, Any]]
    base_timestamp: Optional[str] = None


class ManualPredictResponse(BaseModel):
    """수동 예측 응답 스키마."""
    device_id: str
    best_model: str
    preds: List[Dict[str, Any]]
    missing_feature_count: int
    missing_features: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class AutoPredictResponse(BaseModel):
    """모니터링용 자동 예측 응답 스키마(경량)."""
    device_id: str
    best_model: str
    preds: List[Dict[str, Any]]


class FeaturePredItem(BaseModel):
    """단일 피처의 15·30분 예측값."""
    y_15_pred: float
    y_30_pred: float


class FeaturePredictResponse(BaseModel):
    """피처별 예측 응답 스키마."""
    device_id: str
    best_model: str
    preds: Dict[str, FeaturePredItem]


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
    results: List[AutoPredictResponse] = Field(default_factory=list)
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
    baseline: SimulationDevicePredictionResponse


class SimulationPredictResponse(BaseModel):
    """시뮬레이션 실행 응답 스키마."""
    device_id: str
    base_timestamp: str
    overrides: Dict[str, float] = Field(default_factory=dict)
    baseline: SimulationDevicePredictionResponse
    simulated: SimulationDevicePredictionResponse
    delta: Dict[str, float] = Field(default_factory=dict)
    input_influence: Dict[str, float] = Field(default_factory=dict)


class PeakDispatchDeviceResult(BaseModel):
    """장비별 피크 분배 결과 스키마."""
    device_id: str
    is_donor: bool
    threshold: float
    baseline_15: float
    baseline_30: float
    shift_out_15: float
    shift_out_30: float
    distribution_text: Optional[str] = None


class PeakDispatchResponse(BaseModel):
    """피크 분배 최적화 응답 스키마."""
    status: str
    success: bool
    message: str
    device_count: int
    donor_device_ids: List[str] = Field(default_factory=list)
    idle_device_ids: List[str] = Field(default_factory=list)
    peak_15_reduction: float
    peak_15_reduction_pct: float
    peak_30_reduction: float
    peak_30_reduction_pct: float
    allocation_plan: List[Dict[str, Any]] = Field(default_factory=list)
    devices: List[PeakDispatchDeviceResult] = Field(default_factory=list)
    skipped_devices: List[Dict[str, Any]] = Field(default_factory=list)
