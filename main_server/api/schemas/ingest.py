"""외부 적재용 요청 스키마."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PemsProIngestRow(BaseModel):
    """외부 스케줄러가 전송하는 TB_PEMS_PRO_LOG 1건 스키마."""

    model_config = ConfigDict(populate_by_name=True)

    device_id: UUID = Field(alias="DEVICE_ID")
    log_dt: datetime = Field(alias="LOG_DT")
    pressure: Optional[float] = Field(default=None, alias="PRESSURE")
    temperature: Optional[float] = Field(default=None, alias="TEMPERATURE")
    hz: Optional[float] = Field(default=None, alias="HZ")
    op_status: Optional[bool] = Field(default=None, alias="OP_STATUS")
    avg_voltage: Optional[float] = Field(default=None, alias="AVGVOLTAGE")
    avg_current: Optional[float] = Field(default=None, alias="AVGCURRENT")
    cur_voltage: Optional[float] = Field(default=None, alias="CURVOLTAGE")
    factor: Optional[float] = Field(default=None, alias="FACTOR")
    op_time: Optional[int] = Field(default=None, alias="OP_TIME")
    cs_usage_time: Optional[int] = Field(default=None, alias="CSUSAGETIME")
    mg_refill_time: Optional[int] = Field(default=None, alias="MGREFILLTIME")
    serial_no: Optional[str] = Field(default=None, alias="SERIAL_NO")
    device_type: Optional[str] = Field(default=None, alias="DEVICE_TYPE")
    device_num: Optional[str] = Field(default=None, alias="DEVICE_NUM")
    customer_id: Optional[UUID] = Field(default=None, alias="CUSTOMER_ID")
    device_name: Optional[str] = Field(default=None, alias="DEVICE_NAME")


class IngestErrorItem(BaseModel):
    """배치 적재 중 실패한 레코드 요약."""

    index: int
    device_id: Optional[str] = None
    log_dt: Optional[str] = None
    detail: str


class PemsProIngestResponse(BaseModel):
    """외부 적재 결과 응답."""

    total: int
    inserted: int
    skipped_duplicates: int
    created_devices: int
    failed: int
    errors: List[IngestErrorItem] = Field(default_factory=list)