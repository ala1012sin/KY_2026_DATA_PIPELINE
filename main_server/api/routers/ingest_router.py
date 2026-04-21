"""외부 스케줄러 수신 API 라우터."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.schemas.ingest import PemsProIngestResponse, PemsProIngestRow
from service.ingest_service import ingest_pems_pro_rows
from service.monitoring_service import record_api_event
from setting.database_orm import db_connection_pool

router = APIRouter()


@router.post("/ingest/pems-pro", response_model=PemsProIngestResponse)
def ingest_pems_pro(rows: List[PemsProIngestRow], db: Session = Depends(db_connection_pool)):
    """외부 원격 스케줄러가 전송한 PEMS_PRO 로그를 적재한다."""
    if not rows:
        raise HTTPException(status_code=400, detail="최소 1건 이상의 데이터가 필요합니다")

    try:
        result = ingest_pems_pro_rows(db, rows)
        record_api_event(endpoint="/ingest/pems-pro", device_id=None, status_code=200)
        return result
    except HTTPException:
        raise
    except Exception as e:
        record_api_event(endpoint="/ingest/pems-pro", device_id=None, status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail=f"ingest_pems_pro_failed: {e}")