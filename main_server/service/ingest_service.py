"""외부 스케줄러 연동용 센서 적재 서비스."""

from typing import Any, Dict, List, Tuple

from Logger import Logger as logger
from sqlalchemy.orm import Session

from api.schemas.ingest import PemsProIngestRow
from db.public.models import TB_DEVICE, TB_PEMS_PRO_LOG
from infrastructure.queryFactory.base_orm import BaseQueryFactory


DEFAULT_PEMS_PRO_DEVICE_TYPE = "2"


def _normalize_device_data(row: PemsProIngestRow) -> Dict[str, Any]:
    return {
        "device_id": row.device_id,
        "customer_id": row.customer_id,
        "serial_no": row.serial_no,
        "device_type": str(row.device_type or DEFAULT_PEMS_PRO_DEVICE_TYPE),
        "device_num": row.device_num,
        "device_name": row.device_name,
    }


def _get_or_create_device(db: Session, row: PemsProIngestRow) -> Tuple[TB_DEVICE, bool]:
    device = db.query(TB_DEVICE).filter(TB_DEVICE.device_id == row.device_id).first()
    if device:
        return device, False

    device_query = BaseQueryFactory(db, TB_DEVICE)
    device = device_query.insert_single_row(**_normalize_device_data(row))
    logger.info("[외부적재] TB_DEVICE 생성 device_id=%s", row.device_id)
    return device, True


def ingest_pems_pro_rows(db: Session, rows: List[PemsProIngestRow]) -> Dict[str, Any]:
    inserted = 0
    skipped_duplicates = 0
    created_devices = 0
    errors: List[Dict[str, Any]] = []

    for index, row in enumerate(rows):
        try:
            _, created = _get_or_create_device(db, row)
            if created:
                created_devices += 1

            existing = (
                db.query(TB_PEMS_PRO_LOG)
                .filter(
                    TB_PEMS_PRO_LOG.device_id == row.device_id,
                    TB_PEMS_PRO_LOG.log_dt == row.log_dt,
                )
                .first()
            )
            if existing:
                skipped_duplicates += 1
                continue

            pems_log_query = BaseQueryFactory(db, TB_PEMS_PRO_LOG)
            pems_log_query.insert_single_row(
                device_id=row.device_id,
                log_dt=row.log_dt,
                pressure=row.pressure,
                temperature=row.temperature,
                hz=row.hz,
                op_status=row.op_status,
                avg_voltage=row.avg_voltage,
                avg_current=row.avg_current,
                cur_voltage=row.cur_voltage,
                factor=row.factor,
                op_time=row.op_time,
                cs_usage_time=row.cs_usage_time,
                mg_refill_time=row.mg_refill_time,
            )
            inserted += 1
        except Exception as e:
            try:
                db.rollback()
            except Exception:
                pass
            logger.error("[외부적재] 레코드 적재 실패 index=%s, device_id=%s, error=%s", index, row.device_id, e)
            errors.append(
                {
                    "index": index,
                    "device_id": str(row.device_id),
                    "log_dt": row.log_dt.isoformat() if row.log_dt else None,
                    "detail": str(e),
                }
            )

    return {
        "total": len(rows),
        "inserted": inserted,
        "skipped_duplicates": skipped_duplicates,
        "created_devices": created_devices,
        "failed": len(errors),
        "errors": errors,
    }