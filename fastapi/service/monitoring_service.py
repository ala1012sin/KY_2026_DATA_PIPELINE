"""API 운영 모니터링 집계 서비스."""

from collections import Counter
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, List, Optional

_EVENTS: List[Dict[str, Any]] = []
_EVENTS_LOCK = Lock()
_MAX_EVENTS = 50000

_NO_DATA_KEYWORDS = (
    "선택한 기간에 해당 장비 데이터가 없습니다",
    "선택한 장비/기간에 데이터가 없습니다",
    "TB_PEMS_PRO_LOG 데이터가 없습니다",
    "전처리 가능한 데이터가 없습니다",
    "최신 데이터가 너무 오래되었습니다",
)


def _is_no_data(status_code: int, detail: str) -> bool:
    if status_code != 404:
        return False
    text = str(detail or "")
    return any(keyword in text for keyword in _NO_DATA_KEYWORDS)


def record_api_event(endpoint: str, device_id: Optional[str], status_code: int, detail: str = "") -> None:
    event = {
        "ts": datetime.now(),
        "endpoint": str(endpoint),
        "device_id": (None if not device_id else str(device_id)),
        "status_code": int(status_code),
        "no_data": _is_no_data(int(status_code), str(detail or "")),
    }

    with _EVENTS_LOCK:
        _EVENTS.append(event)
        if len(_EVENTS) > _MAX_EVENTS:
            del _EVENTS[: len(_EVENTS) - _MAX_EVENTS]


def get_monitoring_summary(period_days: int = 7, top_n: int = 5) -> Dict[str, Any]:
    period_days = max(1, min(int(period_days), 90))
    top_n = max(1, min(int(top_n), 20))

    now = datetime.now()
    since = now - timedelta(days=period_days)

    with _EVENTS_LOCK:
        rows = [row for row in _EVENTS if row["ts"] >= since]

    total_requests = len(rows)
    success_requests = sum(1 for row in rows if 200 <= int(row["status_code"]) < 300)
    failed_requests = total_requests - success_requests
    no_data_requests = sum(1 for row in rows if bool(row.get("no_data")))

    no_data_ratio_pct = 0.0 if total_requests == 0 else (no_data_requests / total_requests) * 100.0

    by_date: Dict[str, Dict[str, int]] = {}
    for row in rows:
        date_key = row["ts"].date().isoformat()
        item = by_date.setdefault(
            date_key,
            {
                "date": date_key,
                "total": 0,
                "success": 0,
                "failed": 0,
                "no_data": 0,
            },
        )
        item["total"] += 1
        if 200 <= int(row["status_code"]) < 300:
            item["success"] += 1
        else:
            item["failed"] += 1
        if bool(row.get("no_data")):
            item["no_data"] += 1

    daily_counts = [by_date[key] for key in sorted(by_date.keys())]

    top_counter = Counter(
        row["device_id"] for row in rows if row.get("device_id")
    )
    top_devices = [
        {"device_id": device_id, "count": count}
        for device_id, count in top_counter.most_common(top_n)
    ]

    return {
        "period_days": period_days,
        "total_requests": total_requests,
        "success_requests": success_requests,
        "failed_requests": failed_requests,
        "no_data_requests": no_data_requests,
        "no_data_ratio_pct": round(no_data_ratio_pct, 2),
        "daily_counts": daily_counts,
        "top_devices": top_devices,
    }
