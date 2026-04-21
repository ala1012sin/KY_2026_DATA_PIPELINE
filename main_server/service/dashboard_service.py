"""대시보드용 통합 데이터 서비스."""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import HTTPException
import pandas as pd

from service.feature_prediction_service import predict_features_one_device
from service.prediction_service import predict_one_device
from service.processing.config import PreprocessConfig
from service.processing.pipeline import DataNotFoundError, fetch_pems_pro_log_df
from service.processing.steps import resample_15m_per_device


def _safe_float(value: Any) -> Optional[float]:
    """NaN/inf를 제외한 숫자값만 안전하게 float으로 변환한다."""
    try:
        v = float(value)
        import math
        return None if math.isnan(v) or math.isinf(v) else v
    except Exception:
        return None


def get_current_sensor_values(device_id: str) -> Dict[str, Optional[float]]:
    """DB에서 지난 2시간 내 최신 센서값을 반환한다."""
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(hours=2)

    try:
        raw = fetch_pems_pro_log_df(start_dt=start_dt, end_dt=end_dt, device_ids=[device_id])
    except DataNotFoundError:
        return {}

    raw = raw[raw["DEVICE_ID"].astype(str) == str(device_id)]
    if raw.empty:
        return {}

    latest = raw.sort_values("LOG_DT").iloc[-1]

    return {
        "CURVOLTAGE": _safe_float(latest.get("CUR_VOLTAGE")),
        "PRESSURE": _safe_float(latest.get("PRESSURE")),
        "TEMPERATURE": _safe_float(latest.get("TEMPERATURE")),
        "HZ": _safe_float(latest.get("HZ")),
        "AVGCURRENT": _safe_float(latest.get("AVGCURRENT")),
        "AVGVOLTAGE": _safe_float(latest.get("AVGVOLTAGE")),
        "FACTOR": _safe_float(latest.get("FACTOR")),
    }


def get_daily_energy_wh(device_id: str) -> float:
    """오늘 자정부터 현재까지의 누적 전력량(Wh)을 반환한다.

    15분 리샘플 후 각 bin의 평균 전력(W) × 경과 시간(h) = Wh 합산.
    """
    now = datetime.now()
    today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

    try:
        raw = fetch_pems_pro_log_df(start_dt=today_midnight, end_dt=now, device_ids=[device_id])
    except DataNotFoundError:
        return 0.0

    raw = raw[raw["DEVICE_ID"].astype(str) == str(device_id)].copy()
    if raw.empty:
        return 0.0

    if "CUR_VOLTAGE" not in raw.columns:
        return 0.0

    # 15분 리샘플 → 각 bin의 평균 전력(W)
    pcfg = PreprocessConfig()
    df15 = resample_15m_per_device(
        raw, "LOG_DT", "DEVICE_ID", pcfg.resample_rule, "none"
    )
    df15 = df15[df15["DEVICE_ID"].astype(str) == str(device_id)]
    df15 = df15[df15["n_obs"].fillna(0) > 0]

    if df15.empty or "CUR_VOLTAGE" not in df15.columns:
        return 0.0

    import numpy as np
    power_w = df15["CUR_VOLTAGE"].replace([np.inf, -np.inf], float("nan")).fillna(0.0).clip(lower=0.0)

    # 마지막 미완료 15분 bin은 현재 시각까지 경과한 시간만 비례 반영한다.
    bin_start = pd.to_datetime(df15["LOG_DT"], errors="coerce")
    rule_td = pd.to_timedelta(pcfg.resample_rule)
    rule_seconds = float(rule_td.total_seconds())
    now_ts = pd.Timestamp(now)
    bin_end = bin_start + rule_td

    elapsed_seconds = (bin_end.clip(upper=now_ts) - bin_start).dt.total_seconds()
    elapsed_seconds = elapsed_seconds.clip(lower=0.0, upper=rule_seconds).fillna(0.0)
    elapsed_hours = elapsed_seconds / 3600.0

    energy_wh = float((power_w * elapsed_hours).sum())
    return round(energy_wh, 4)


def get_history_by_time(device_id: str, lookback_hours: int) -> Dict[str, Dict[str, Optional[float]]]:
    """lookback 구간의 15분 리샘플 시계열을 {timestamp: {feature: value}}로 반환한다."""
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(hours=lookback_hours)

    try:
        raw = fetch_pems_pro_log_df(start_dt=start_dt, end_dt=end_dt, device_ids=[device_id])
    except DataNotFoundError:
        return {}

    raw = raw[raw["DEVICE_ID"].astype(str) == str(device_id)].copy()
    if raw.empty:
        return {}

    pcfg = PreprocessConfig()
    df15 = resample_15m_per_device(raw, "LOG_DT", "DEVICE_ID", pcfg.resample_rule, "none")
    df15 = df15[df15["DEVICE_ID"].astype(str) == str(device_id)].copy()
    df15 = df15[df15["n_obs"].fillna(0) > 0]
    if df15.empty:
        return {}

    df15["LOG_DT"] = pd.to_datetime(df15["LOG_DT"], errors="coerce")
    df15 = df15.dropna(subset=["LOG_DT"]).sort_values("LOG_DT")

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for _, row in df15.iterrows():
        ts = pd.Timestamp(row["LOG_DT"]).isoformat()
        out[ts] = {
            "CURVOLTAGE": _safe_float(row.get("CUR_VOLTAGE")),
            "PRESSURE": _safe_float(row.get("PRESSURE")),
            "TEMPERATURE": _safe_float(row.get("TEMPERATURE")),
            "HZ": _safe_float(row.get("HZ")),
            "AVGCURRENT": _safe_float(row.get("AVGCURRENT")),
            "AVGVOLTAGE": _safe_float(row.get("AVGVOLTAGE")),
            "FACTOR": _safe_float(row.get("FACTOR")),
        }
    return out


def _fetch_feature_predictions(device_id: str, lookback_hours: int) -> Dict[str, Dict[str, Any]]:
    """피처 예측 결과를 조회하고 실패 시 빈 딕셔너리를 반환한다."""
    try:
        feat_result = predict_features_one_device(device_id, lookback_hours)
        return feat_result.get("preds", {})
    except HTTPException:
        return {}


def _fetch_power_predictions(device_id: str, lookback_hours: int) -> tuple[Optional[float], Optional[float]]:
    """전력 예측 결과에서 15분/30분 값만 추려 반환한다."""
    try:
        power_result = predict_one_device(device_id, lookback_hours, 24)
        power_pred = (power_result.get("preds") or [{}])[-1]
        return (
            max(0.0, float(power_pred.get("y_15_pred", 0.0))),
            max(0.0, float(power_pred.get("y_30_pred", 0.0))),
        )
    except HTTPException:
        return None, None


def _resolve_feature_prediction_value(feature_preds: Dict[str, Dict[str, Any]], key: str, horizon: str) -> Optional[float]:
    """피처 예측 응답에서 horizon별 값을 안전하게 읽는다."""
    pred = feature_preds.get(key)
    if pred is None:
        return None
    return _safe_float(pred.get(horizon, 0.0))


def _append_current_snapshot(
    history_by_time: Dict[str, Dict[str, Optional[float]]],
    now: datetime,
    current_values: Dict[str, Optional[float]],
    instantaneous_w: Optional[float],
) -> None:
    """현재 센서값을 현재 시각의 timeline 항목으로 추가한다."""
    ts_now = now.replace(second=0, microsecond=0).isoformat()
    history_by_time[ts_now] = {
        "CURVOLTAGE": instantaneous_w,
        "PRESSURE": current_values.get("PRESSURE"),
        "TEMPERATURE": current_values.get("TEMPERATURE"),
        "HZ": current_values.get("HZ"),
        "AVGCURRENT": current_values.get("AVGCURRENT"),
        "AVGVOLTAGE": current_values.get("AVGVOLTAGE"),
        "FACTOR": current_values.get("FACTOR"),
    }


def _append_prediction_snapshots(
    history_by_time: Dict[str, Dict[str, Optional[float]]],
    now: datetime,
    feature_preds: Dict[str, Dict[str, Any]],
    power_y15: Optional[float],
    power_y30: Optional[float],
) -> None:
    """15분/30분 미래 예측값을 timeline에 추가한다."""
    ts_15 = (now + timedelta(minutes=15)).replace(second=0, microsecond=0).isoformat()
    ts_30 = (now + timedelta(minutes=30)).replace(second=0, microsecond=0).isoformat()
    history_by_time[ts_15] = {
        "CURVOLTAGE": power_y15,
        "PRESSURE": _resolve_feature_prediction_value(feature_preds, "PRESSURE", "y_15_pred"),
        "TEMPERATURE": _resolve_feature_prediction_value(feature_preds, "TEMPERATURE", "y_15_pred"),
        "HZ": _resolve_feature_prediction_value(feature_preds, "HZ", "y_15_pred"),
        "AVGCURRENT": _resolve_feature_prediction_value(feature_preds, "AVGCURRENT", "y_15_pred"),
        "AVGVOLTAGE": _resolve_feature_prediction_value(feature_preds, "AVGVOLTAGE", "y_15_pred"),
        "FACTOR": _resolve_feature_prediction_value(feature_preds, "FACTOR", "y_15_pred"),
    }
    history_by_time[ts_30] = {
        "CURVOLTAGE": power_y30,
        "PRESSURE": _resolve_feature_prediction_value(feature_preds, "PRESSURE", "y_30_pred"),
        "TEMPERATURE": _resolve_feature_prediction_value(feature_preds, "TEMPERATURE", "y_30_pred"),
        "HZ": _resolve_feature_prediction_value(feature_preds, "HZ", "y_30_pred"),
        "AVGCURRENT": _resolve_feature_prediction_value(feature_preds, "AVGCURRENT", "y_30_pred"),
        "AVGVOLTAGE": _resolve_feature_prediction_value(feature_preds, "AVGVOLTAGE", "y_30_pred"),
        "FACTOR": _resolve_feature_prediction_value(feature_preds, "FACTOR", "y_30_pred"),
    }


def get_dashboard_data(device_id: str, lookback_hours: int = 24) -> Dict[str, Any]:
    """대시보드에 필요한 모든 데이터를 한 번에 반환한다."""
    now = datetime.now()
    current = get_current_sensor_values(device_id)
    current_values = {k: v for k, v in current.items() if k != "CURVOLTAGE"}
    feature_preds = _fetch_feature_predictions(device_id, lookback_hours)
    power_y15, power_y30 = _fetch_power_predictions(device_id, lookback_hours)
    daily_energy_wh = get_daily_energy_wh(device_id)
    current_power_w = current.get("CURVOLTAGE")
    instantaneous_w = round(current_power_w, 4) if current_power_w is not None else None
    history_by_time = get_history_by_time(device_id=device_id, lookback_hours=2)
    _append_current_snapshot(history_by_time, now, current_values, instantaneous_w)
    _append_prediction_snapshots(history_by_time, now, feature_preds, power_y15, power_y30)

    return {
        "device_id": device_id,
        "timestamp": now.isoformat(),
        "daily_energy_wh": daily_energy_wh,
        "history_by_time": history_by_time,
    }
