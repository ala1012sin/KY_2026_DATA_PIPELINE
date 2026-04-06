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

    def _safe(val) -> Optional[float]:
        try:
            v = float(val)
            import math
            return None if math.isnan(v) or math.isinf(v) else v
        except Exception:
            return None

    return {
        "CURVOLTAGE": _safe(latest.get("CUR_VOLTAGE")),
        "PRESSURE": _safe(latest.get("PRESSURE")),
        "TEMPERATURE": _safe(latest.get("TEMPERATURE")),
        "HZ": _safe(latest.get("HZ")),
        "AVGCURRENT": _safe(latest.get("AVGCURRENT")),
        "AVGVOLTAGE": _safe(latest.get("AVGVOLTAGE")),
        "FACTOR": _safe(latest.get("FACTOR")),
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


def get_dashboard_data(device_id: str, lookback_hours: int = 24) -> Dict[str, Any]:
    """대시보드에 필요한 모든 데이터를 한 번에 반환한다."""
    # 1) 현재 센서값
    current = get_current_sensor_values(device_id)

    # 2) 피처별 예측 (PRESSURE, TEMPERATURE, HZ, AVGCURRENT, AVGVOLTAGE, FACTOR)
    try:
        feat_result = predict_features_one_device(device_id, lookback_hours)
        feature_preds = feat_result.get("preds", {})
    except HTTPException:
        feature_preds = {}

    # 3) 순시 전력(CURVOLTAGE) 예측
    try:
        power_result = predict_one_device(device_id, lookback_hours, 24)
        power_preds = power_result.get("preds", [{}])
        power_pred = power_preds[-1] if power_preds else {}
        power_y15 = max(0.0, float(power_pred.get("y_15_pred", 0.0)))
        power_y30 = max(0.0, float(power_pred.get("y_30_pred", 0.0)))
    except HTTPException:
        power_y15 = None
        power_y30 = None

    # 4) 일 누적 전력량(Wh)
    daily_energy_wh = get_daily_energy_wh(device_id)

    # 5) 순시 전력(현재값 W)
    current_power_w = current.get("CURVOLTAGE")
    instantaneous_w = round(current_power_w, 4) if current_power_w is not None else None

    return {
        "device_id": device_id,
        "timestamp": datetime.now().isoformat(),
        "current_values": current,
        "feature_predictions": feature_preds,
        "power_prediction": {
            "y_15_pred": power_y15,
            "y_30_pred": power_y30,
        },
        "instantaneous_power_w": instantaneous_w,
        "daily_energy_wh": daily_energy_wh,
    }
