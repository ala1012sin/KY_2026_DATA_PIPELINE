"""피처별 예측 비즈니스 로직 서비스."""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException

from service.model_constants import AGG_RULES, ROLL_WINDOWS
from service.model_input_utils import add_current_model_aliases
from service.processing.config import PreprocessConfig
from service.processing.pipeline import (
    DataNotFoundError,
    fetch_pems_pro_log_df,
)

# feature 예측 모델 루트
_FEATURE_MODEL_ROOT = str(
    Path(os.environ.get("FEATURE_MODEL_ROOT", "./ai_models/feature_model")).expanduser().resolve()
)

# 예측 대상 피처 목록과 모델이 기대하는 피처별 라벨 매핑
FEATURE_TARGETS = ["FACTOR", "TEMPERATURE", "PRESSURE", "AVGVOLTAGE", "AVGCURRENT", "HZ"]
SELF_LAGS_FM = [1, 4, 12, 24, 48, 96] # 과거 데이터인 Lag를 몇 스텝 사용할지 정의하는 리스트
ROLL_WINS_FM = ROLL_WINDOWS # 과거 데이터의 이동 통계량을 계산할 때 사용할 윈도우 크기를 정의하는 리스트

# 현재 모델이 예측하는 피처 이름과 사용자에게 보여줄 라벨이 동일하므로 간단히 매핑. 필요시 수정 가능
_FEATURE_TARGET_TO_LABEL = {target: target for target in FEATURE_TARGETS} 

# 각 피쳐들을 어떻게 집계할지 정의해놓은 매핑표
_FM_AGG_RULES = AGG_RULES


def _feature_model_exists() -> bool:
    """feature_model 디렉터리가 존재하고 비어있지 않은지 확인한다."""
    root = Path(_FEATURE_MODEL_ROOT)
    return root.exists() and any(root.iterdir())


def _preprocess_for_feature_model(df_raw: pd.DataFrame) -> pd.DataFrame:
    """feature_model이 기대하는 형태로 원본 데이터를 전처리한다."""
    df = add_current_model_aliases(df_raw).copy()
    if df.empty:
        return df

    df["LOG_DT"] = pd.to_datetime(df["LOG_DT"], errors="coerce")
    df = df.dropna(subset=["LOG_DT"])
    if df.empty:
        return df

    if "CUR_VOLTAGE" in df.columns:
        df["CUR_VOLTAGE"] = pd.to_numeric(df["CUR_VOLTAGE"], errors="coerce").clip(lower=0)
    if "FACTOR" in df.columns:
        df["FACTOR"] = pd.to_numeric(df["FACTOR"], errors="coerce").clip(0, 1)

    agg = {col: rule for col, rule in _FM_AGG_RULES.items() if col in df.columns}
    if not agg:
        return pd.DataFrame(columns=["DEVICE_ID", "LOG_DT"])

    parts = []
    for device_id, group in df.groupby("DEVICE_ID"):
        g2 = (
            group.set_index("LOG_DT")
            .resample("15min")[list(agg.keys())]
            .agg(agg)
            .reset_index()
        )
        g2.insert(0, "DEVICE_ID", device_id)
        parts.append(g2)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["DEVICE_ID", "LOG_DT"])
    if df.empty:
        return df

    if "OP_STATUS" in df.columns:
        df["IS_STOPPED"] = (pd.to_numeric(df["OP_STATUS"], errors="coerce").fillna(0) == 0).astype(int)
    else:
        df["IS_STOPPED"] = 0

    stopped = df["IS_STOPPED"] == 1
    for col in ["CUR_VOLTAGE", "AVGVOLTAGE", "AVGCURRENT"]:
        if col in df.columns:
            df.loc[stopped, col] = 0.0

    for col in df.select_dtypes(include="number").columns:
        df[col] = df.groupby("DEVICE_ID")[col].transform(
            lambda s: s.interpolate(method="linear", limit=3)
        )

    if "CUR_VOLTAGE" in df.columns:
        def clip_iqr(series: pd.Series) -> pd.Series:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            return series.clip(lower=max(0, q1 - 1.5 * iqr), upper=q3 + 1.5 * iqr)

        df["CUR_VOLTAGE"] = df.groupby("DEVICE_ID")["CUR_VOLTAGE"].transform(clip_iqr)

    return df.sort_values(["DEVICE_ID", "LOG_DT"]).reset_index(drop=True)


def _build_fm_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """feature_model이 기대하는 형태로 피처를 가공한다."""
    tmp = pd.DataFrame(index=df.index)

    for lag in SELF_LAGS_FM:
        tmp[f"{target}_lag_{lag}"] = df[target].shift(lag) if target in df.columns else 0.0

    shifted = df[target].shift(1) if target in df.columns else pd.Series(0.0, index=df.index)
    for window in ROLL_WINS_FM:
        rolled = shifted.rolling(window, min_periods=1)
        tmp[f"{target}_roll_mean_{window}"] = rolled.mean()
        tmp[f"{target}_roll_std_{window}"] = rolled.std().fillna(0)
        tmp[f"{target}_roll_max_{window}"] = rolled.max()
        tmp[f"{target}_roll_min_{window}"] = rolled.min()

    for cross_target in [item for item in FEATURE_TARGETS if item != target]:
        tmp[f"{cross_target}_lag1"] = df[cross_target].shift(1) if cross_target in df.columns else 0.0

    tmp["HOUR"] = df["LOG_DT"].dt.hour
    tmp["DAY_OF_WEEK"] = df["LOG_DT"].dt.dayofweek
    tmp["IS_WEEKEND"] = (tmp["DAY_OF_WEEK"] >= 5).astype(int)
    tmp["IS_PEAK_HOUR"] = tmp["HOUR"].apply(lambda hour: 1 if (9 <= hour <= 12 or 13 <= hour <= 17) else 0)
    tmp["OP_STATUS_fl"] = df["OP_STATUS"].fillna(0) if "OP_STATUS" in df.columns else 0.0
    tmp["IS_STOPPED_fl"] = df["IS_STOPPED"].fillna(0) if "IS_STOPPED" in df.columns else 0.0
    return tmp


def _clip_feature_value(label: str, value: float) -> float:
    """모델 예측값을 피처별로 클리핑, FACTOR는 -1 ~ 1의 범위를 가짐, 나머지는 0 이상의 값으로""" 
    if label == "FACTOR":
        return max(-1.0, min(1.0, value))
    return max(0.0, value)


def _resolve_feature_model_horizon_dir(target_dir: Path, horizon_min: int) -> Path:
    """ horizon_min이 15 또는 30이면 해당 서브디렉토리를 우선적으로 반환하고, 그렇지 않으면 target_dir 자체를 반환한다."""
    horizon_dir = target_dir / f"{int(horizon_min)}min"
    if horizon_dir.exists():
        return horizon_dir
    return target_dir


def _load_feature_model_horizon_prediction(target_dir: Path, df: pd.DataFrame, target: str, horizon_min: int) -> float | None:
    """target_dir에서 horizon_min에 해당하는 모델을 찾아 예측값을 반환한다. 모델이나 예측값이 없으면 None을 반환한다."""
    model_dir = _resolve_feature_model_horizon_dir(target_dir, horizon_min)

    feat_cols_path = model_dir / "feature_cols.json"
    model_path = model_dir / "model.pkl"
    scaler_path = model_dir / "feat_scaler.joblib"
    if not (feat_cols_path.exists() and model_path.exists() and scaler_path.exists()):
        return None

    feature_cols = json_load(feat_cols_path)
    tmp = _build_fm_features(df, target)
    X = tmp.reindex(columns=feature_cols).fillna(0).to_numpy(dtype=np.float32)

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    preds_arr = model.predict(scaler.transform(X))
    if len(preds_arr) == 0:
        return None
    return _clip_feature_value(target, float(preds_arr[-1]))


def _predict_features_with_feature_model(device_id: str, lookback_hours: int) -> Dict[str, Any]:
    """장비 한 대에 대해 feature_model 기반 피처 예측을 수행한다."""
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(hours=lookback_hours)

    try:
        raw = fetch_pems_pro_log_df(start_dt=start_dt, end_dt=end_dt, device_ids=[device_id])
    except DataNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    df = _preprocess_for_feature_model(raw)
    df = df[df["DEVICE_ID"].astype(str) == str(device_id)].copy()
    if df.empty:
        raise HTTPException(status_code=404, detail="해당 장비의 피처 예측용 데이터가 없습니다")

    preds: Dict[str, Dict[str, float]] = {}
    best_model_name = "feature_model"
    found_any = False
    feature_root = Path(_FEATURE_MODEL_ROOT) / device_id

    for target in FEATURE_TARGETS:
        target_dir = feature_root / target
        if not target_dir.exists():
            continue
        pred_15 = _load_feature_model_horizon_prediction(target_dir, df, target, 15)
        pred_30 = _load_feature_model_horizon_prediction(target_dir, df, target, 30)
        if pred_15 is None and pred_30 is None:
            continue

        y15 = pred_15 if pred_15 is not None else pred_30
        y30 = pred_30 if pred_30 is not None else y15
        preds[_FEATURE_TARGET_TO_LABEL[target]] = {
            "y_15_pred": y15,
            "y_30_pred": y30,
        }
        found_any = True

        meta_path = _resolve_feature_model_horizon_dir(target_dir, 15) / "meta.json"
        if not meta_path.exists():
            meta_path = _resolve_feature_model_horizon_dir(target_dir, 30) / "meta.json"
        if meta_path.exists():
            meta = json_load(meta_path)
            best_model_name = str(meta.get("best_model", best_model_name))

    if not found_any:
        raise HTTPException(
            status_code=404,
            detail="해당 장비의 feature_model 예측 결과를 얻을 수 없습니다",
        )

    return {
        "device_id": device_id,
        "best_model": best_model_name,
        "preds": preds,
    }


def predict_features_one_device(device_id: str, lookback_hours: int) -> Dict[str, Any]:
    """장비 한 대에 대해 feature_model 기반 피처 예측을 수행한다."""
    if lookback_hours <= 0:
        raise HTTPException(status_code=400, detail="lookback_hours는 1 이상이어야 합니다")
    if lookback_hours > 24 * 31:
        raise HTTPException(status_code=400, detail="lookback_hours가 너무 큽니다(최대 744시간)")

    if not _feature_model_exists():
        raise HTTPException(status_code=503, detail="feature_model 디렉터리가 비어 있습니다")
    return _predict_features_with_feature_model(device_id, lookback_hours)


def json_load(path: Path) -> Union[Dict[str, Any], List[Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
