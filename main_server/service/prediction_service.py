"""예측/시뮬레이션 공통 비즈니스 로직 서비스."""

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import HTTPException

from Logger import Logger as logger
from db.public.models import TB_SIMULATION_LOG
from infrastructure.queryFactory.base_orm import BaseQueryFactory
from setting.database_orm import db_connection_pool
from service.model_input_utils import (
    add_current_model_aliases,
    list_model_device_ids,
    resolve_editable_raw_fields,
)
from service.model_store import ModelStore
from service.processing.config import PreprocessConfig
from service.processing.pipeline import (
    DataNotFoundError,
    fetch_pems_pro_log_df,
)

MODEL_ROOT = str(Path(os.environ.get("MODEL_ROOT", "./ai_models/current")).expanduser().resolve())
store = ModelStore(MODEL_ROOT)
POWER_RESPONSE_KEYS = ("y_15_pred", "y_30_pred")
POWER_INPUT_FIELDS = {"CURVOLTAGE"}


def _w_to_kw(value: Any) -> Optional[float]:
    """W 단위 값을 kW 단위로 변환한다."""
    if value is None:
        return None
    try:
        return round(float(value) / 1000.0, 4)
    except Exception:
        return None


def _convert_prediction_payload_to_kw(payload: Dict[str, Any]) -> Dict[str, Any]:
    """예측 응답 payload의 전력 예측값을 kW 단위로 변환한다."""
    converted = dict(payload)
    preds = []
    for pred in payload.get("preds") or []:
        if not isinstance(pred, dict):
            preds.append(pred)
            continue
        pred_out = dict(pred)
        for key in POWER_RESPONSE_KEYS:
            if key in pred_out:
                converted_value = _w_to_kw(pred_out.get(key))
                pred_out[key] = pred_out.get(key) if converted_value is None else converted_value
        preds.append(pred_out)
    converted["preds"] = preds
    return converted


def _convert_power_fields_to_response_units(values: Dict[str, Any]) -> Dict[str, Any]:
    """응답의 전력 관련 raw 필드만 kW 단위로 변환한다."""
    converted: Dict[str, Any] = {}
    for field, value in (values or {}).items():
        if str(field).upper() in POWER_INPUT_FIELDS:
            converted_value = _w_to_kw(value)
            converted[field] = value if converted_value is None else converted_value
        else:
            converted[field] = value
    return converted


def _convert_power_fields_to_internal_units(values: Dict[str, float]) -> Dict[str, float]:
    """외부 입력의 전력 관련 raw 필드를 내부 계산용 W 단위로 되돌린다."""
    converted: Dict[str, float] = {}
    for field, value in (values or {}).items():
        numeric = float(value)
        if str(field).upper() in POWER_INPUT_FIELDS:
            converted[field] = numeric * 1000.0
        else:
            converted[field] = numeric
    return converted


def _convert_simulation_delta_to_kw(delta: Dict[str, Any]) -> Dict[str, Any]:
    """시뮬레이션 delta 응답의 전력 차이를 kW 단위로 변환한다."""
    converted = dict(delta or {})
    for key in POWER_RESPONSE_KEYS:
        if key in converted:
            converted_value = _w_to_kw(converted.get(key))
            converted[key] = converted.get(key) if converted_value is None else converted_value
    return converted


def _validate_lookback_hours(lookback_hours: int) -> None:
    """조회 lookback 입력 범위를 검증한다."""
    if lookback_hours <= 0:
        raise HTTPException(status_code=400, detail="lookback_hours는 1 이상이어야 합니다")
    if lookback_hours > 24 * 31:
        raise HTTPException(status_code=400, detail="lookback_hours가 너무 큽니다(최대 744시간)")


def _validate_max_data_age_hours(max_data_age_hours: int) -> None:
    """허용 가능한 최신 데이터 나이 입력을 검증한다."""
    if max_data_age_hours <= 0:
        raise HTTPException(status_code=400, detail="max_data_age_hours는 1 이상이어야 합니다")


def _get_runner(device_id: str):
    """장비용 예측 러너를 조회하고 모델 관련 예외를 HTTP 예외로 변환한다."""
    try:
        return store.get_runner(device_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _to_naive_timestamp(value: Any) -> pd.Timestamp:
    """타임존 포함 여부와 상관없이 naive timestamp로 정규화한다."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _enforce_freshness_or_404(latest_ts: Any, max_data_age_hours: int) -> None:
    """기준 시각이 너무 오래됐으면 404로 응답한다."""
    latest_ts = _to_naive_timestamp(latest_ts)
    now_ts = pd.Timestamp.now().tz_localize(None)
    age_hours = float((now_ts - latest_ts).total_seconds() / 3600.0)
    if age_hours > max_data_age_hours:
        raise HTTPException(
            status_code=404,
            detail=(
                f"최신 데이터가 너무 오래되었습니다. "
                f"기준시각={latest_ts.isoformat()}, 경과={age_hours:.1f}h, 허용={max_data_age_hours}h"
            ),
        )


def _resolve_required_rows(runner) -> int:
    """모델 타입에 맞는 최소 입력 행 수를 계산한다."""
    required_rows = runner.dl_seq_len if runner.model_type == "DL" else 1
    return 1 if required_rows is None else int(required_rows)


# ---------------------------------------------------------------------------
# 예측 공통 경로
# ---------------------------------------------------------------------------
def predict_from_raw_history(
    device_id: str,
    runner,
    raw: pd.DataFrame,
    max_data_age_hours: int,
    enforce_freshness: bool,
    reference_timestamp: Optional[Any] = None,
) -> Dict[str, Any]:
    if raw is None or raw.empty:
        raise HTTPException(status_code=404, detail="선택한 기간에 원본 데이터가 없습니다")

    if "DEVICE_ID" not in raw.columns:
        raw = raw.copy()
        raw["DEVICE_ID"] = device_id

    raw = raw[raw["DEVICE_ID"].astype(str) == str(device_id)].copy()
    if raw.empty:
        raise HTTPException(status_code=404, detail="선택한 기간에 해당 장비 데이터가 없습니다")

    if "LOG_DT" not in raw.columns:
        raise HTTPException(status_code=500, detail="원본 데이터에 시간 컬럼이 없습니다: LOG_DT")

    raw["LOG_DT"] = pd.to_datetime(raw["LOG_DT"], errors="coerce")
    raw = raw.dropna(subset=["LOG_DT"]).sort_values("LOG_DT").reset_index(drop=True)
    if raw.empty:
        raise HTTPException(status_code=404, detail="시간 정보가 유효한 데이터가 없습니다")

    if enforce_freshness:
        _enforce_freshness_or_404(raw.iloc[-1]["LOG_DT"], max_data_age_hours)

    selected = raw
    if reference_timestamp is not None:
        ref_ts = _to_naive_timestamp(reference_timestamp)
        selected = raw[raw["LOG_DT"] <= ref_ts].copy()
        if selected.empty:
            raise HTTPException(status_code=404, detail="기준시각 이전 원본 데이터가 없습니다")

    try:
        result = runner.predict_latest(selected, reference_timestamp=reference_timestamp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return {
        "device_id": device_id,
        "best_model": runner.best_model,
        "preds": result["preds"],
        "missing_features": runner.last_missing_features,
        "base_timestamp": result["timestamp"],
    }


def predict_one_device(
    device_id: str,
    lookback_hours: int,
    max_data_age_hours: int,
    power_in_kw: bool = False,
) -> Dict[str, Any]:
    # 자동 예측 공통 경로(조회→전처리→예측)
    _validate_lookback_hours(lookback_hours)
    _validate_max_data_age_hours(max_data_age_hours)
    runner = _get_runner(device_id)

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(hours=lookback_hours)

    try:
        raw = fetch_pems_pro_log_df(
            start_dt=start_dt,
            end_dt=end_dt,
            device_ids=[device_id],
        )
    except DataNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    raw = add_current_model_aliases(raw)
    result = predict_from_raw_history(
        device_id=device_id,
        runner=runner,
        raw=raw,
        max_data_age_hours=max_data_age_hours,
        enforce_freshness=True,
    )
    if power_in_kw:
        return _convert_prediction_payload_to_kw(result)
    return result


def predict_from_preprocessed(
    device_id: str,
    runner,
    meta: Dict[str, Any],
    max_data_age_hours: int,
    enforce_freshness: bool,
    reference_timestamp: Optional[Any] = None,
) -> Dict[str, Any]:
    # 전처리 결과(DataFrame)에서 장비/시간 검증 후 모델 입력을 구성
    df = meta.get("df_sup")
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="선택한 기간에 전처리 가능한 데이터가 없습니다")

    device_col = meta["device_col"]
    time_col = meta["time_col"]

    if device_col not in df.columns:
        raise HTTPException(status_code=500, detail=f"전처리 결과에 장비 컬럼이 없습니다: {device_col}")
    if time_col not in df.columns:
        raise HTTPException(status_code=500, detail=f"전처리 결과에 시간 컬럼이 없습니다: {time_col}")

    df = df[df[device_col].astype(str) == str(device_id)].copy()
    if df.empty:
        raise HTTPException(status_code=404, detail="선택한 기간에 해당 장비 데이터가 없습니다")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    if df.empty:
        raise HTTPException(status_code=404, detail="시간 정보가 유효한 데이터가 없습니다")

    if enforce_freshness:
        _enforce_freshness_or_404(df.iloc[-1][time_col], max_data_age_hours)

    required_rows = _resolve_required_rows(runner)

    if len(df) < required_rows:
        raise HTTPException(
            status_code=400,
            detail=f"예측에 필요한 전처리 행이 부족합니다. 필요={required_rows}, 현재={len(df)}",
        )

    selected = df
    if reference_timestamp is not None:
        ref_ts = _to_naive_timestamp(reference_timestamp)
        candidate = df[df[time_col] <= ref_ts]
        if candidate.empty:
            candidate = df[df[time_col] == df[time_col].min()]
        selected = candidate

    if len(selected) < required_rows:
        raise HTTPException(
            status_code=400,
            detail=f"기준시각 기준 예측 행이 부족합니다. 필요={required_rows}, 현재={len(selected)}",
        )

    rows_df = selected.tail(required_rows)
    rows = rows_df.to_dict(orient="records")
    preds, _ = runner.predict(rows)

    selected_ts = _to_naive_timestamp(rows_df.iloc[-1][time_col])

    return {
        "device_id": device_id,
        "best_model": runner.best_model,
        "preds": preds,
        "missing_features": runner.last_missing_features,
        "base_timestamp": selected_ts.isoformat(),
    }


def parse_base_timestamp(base_timestamp: str) -> datetime:
    # 시뮬 기준시각(ISO8601) 파싱
    if not base_timestamp:
        return datetime.now()
    try:
        parsed = datetime.fromisoformat(base_timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone().replace(tzinfo=None)
        return parsed
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"base_timestamp 형식이 잘못되었습니다: {base_timestamp}") from e


def clear_model_caches() -> None:
    # 모델 객체/성능표 캐시를 모두 비움
    store.clear_cache()


# ---------------------------------------------------------------------------
# 수동 예측 / 시뮬레이션 입력 생성
# ---------------------------------------------------------------------------
def predict_manual(device_id: str, rows: List[Dict[str, float]], power_in_kw: bool = False) -> Dict[str, Any]:
    # 수동 예측 공통 경로(raw row 직접 입력)
    runner = _get_runner(device_id)

    if not rows:
        raise HTTPException(status_code=400, detail="rows는 1개 이상이어야 합니다")

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise HTTPException(status_code=400, detail="rows가 비어 있습니다")

    if "DEVICE_ID" not in raw.columns:
        raw["DEVICE_ID"] = device_id
    if "LOG_DT" not in raw.columns:
        end = pd.Timestamp.now().floor("15min")
        raw["LOG_DT"] = pd.date_range(end=end, periods=len(raw), freq="15min")

    raw = add_current_model_aliases(raw)
    result = predict_from_raw_history(
        device_id=device_id,
        runner=runner,
        raw=raw,
        max_data_age_hours=24 * 365,
        enforce_freshness=False,
    )

    result = {
        "device_id": device_id,
        "best_model": runner.best_model,
        "preds": result["preds"],
        "missing_feature_count": runner.last_missing_count,
        "missing_features": runner.last_missing_features,
        "warnings": [],
    }
    if power_in_kw:
        return _convert_prediction_payload_to_kw(result)
    return result


def _load_simulation_raw(device_id: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    try:
        raw = fetch_pems_pro_log_df(start_dt=start_dt, end_dt=end_dt, device_ids=[device_id])
    except DataNotFoundError as e:
        raise HTTPException(status_code=404, detail="선택한 장비/기간에 데이터가 없습니다") from e

    if raw.empty:
        raise HTTPException(status_code=404, detail="선택한 장비/기간에 데이터가 없습니다")

    sort_cols = ["LOG_DT", "LOG_ID"] if "LOG_ID" in raw.columns else ["LOG_DT"]
    return add_current_model_aliases(raw.sort_values(sort_cols).copy())


def build_simulation_template(device_id: str, lookback_hours: int = 24) -> Dict[str, Any]:
    # 시뮬레이션 시작 시점에 필요한 기준 정보(기준행/기준예측/수정가능필드) 구성
    _validate_lookback_hours(lookback_hours)
    runner = _get_runner(device_id)

    base_dt = datetime.now()
    start_dt = base_dt - timedelta(hours=lookback_hours)

    raw = _load_simulation_raw(device_id=device_id, start_dt=start_dt, end_dt=base_dt)
    latest = raw.iloc[-1]

    editable_fields = resolve_editable_raw_fields(runner.feature_cols, list(raw.columns))

    values: Dict[str, Any] = {}
    for field in editable_fields:
        if field not in raw.columns:
            values[field] = None
            continue
        latest_value = latest[field]
        if pd.isna(latest_value):
            values[field] = None
            continue
        values[field] = float(latest_value)

    baseline = predict_one_device(
        device_id=device_id,
        lookback_hours=lookback_hours,
        max_data_age_hours=24,
        power_in_kw=True,
    )

    return {
        "device_id": device_id,
        "base_timestamp": pd.Timestamp(latest["LOG_DT"]).isoformat(),
        "base_log_id": (None if "LOG_ID" not in raw.columns else int(latest["LOG_ID"])),
        "editable_fields": _convert_power_fields_to_response_units(values),
        "baseline": baseline,
    }


def _resolve_simulation_target(raw: pd.DataFrame, base_log_id: Optional[int]) -> tuple[int, Any]:
    """시뮬레이션 기준 row 인덱스와 기준 시각을 결정한다."""
    if base_log_id is not None and "LOG_ID" in raw.columns:
        candidates = raw[raw["LOG_ID"] == int(base_log_id)]
        if candidates.empty:
            raise HTTPException(status_code=404, detail=f"시뮬레이션 기준 LOG_ID를 찾지 못했습니다: {base_log_id}")
        target_idx = candidates.index[-1]
    else:
        target_idx = raw.index[-1]

    target_ts = raw.loc[target_idx, "LOG_DT"]
    if pd.isna(target_ts):
        raise HTTPException(status_code=404, detail="시뮬레이션 기준 시점을 찾지 못했습니다")
    return int(target_idx), target_ts


def _validate_simulation_overrides(
    requested_overrides: Dict[str, float],
    editable_fields: List[str],
) -> Dict[str, float]:
    """허용된 override 컬럼만 남기고 잘못된 입력은 400으로 차단한다."""
    allowed_fields = set(editable_fields)
    invalid_fields = sorted(set(requested_overrides.keys()) - allowed_fields)
    if invalid_fields:
        raise HTTPException(status_code=400, detail=f"허용되지 않은 override 컬럼: {invalid_fields}")
    return requested_overrides


def _collect_effective_overrides(
    raw: pd.DataFrame,
    target_idx: int,
    requested_overrides: Dict[str, float],
) -> Dict[str, float]:
    """기준 row 대비 실제로 값이 바뀌는 override만 추린다."""
    effective_overrides: Dict[str, float] = {}
    for field, value in requested_overrides.items():
        if field not in raw.columns:
            continue
        current_value = raw.loc[target_idx, field]
        if pd.isna(current_value) or abs(float(current_value) - float(value)) > 1e-9:
            effective_overrides[field] = float(value)
    return effective_overrides


def _build_simulation_bin_mask(raw: pd.DataFrame, target_ts: Any) -> pd.Series:
    """기준 시각이 속한 리샘플 bin 전체를 찾고, 없으면 기준 row만 선택한다."""
    target_ts_norm = _to_naive_timestamp(target_ts)
    resample_rule = PreprocessConfig().resample_rule
    normalized_rule = re.sub(r"(?i)(\d+)t$", r"\1min", str(resample_rule or "15min"))
    bin_start = target_ts_norm.floor(normalized_rule)
    bin_end = bin_start + pd.to_timedelta(normalized_rule)
    bin_mask = (
        (pd.to_datetime(raw["LOG_DT"], errors="coerce") >= bin_start)
        & (pd.to_datetime(raw["LOG_DT"], errors="coerce") < bin_end)
    )
    return bin_mask


def _apply_overrides_to_bin(raw: pd.DataFrame, bin_mask: pd.Series, overrides: Dict[str, float]) -> None:
    """선택된 리샘플 bin 전체에 override 값을 반영한다."""
    for field, value in overrides.items():
        raw.loc[bin_mask, field] = value


def _predict_simulation_pair(
    device_id: str,
    runner,
    baseline_raw: pd.DataFrame,
    simulated_raw: pd.DataFrame,
    target_ts: Any,
    effective_overrides: Dict[str, float],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """같은 기준 시각으로 baseline과 simulated 예측을 함께 계산한다."""
    baseline = predict_from_raw_history(
        device_id=device_id,
        runner=runner,
        raw=baseline_raw,
        max_data_age_hours=24,
        enforce_freshness=False,
        reference_timestamp=target_ts,
    )
    if not effective_overrides:
        return baseline, baseline

    simulated = predict_from_raw_history(
        device_id=device_id,
        runner=runner,
        raw=simulated_raw,
        max_data_age_hours=24,
        enforce_freshness=False,
        reference_timestamp=target_ts,
    )
    return baseline, simulated


def _extract_prediction_values(pred_result: Dict[str, Any]) -> tuple[float, float]:
    """예측 응답에서 15분/30분 전력값을 안전하게 꺼낸다."""
    pred = pred_result["preds"][0] if pred_result.get("preds") else {}
    return float(pred.get("y_15_pred", 0.0)), float(pred.get("y_30_pred", 0.0))


def _build_input_influence(
    device_id: str,
    runner,
    baseline_raw: pd.DataFrame,
    bin_mask: pd.Series,
    effective_overrides: Dict[str, float],
    target_ts: Any,
    y15_base: float,
    y30_base: float,
) -> Dict[str, float]:
    """각 override 컬럼이 예측에 미치는 상대 영향도를 계산한다."""
    input_influence: Dict[str, float] = {}
    for field, value in effective_overrides.items():
        try:
            single_raw = baseline_raw.copy()
            single_raw.loc[bin_mask, field] = value
            single_pred = predict_from_raw_history(
                device_id=device_id,
                runner=runner,
                raw=single_raw,
                max_data_age_hours=24,
                enforce_freshness=False,
                reference_timestamp=target_ts,
            )
            y15, y30 = _extract_prediction_values(single_pred)
            pct15 = 0.0 if y15_base == 0 else (y15 - y15_base) / abs(y15_base) * 100.0
            pct30 = 0.0 if y30_base == 0 else (y30 - y30_base) / abs(y30_base) * 100.0
            input_influence[field] = round(max(abs(pct15), abs(pct30)), 4)
        except Exception:
            input_influence[field] = 0.0
    return input_influence


def _build_simulation_result_value(y15_base: float, y30_base: float, y15_sim: float, y30_sim: float) -> Dict[str, Any]:
    """DB 저장용 baseline/simulated/delta JSON 구조를 만든다."""
    return {
        "baseline": {"y_15_pred": y15_base, "y_30_pred": y30_base},
        "simulated": {"y_15_pred": y15_sim, "y_30_pred": y30_sim},
        "delta": {
            "y_15_pred": y15_sim - y15_base,
            "y_30_pred": y30_sim - y30_base,
            "y_15_pct": (0.0 if y15_base == 0 else ((y15_sim - y15_base) / abs(y15_base) * 100.0)),
            "y_30_pct": (0.0 if y30_base == 0 else ((y30_sim - y30_base) / abs(y30_base) * 100.0)),
        },
    }


def _build_change_column_info(
    baseline_raw: pd.DataFrame,
    target_idx: int,
    effective_overrides: Dict[str, float],
) -> Dict[str, Any]:
    """변경된 입력 컬럼의 before/after/delta 정보를 저장용 구조로 만든다."""
    change_rows: List[Dict[str, Any]] = []
    for field, after_value in effective_overrides.items():
        before_raw = baseline_raw.loc[target_idx, field] if field in baseline_raw.columns else None
        before_value = None if pd.isna(before_raw) else float(before_raw)
        after_num = float(after_value)
        delta_value = None if before_value is None else (after_num - before_value)
        delta_pct = None if before_value is None or before_value == 0 else ((after_num - before_value) / abs(before_value) * 100.0)
        change_rows.append(
            {
                "feature_name": field,
                "before_value": before_value,
                "after_value": after_num,
                "delta_value": delta_value,
                "delta_pct": delta_pct,
            }
        )
    return {"changes": change_rows}


def _save_simulation_log(
    device_id: str,
    target_ts: Any,
    lookback_hours: int,
    runner,
    editable_fields: List[str],
    effective_overrides: Dict[str, float],
    result_value: Dict[str, Any],
    change_column_info: Dict[str, Any],
) -> None:
    """실제 변경이 있을 때만 시뮬레이션 결과 로그를 저장한다."""
    if len(effective_overrides) == 0:
        logger.info(f"[AI 시뮬레이션] 저장 스킵(변경값 없음) device_id={device_id}, lookback_hours={lookback_hours}")
        return

    feature_importance = {
        "model_name": str(runner.best_model),
        "importance_method": "not_supported",
        "features": [],
    }

    db_gen = db_connection_pool()
    db = next(db_gen)
    try:
        BaseQueryFactory(db, TB_SIMULATION_LOG).insert_single_row(
            device_id=device_id,
            baseline_pd_time=pd.Timestamp(target_ts).to_pydatetime(),
            search_time=int(lookback_hours),
            use_model=str(runner.best_model),
            available_feature=len(editable_fields),
            change_feature=len(effective_overrides),
            result_value=result_value,
            change_column_info=change_column_info,
            feature_importance=feature_importance,
        )
    except Exception as e:
        logger.error(f"[AI 시뮬레이션] 결과 저장 실패 device_id={device_id}: {e}")
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


def _prepare_simulation_inputs(
    device_id: str,
    lookback_hours: int,
    base_timestamp: str,
    base_log_id: Optional[int],
):
    """시뮬레이션 실행에 필요한 runner/raw/기준 row/override 범위를 한 번에 준비한다."""
    base_dt = parse_base_timestamp(base_timestamp)
    start_dt = base_dt - timedelta(hours=lookback_hours)
    runner = _get_runner(device_id)
    raw = _load_simulation_raw(device_id=device_id, start_dt=start_dt, end_dt=base_dt)
    baseline_raw = raw.copy()
    editable_fields = resolve_editable_raw_fields(runner.feature_cols, list(raw.columns))
    target_idx, target_ts = _resolve_simulation_target(raw, base_log_id)
    return runner, raw, baseline_raw, editable_fields, target_idx, target_ts


def run_simulation(
    device_id: str,
    overrides: Dict[str, float],
    lookback_hours: int,
    base_timestamp: str,
    base_log_id: Optional[int] = None,
    save_log: bool = True,
) -> Dict[str, Any]:
    """기준 row override를 반영해 baseline과 simulated 예측을 비교한다."""
    _validate_lookback_hours(lookback_hours)
    runner, raw, baseline_raw, editable_fields, target_idx, target_ts = _prepare_simulation_inputs(
        device_id=device_id,
        lookback_hours=lookback_hours,
        base_timestamp=base_timestamp,
        base_log_id=base_log_id,
    )
    requested_overrides = _validate_simulation_overrides(overrides or {}, editable_fields)
    requested_overrides = _convert_power_fields_to_internal_units(requested_overrides)
    effective_overrides = _collect_effective_overrides(raw, target_idx, requested_overrides)
    bin_mask = _build_simulation_bin_mask(raw, target_ts)
    if not bool(bin_mask.any()):
        bin_mask = (raw.index == target_idx)
    _apply_overrides_to_bin(raw, bin_mask, effective_overrides)

    baseline, simulated = _predict_simulation_pair(
        device_id=device_id,
        runner=runner,
        baseline_raw=baseline_raw,
        simulated_raw=raw,
        target_ts=target_ts,
        effective_overrides=effective_overrides,
    )
    y15_base, y30_base = _extract_prediction_values(baseline)
    y15_sim, y30_sim = _extract_prediction_values(simulated)

    input_influence = _build_input_influence(
        device_id=device_id,
        runner=runner,
        baseline_raw=baseline_raw,
        bin_mask=bin_mask,
        effective_overrides=effective_overrides,
        target_ts=target_ts,
        y15_base=y15_base,
        y30_base=y30_base,
    )
    result_value = _build_simulation_result_value(y15_base, y30_base, y15_sim, y30_sim)
    change_column_info = _build_change_column_info(baseline_raw, target_idx, effective_overrides)
    if save_log:
        _save_simulation_log(
            device_id=device_id,
            target_ts=target_ts,
            lookback_hours=lookback_hours,
            runner=runner,
            editable_fields=editable_fields,
            effective_overrides=effective_overrides,
            result_value=result_value,
            change_column_info=change_column_info,
        )

    return {
        "device_id": device_id,
        "base_timestamp": pd.Timestamp(target_ts).isoformat(),
        "overrides": _convert_power_fields_to_response_units(effective_overrides),
        "baseline": _convert_prediction_payload_to_kw(baseline),
        "simulated": _convert_prediction_payload_to_kw(simulated),
        "delta": _convert_simulation_delta_to_kw(result_value["delta"]),
        "input_influence": input_influence,
    }
