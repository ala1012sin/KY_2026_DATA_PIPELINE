"""예측/시뮬레이션 공통 비즈니스 로직 서비스."""

import ast
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
from service.model_store import ModelStore
from service.processing.config import PreprocessConfig
from service.processing.pipeline import (
    DataNotFoundError,
    fetch_pems_pro_log_df,
    preprocess_pems_pro_from_db_in_memory,
    preprocess_raw_df_to_supervised,
)

MODEL_ROOT = str(Path(os.environ.get("MODEL_ROOT", "./ai_models/current")).expanduser().resolve())
store = ModelStore(MODEL_ROOT)

_FE_SUFFIX_PATTERNS = (
    r"_lag\d+$",
    r"_diff\d+$",
    r"_pct\d+$",
    r"_roll\d+_(mean|std)$",
)


def _to_base_feature_name(feature_name: str) -> str:
    # lag/rolling/diff/pct 파생 suffix를 제거해 원본 컬럼명을 추출
    base = str(feature_name)
    for pattern in _FE_SUFFIX_PATTERNS:
        base = re.sub(pattern, "", base)
    return base


def _expand_feature_name_tokens(feature_name: str) -> List[str]:
    # "('PRESSURE', 'LOG_ID')" 같은 tuple-string 피처도 raw 컬럼 후보로 펼친다.
    base = _to_base_feature_name(feature_name)
    tokens = {base}

    try:
        parsed = ast.literal_eval(base)
    except Exception:
        parsed = None

    if isinstance(parsed, (tuple, list)):
        for item in parsed:
            token = str(item).strip()
            if token:
                tokens.add(token)

    return [token for token in tokens if token]


def resolve_editable_raw_fields(feature_cols: List[str], raw_columns: List[str]) -> List[str]:
    # 직접 제어 의미가 낮은 식별/시간/타깃성 컬럼은 시뮬 입력에서 제외
    raw_excludes = {
        "LOG_ID", "DEVICE_ID", "LOG_DT",
        "CURVOLTAGE", "CUR_VOLTAGE",
        "AVG_VOLTAGE", "AVG_CURRENT",
        "OP_STATUS", "CSUSAGETIME", "MGREFILLTIME",
    }
    base_feature_names = {
        token
        for col in feature_cols
        for token in _expand_feature_name_tokens(col)
    }
    return [
        col for col in raw_columns
        if col not in raw_excludes and col in base_feature_names
    ]


def list_model_device_ids() -> List[str]:
    # 모델 저장소(device=...) 디렉터리 기준으로 장비 목록 구성
    if not os.path.isdir(MODEL_ROOT):
        return []

    device_ids: List[str] = []
    for name in os.listdir(MODEL_ROOT):
        full_path = os.path.join(MODEL_ROOT, name)
        if os.path.isdir(full_path) and name.startswith("device="):
            device_id = name.split("device=", 1)[-1].strip()
            if device_id:
                device_ids.append(device_id)

    return sorted(set(device_ids))


def predict_one_device(device_id: str, lookback_hours: int, max_data_age_hours: int) -> Dict[str, Any]:
    # 자동 예측 공통 경로(조회→전처리→예측)
    if lookback_hours <= 0:
        raise HTTPException(status_code=400, detail="lookback_hours는 1 이상이어야 합니다")
    if lookback_hours > 24 * 31:
        raise HTTPException(status_code=400, detail="lookback_hours가 너무 큽니다(최대 744시간)")
    if max_data_age_hours <= 0:
        raise HTTPException(status_code=400, detail="max_data_age_hours는 1 이상이어야 합니다")

    try:
        runner = store.get_runner(device_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(hours=lookback_hours)

    pcfg = PreprocessConfig()
    meta = preprocess_pems_pro_from_db_in_memory(
        start_dt=start_dt,
        end_dt=end_dt,
        pcfg=pcfg,
        device_ids=[device_id],
    )

    return predict_from_preprocessed(
        device_id=device_id,
        runner=runner,
        meta=meta,
        max_data_age_hours=max_data_age_hours,
        enforce_freshness=True,
    )


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

    base_timestamp = df.iloc[-1][time_col]
    now_ts = pd.Timestamp.now().tz_localize(None)
    base_ts = pd.Timestamp(base_timestamp)
    if base_ts.tzinfo is not None:
        base_ts = base_ts.tz_convert("UTC").tz_localize(None)

    if enforce_freshness:
        age_hours = float((now_ts - base_ts).total_seconds() / 3600.0)
        if age_hours > max_data_age_hours:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"최신 데이터가 너무 오래되었습니다. "
                    f"기준시각={base_ts.isoformat()}, 경과={age_hours:.1f}h, 허용={max_data_age_hours}h"
                ),
            )

    required_rows = runner.dl_seq_len if runner.model_type == "DL" else 1
    if required_rows is None:
        required_rows = 1

    if len(df) < required_rows:
        raise HTTPException(
            status_code=400,
            detail=f"예측에 필요한 전처리 행이 부족합니다. 필요={required_rows}, 현재={len(df)}",
        )

    selected = df
    if reference_timestamp is not None:
        ref_ts = pd.Timestamp(reference_timestamp)
        if ref_ts.tzinfo is not None:
            ref_ts = ref_ts.tz_convert("UTC").tz_localize(None)

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

    selected_ts = pd.Timestamp(rows_df.iloc[-1][time_col])
    if selected_ts.tzinfo is not None:
        selected_ts = selected_ts.tz_convert("UTC").tz_localize(None)

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


def predict_manual(device_id: str, rows: List[Dict[str, float]]) -> Dict[str, Any]:
    # 수동 예측 공통 경로(feature row 직접 입력)
    try:
        runner = store.get_runner(device_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    preds, warns = runner.predict(rows)
    return {
        "device_id": device_id,
        "best_model": runner.best_model,
        "preds": preds,
        "missing_feature_count": runner.last_missing_count,
        "missing_features": runner.last_missing_features,
        "warnings": warns,
    }


def _load_simulation_raw(device_id: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    try:
        raw = fetch_pems_pro_log_df(start_dt=start_dt, end_dt=end_dt, device_ids=[device_id])
    except DataNotFoundError as e:
        raise HTTPException(status_code=404, detail="선택한 장비/기간에 데이터가 없습니다") from e

    if raw.empty:
        raise HTTPException(status_code=404, detail="선택한 장비/기간에 데이터가 없습니다")

    sort_cols = ["LOG_DT", "LOG_ID"] if "LOG_ID" in raw.columns else ["LOG_DT"]
    return raw.sort_values(sort_cols).copy()


def build_simulation_template(device_id: str, lookback_hours: int = 24) -> Dict[str, Any]:
    # 시뮬레이션 시작 시점에 필요한 기준 정보(기준행/기준예측/수정가능필드) 구성
    if lookback_hours <= 0:
        raise HTTPException(status_code=400, detail="lookback_hours는 1 이상이어야 합니다")

    try:
        runner = store.get_runner(device_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

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

    baseline = predict_one_device(device_id=device_id, lookback_hours=lookback_hours, max_data_age_hours=24)

    return {
        "device_id": device_id,
        "base_timestamp": pd.Timestamp(latest["LOG_DT"]).isoformat(),
        "base_log_id": (None if "LOG_ID" not in raw.columns else int(latest["LOG_ID"])),
        "editable_fields": values,
        "baseline": baseline,
    }


def run_simulation(
    device_id: str,
    overrides: Dict[str, float],
    lookback_hours: int,
    base_timestamp: str,
    base_log_id: Optional[int] = None,
    save_log: bool = True,
) -> Dict[str, Any]:
    # 기준행 기반 override를 적용하고 baseline/simulated를 같은 조건으로 비교
    if lookback_hours <= 0:
        raise HTTPException(status_code=400, detail="lookback_hours는 1 이상이어야 합니다")

    base_dt = parse_base_timestamp(base_timestamp)
    start_dt = base_dt - timedelta(hours=lookback_hours)

    try:
        runner = store.get_runner(device_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    raw = _load_simulation_raw(device_id=device_id, start_dt=start_dt, end_dt=base_dt)
    baseline_raw = raw.copy()

    editable_fields = resolve_editable_raw_fields(runner.feature_cols, list(raw.columns))
    allowed_fields = set(editable_fields)
    requested_overrides = overrides or {}
    invalid_fields = sorted(set(requested_overrides.keys()) - allowed_fields)
    if invalid_fields:
        raise HTTPException(status_code=400, detail=f"허용되지 않은 override 컬럼: {invalid_fields}")

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

    effective_overrides: Dict[str, float] = {}
    for field, value in requested_overrides.items():
        if field not in raw.columns:
            continue
        current_value = raw.loc[target_idx, field]
        if pd.isna(current_value):
            effective_overrides[field] = float(value)
            continue
        if abs(float(current_value) - float(value)) > 1e-9:
            effective_overrides[field] = float(value)

    # 리샘플(mean) 기반 입력에서 단일 raw 행만 바꾸면 변화가 매우 작을 수 있어,
    # 선택 기준행이 속한 리샘플 bin 전체에 override를 적용한다.
    target_ts_norm = pd.Timestamp(target_ts)
    if target_ts_norm.tzinfo is not None:
        target_ts_norm = target_ts_norm.tz_convert("UTC").tz_localize(None)

    resample_rule = PreprocessConfig().resample_rule
    normalized_rule = re.sub(r"(?i)(\d+)t$", r"\1min", str(resample_rule or "15min"))
    bin_start = target_ts_norm.floor(normalized_rule)
    bin_step = pd.to_timedelta(normalized_rule)
    bin_end = bin_start + bin_step
    bin_mask = (pd.to_datetime(raw["LOG_DT"], errors="coerce") >= bin_start) & (pd.to_datetime(raw["LOG_DT"], errors="coerce") < bin_end)

    # 안전장치: bin 마스크가 비었으면 기존처럼 target_idx 단일 행에 적용
    if not bool(bin_mask.any()):
        bin_mask = (raw.index == target_idx)

    for field, value in effective_overrides.items():
        raw.loc[bin_mask, field] = value

    pcfg = PreprocessConfig()
    baseline_meta = preprocess_raw_df_to_supervised(raw=baseline_raw, pcfg=pcfg, persist_outputs=False)
    baseline = predict_from_preprocessed(
        device_id=device_id,
        runner=runner,
        meta=baseline_meta,
        max_data_age_hours=24,
        enforce_freshness=False,
        reference_timestamp=target_ts,
    )

    if effective_overrides:
        simulated_meta = preprocess_raw_df_to_supervised(raw=raw, pcfg=pcfg, persist_outputs=False)
        simulated = predict_from_preprocessed(
            device_id=device_id,
            runner=runner,
            meta=simulated_meta,
            max_data_age_hours=24,
            enforce_freshness=False,
            reference_timestamp=target_ts,
        )
    else:
        simulated = baseline

    base_pred = baseline["preds"][0] if baseline.get("preds") else {}
    sim_pred = simulated["preds"][0] if simulated.get("preds") else {}

    y15_base = float(base_pred.get("y_15_pred", 0.0))
    y30_base = float(base_pred.get("y_30_pred", 0.0))
    y15_sim = float(sim_pred.get("y_15_pred", 0.0))
    y30_sim = float(sim_pred.get("y_30_pred", 0.0))

    # DB RESULT_VALUE(JSONB) 스키마에 맞춰 baseline/simulated/delta를 구성
    result_value = {
        "baseline": {
            "y_15_pred": y15_base,
            "y_30_pred": y30_base,
        },
        "simulated": {
            "y_15_pred": y15_sim,
            "y_30_pred": y30_sim,
        },
        "delta": {
            "y_15_pred": y15_sim - y15_base,
            "y_30_pred": y30_sim - y30_base,
            "y_15_pct": (0.0 if y15_base == 0 else ((y15_sim - y15_base) / abs(y15_base) * 100.0)),
            "y_30_pct": (0.0 if y30_base == 0 else ((y30_sim - y30_base) / abs(y30_base) * 100.0)),
        },
    }

    # 변경된 입력 컬럼별 before/after/증감 정보를 JSON 배열로 저장
    change_rows: List[Dict[str, Any]] = []
    for field, after_value in effective_overrides.items():
        before_raw = baseline_raw.loc[target_idx, field] if field in baseline_raw.columns else None
        before_value = None if pd.isna(before_raw) else float(before_raw)
        after_num = float(after_value)
        delta_value = None if before_value is None else (after_num - before_value)
        delta_pct = (
            None
            if before_value is None or before_value == 0
            else ((after_num - before_value) / abs(before_value) * 100.0)
        )
        change_rows.append({
            "feature_name": field,
            "before_value": before_value,
            "after_value": after_num,
            "delta_value": delta_value,
            "delta_pct": delta_pct,
        })

    change_column_info = {"changes": change_rows}

    # XGB 모델 사용 시 gain 기반 피처 중요도를 기록(기타 모델은 빈 배열 유지)
    feature_importance = {
        "model_name": str(runner.best_model),
        "importance_method": "gain",
        "features": [],
    }
    if str(runner.model_type).upper() == "XGB":
        try:
            booster = runner.model_obj["15"].get_booster()
            raw_scores = booster.get_score(importance_type="gain")
            scored_features: List[Dict[str, Any]] = []
            for key, value in raw_scores.items():
                feature_name = key
                if isinstance(key, str) and key.startswith("f") and key[1:].isdigit():
                    idx = int(key[1:])
                    if 0 <= idx < len(runner.feature_cols):
                        feature_name = runner.feature_cols[idx]
                scored_features.append({
                    "feature_name": feature_name,
                    "importance_value": float(value),
                })

            scored_features = sorted(scored_features, key=lambda item: item["importance_value"], reverse=True)
            feature_importance["features"] = [
                {
                    "rank": rank,
                    "feature_name": item["feature_name"],
                    "importance_value": item["importance_value"],
                }
                for rank, item in enumerate(scored_features, start=1)
            ]
        except Exception as e:
            logger.warning(f"[AI 시뮬레이션] feature importance 추출 실패: {e}")

    # 저장 요청(save_log=true)이고 실제 변경값이 있을 때만 시뮬 로그를 적재
    if save_log and len(effective_overrides) > 0:
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
    # 저장 요청은 있었지만 변경값이 없으면 로그 적재를 생략
    elif save_log:
        logger.info(
            f"[AI 시뮬레이션] 저장 스킵(변경값 없음) device_id={device_id}, lookback_hours={lookback_hours}"
        )

    return {
        "device_id": device_id,
        "base_timestamp": pd.Timestamp(target_ts).isoformat(),
        "overrides": effective_overrides,
        "baseline": baseline,
        "simulated": simulated,
        "delta": result_value["delta"],
    }
