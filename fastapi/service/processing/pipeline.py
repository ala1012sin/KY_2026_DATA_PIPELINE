"""
전처리 파이프라인 진입점 모듈.

주요 흐름:
1) TB_PEMS_PRO_LOG 조회
2) DataFrame 기반 전처리로 supervised 데이터 생성
"""
import json
import os
import uuid
from datetime import datetime
from dataclasses import asdict
from typing import Any, Dict, Optional, Sequence

import pandas as pd
from sqlalchemy.orm import Session

from db.public.models import TB_PEMS_PRO_LOG
from setting.database_orm import SessionLocal
from .config import PreprocessConfig
from .steps import (
    add_cumulative_delta_features,
    add_feature_engineering,
    add_gap_features,
    add_load_factor_feature,
    add_session_features,
    add_splits_per_device,
    add_splits_per_device_by_days,
    add_targets,
    add_time_features,
    fill_feature_nans_devicewise,
    infer_col,
    infer_target_col_robust,
    resample_15m_per_device,
)


class DataNotFoundError(ValueError):
    """조회/전처리 대상 데이터가 없는 경우 예외."""


def fetch_pems_pro_log_df(
    start_dt: datetime,
    end_dt: datetime,
    device_ids: Optional[Sequence[str]] = None,
    db: Optional[Session] = None,
) -> pd.DataFrame:
    """TB_PEMS_PRO_LOG를 조회해 학습/서빙 공통 스키마 DataFrame으로 반환한다."""
    # 세션을 외부에서 주입하면 재사용하고, 없으면 함수 내부에서 생성/종료
    should_close = False
    session = db
    if session is None:
        session = SessionLocal()
        should_close = True

    try:
        query = session.query(TB_PEMS_PRO_LOG).filter(
            TB_PEMS_PRO_LOG.log_dt >= start_dt,
            TB_PEMS_PRO_LOG.log_dt <= end_dt,
        )
        if device_ids:
            device_uuid_list = []
            for device_id in device_ids:
                try:
                    device_uuid_list.append(uuid.UUID(str(device_id)))
                except ValueError as e:
                    raise ValueError(f"invalid device_id format: {device_id}") from e
            query = query.filter(TB_PEMS_PRO_LOG.device_id.in_(device_uuid_list))

        # 시간 오름차순으로 정렬해 시계열 전처리 안정성 확보
        rows = query.order_by(TB_PEMS_PRO_LOG.log_dt.asc()).all()
        if not rows:
            raise DataNotFoundError("선택한 장비/기간에 TB_PEMS_PRO_LOG 데이터가 없습니다")

        records = [
            {
                "LOG_ID": row.log_id,
                "DEVICE_ID": str(row.device_id),
                "LOG_DT": row.log_dt,
                "PRESSURE": row.pressure,
                "TEMPERATURE": row.temperature,
                "HZ": row.hz,
                "OP_STATUS": row.op_status,
                "AVGVOLTAGE": row.avg_voltage,
                "AVGCURRENT": row.avg_current,
                "CURVOLTAGE": row.cur_voltage,
                # 학습 코드에서 underscore 컬럼명을 사용한 모델과 서빙 컬럼명을 맞춘다.
                "AVG_VOLTAGE": row.avg_voltage,
                "AVG_CURRENT": row.avg_current,
                "CUR_VOLTAGE": row.cur_voltage,
                "FACTOR": row.factor,
                "OP_TIME": row.op_time,
                "CSUSAGETIME": row.cs_usage_time,
                "MGREFILLTIME": row.mg_refill_time,
            }
            for row in rows
        ]
        return pd.DataFrame.from_records(records)
    except Exception:
        if session is not None:
            session.rollback()
        raise
    finally:
        if should_close and session is not None:
            session.close()


def preprocess_pems_pro_from_db_in_memory(
    start_dt: datetime,
    end_dt: datetime,
    pcfg: PreprocessConfig,
    device_ids: Optional[Sequence[str]] = None,
    db: Optional[Session] = None,
) -> Dict[str, Any]:
    """DB 조회부터 전처리까지 메모리에서 수행한다(CSV 파일 미생성)."""
    # 예측 API 기본 경로: 디스크 I/O 없이 DataFrame으로 바로 전처리
    raw = fetch_pems_pro_log_df(
        start_dt=start_dt,
        end_dt=end_dt,
        device_ids=device_ids,
        db=db,
    )
    return preprocess_raw_df_to_supervised(raw=raw, pcfg=pcfg, persist_outputs=False)


def preprocess_raw_df_to_supervised(
    raw: pd.DataFrame,
    pcfg: PreprocessConfig,
    raw_csv_path: Optional[str] = None,
    persist_outputs: bool = False,
) -> Dict[str, Any]:
    """원본 DataFrame을 supervised 전처리 데이터로 변환한다."""
    if raw is None or raw.empty:
        raise DataNotFoundError("원본 데이터프레임이 비어 있습니다")

    # 필수 핵심 컬럼 추론(시간/장비/타깃)
    time_col = infer_col(raw, pcfg.time_col_candidates, required=True)
    device_col = infer_col(raw, pcfg.device_col_candidates, required=True)
    target_col_raw = infer_target_col_robust(raw, pcfg.target_col_candidates, required=True)

    raw = raw.copy()
    raw[time_col] = pd.to_datetime(raw[time_col], errors="coerce")
    raw = raw.dropna(subset=[time_col])
    if pcfg.cutoff_datetime:
        raw = raw[raw[time_col] >= pcfg.cutoff_datetime].copy()
    if raw.empty:
        raise DataNotFoundError("cutoff 적용 후 사용 가능한 원본 데이터가 없습니다")

    # 1) 15분 리샘플 + 관측 개수(n_obs) 생성
    df_15 = resample_15m_per_device(
        raw, time_col, device_col,
        pcfg.resample_rule, pcfg.fill_method, fill_limit=pcfg.fill_limit
    )
    target_col = infer_target_col_robust(df_15, pcfg.target_col_candidates, required=False)
    if target_col is None:
        target_col = target_col_raw if target_col_raw in df_15.columns else infer_target_col_robust(df_15, (target_col_raw,), required=True)

    # 2) 관측 간 간격 관련 피처
    df_15 = add_gap_features(df_15, time_col, device_col, rule=pcfg.resample_rule, n_obs_col="n_obs")

    # 3) 관측이 있는 구간만 유지
    df_15 = df_15[df_15["n_obs"].fillna(0) > 0].copy()
    if df_15.empty:
        raise DataNotFoundError("리샘플링/필터링 이후 사용 가능한 관측치가 없습니다")

    # 4) 세션(연속 구간) 피처 생성
    df_15 = add_session_features(df_15, time_col, device_col, rule=pcfg.resample_rule, gap_col="gap_prev_obs_bins", break_gap_bins=2)

    # 5) 시간 피처(요일/업무시간 등)
    if pcfg.add_time_features:
        df_15 = add_time_features(
            df_15,
            time_col,
            pcfg.business_hour_start,
            pcfg.business_hour_end,
            add_business_hour=pcfg.add_business_hour_flag,
        )

    if pcfg.use_date_split:
        df_15 = add_splits_per_device_by_days(
            df_15,
            time_col,
            device_col,
            test_days=pcfg.test_days,
            val_days=pcfg.val_days,
            min_train_days=pcfg.min_train_days,
            fallback_val_ratio=pcfg.val_ratio,
            fallback_test_ratio=pcfg.test_ratio,
        )
    else:
        df_15 = add_splits_per_device(df_15, time_col, device_col, pcfg.val_ratio, pcfg.test_ratio)

    df_15 = add_load_factor_feature(df_15, device_col, target_col, split_col="split", q=0.99)
    df_15 = add_cumulative_delta_features(df_15, device_col, time_col)

    # FE 제외 대상(메타/시간 피쳐)
    fe_ban = {
        "n_obs", "gap_prev_obs_bins",
        "hour", "dayofweek", "is_weekend", "is_business_day", "is_business_hour"
    }

    # 기본 FE 후보: target + 주요 센서 컬럼
    default_fe = [
        target_col,
        "HZ",
        "AVGVOLTAGE",
        "AVGCURRENT",
        "PRESSURE",
        "TEMPERATURE",
        "FACTOR",
        "load_factor",
        "CSUSAGETIME_DELTA",
        "MGREFILLTIME_DELTA",
    ]
    default_fe = [c for c in default_fe if (c in df_15.columns) and (c not in fe_ban)]

    # 사용자가 명시한 FE 컬럼 우선
    if pcfg.fe_cols is not None:
        base_cols = [target_col] + [c for c in pcfg.fe_cols if (c in df_15.columns) and (c not in fe_ban) and (c != target_col)]
        # 혹시 target_col이 ban에 걸렸을 가능성 방지
        base_cols = [c for c in base_cols if c in df_15.columns]
        if target_col not in base_cols and target_col in df_15.columns:
            base_cols = [target_col] + base_cols
    else:
        base_cols = default_fe

    # 최소 보장: target 1개는 반드시 포함
    if not base_cols:
        base_cols = [target_col]

    # 6) 시계열 파생피처(lag/rolling/diff/pct) 생성
    df_feat = add_feature_engineering(
        df_15, time_col, device_col,
        base_cols=base_cols,
        lag_steps=pcfg.lag_steps,
        roll_windows=pcfg.roll_windows,
        diff_steps=pcfg.diff_steps,
        add_pct_change=pcfg.add_pct_change,
        pct_eps=pcfg.pct_eps,
        add_roll_std=pcfg.add_roll_std,
        session_col="session_id",  # 세션 단위로 lag/rolling/diff/pct 생성
        split_col="split",
    )

    # 7) 타깃/결측 정리
    df_sup = add_targets(
        df_feat,
        time_col,
        device_col,
        target_col,
        pcfg.horizons_steps,
        session_col="session_id",
        split_col="split",
    )
    if pcfg.lag_steps:
        max_lag = max(pcfg.lag_steps)
        max_lag_col = f"{target_col}_lag{max_lag}"
        if max_lag_col in df_sup.columns:
            df_sup = df_sup.dropna(subset=[max_lag_col]).copy()

    df_sup = fill_feature_nans_devicewise(
        df_sup,
        time_col,
        device_col,
        add_missing_flags=pcfg.add_missing_flags,
        base_fill_method=pcfg.base_fill_method_after_fe,
        base_fill_limit=pcfg.base_fill_limit_after_fe,
        engineered_fill_value=pcfg.engineered_fill_value,
    )
    if df_sup.empty:
        raise DataNotFoundError("전처리 결과 데이터가 비어 있습니다")

    # persist_outputs=True일 때만 파일을 남기고, 기본은 메모리 전용 처리
    out_csv = None
    if persist_outputs:
        os.makedirs(pcfg.out_dir, exist_ok=True)
        out_csv = os.path.join(pcfg.out_dir, "preprocessed_supervised.csv")
        df_sup.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 추적/디버깅용 메타 저장
    meta = {
        "raw_csv_path": raw_csv_path,
        "out_csv": out_csv,
        "time_col": time_col,
        "device_col": device_col,
        "target_col_raw": target_col_raw,
        "target_col": target_col,
        "preprocess_config": asdict(pcfg),
        "n_rows": int(len(df_sup)),
        "n_devices": int(df_sup[device_col].nunique()),
        #디버깅/보고용: 어떤 컬럼에 FE 적용했는지 기록
        "fe_base_cols_used": base_cols,
        "df_sup": df_sup,  # 호출자가 바로 예측에 사용할 in-memory 결과
    }
    if persist_outputs:
        meta_to_save = {
            "raw_csv_path": raw_csv_path,
            "out_csv": out_csv,
            "time_col": time_col,
            "device_col": device_col,
            "target_col_raw": target_col_raw,
            "target_col": target_col,
            "preprocess_config": asdict(pcfg),
            "n_rows": int(len(df_sup)),
            "n_devices": int(df_sup[device_col].nunique()),
            "fe_base_cols_used": base_cols,
        }
        with open(os.path.join(pcfg.out_dir, "preprocess_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_to_save, f, ensure_ascii=False, indent=2)

    return meta
