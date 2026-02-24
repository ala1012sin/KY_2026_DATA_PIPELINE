"""예측 관련 API 라우터.

- 수동 예측: feature row를 직접 받아 예측
- 자동 예측: device_id만 받아 DB 조회 -> 전처리 -> 예측까지 수행
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List

from Logger import Logger as logger
from service.model_store import ModelStore
from service.processing.config import PreprocessConfig
from service.processing.pipeline import DataNotFoundError, preprocess_pems_pro_from_db_in_memory

# 모델 저장 루트 경로와 러너 스토어 초기화
MODEL_ROOT = str(Path(os.environ.get("MODEL_ROOT", "./models/current")).expanduser().resolve())
store = ModelStore(MODEL_ROOT)

router = APIRouter()


class PredictRequest(BaseModel):
    """수동 예측 요청 스키마."""
    device_id: str
    rows: List[Dict[str, float]] = Field(..., description="feature dict 리스트")


class MultiPredictRequest(BaseModel):
    """여러 장비 자동 예측 요청 스키마."""
    device_ids: List[str] = Field(..., description="예측할 장비 ID 목록")
    lookback_hours: int = Field(24, description="조회 기간(시간)")
    max_data_age_hours: int = Field(24, description="허용 데이터 신선도(시간)")


def _predict_one_device(device_id: str, lookback_hours: int, max_data_age_hours: int) -> Dict[str, Any]:
    """장비 1대에 대한 자동 예측 내부 로직."""
    # 공통 유효성 검증: 단건/배치 모두 이 함수를 재사용한다.
    if lookback_hours <= 0:
        raise HTTPException(status_code=400, detail="lookback_hours는 1 이상이어야 합니다")
    if lookback_hours > 24 * 31:
        raise HTTPException(status_code=400, detail="lookback_hours가 너무 큽니다(최대 744시간)")
    if max_data_age_hours <= 0:
        raise HTTPException(status_code=400, detail="max_data_age_hours는 1 이상이어야 합니다")

    runner = store.get_runner(device_id)

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(hours=lookback_hours)

    # 메모리 기반 전처리 경로 사용(중간 CSV 파일 생성 없음)
    pcfg = PreprocessConfig()
    meta = preprocess_pems_pro_from_db_in_memory(
        start_dt=start_dt,
        end_dt=end_dt,
        pcfg=pcfg,
        device_ids=[device_id],
    )

    # 전처리 결과 DataFrame을 메타에서 직접 받아 사용
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

    # 최신 데이터가 너무 오래되면 예측을 막아 잘못된 추론을 방지
    age_hours = float((now_ts - base_ts).total_seconds() / 3600.0)
    if age_hours > max_data_age_hours:
        raise HTTPException(
            status_code=404,
            detail=(
                f"최신 데이터가 너무 오래되었습니다. "
                f"기준시각={base_ts.isoformat()}, 경과={age_hours:.1f}h, 허용={max_data_age_hours}h"
            ),
        )

    # XGB는 1행, DL은 seq_len만큼 필요
    required_rows = runner.dl_seq_len if runner.model_type == "DL" else 1
    if required_rows is None:
        required_rows = 1

    if len(df) < required_rows:
        raise HTTPException(
            status_code=400,
            detail=f"예측에 필요한 전처리 행이 부족합니다. 필요={required_rows}, 현재={len(df)}",
        )

    # 최근 데이터만 모델 입력으로 사용
    rows = df.tail(required_rows).to_dict(orient="records")
    preds, _ = runner.predict(rows)

    return {
        "device_id": device_id,
        "best_model": runner.best_model,
        "preds": preds,
        "missing_features": runner.last_missing_features,
        "base_timestamp": base_ts.isoformat(),
    }


@router.get("/model-info/{device_id}")
def model_info(device_id: str):
    """장비별 모델 메타정보 조회."""
    try:
        runner = store.get_runner(device_id)
        return {
            "device_id": device_id,
            "best_model": runner.best_model,
            "model_type": runner.model_type,
            "dl_seq_len": runner.dl_seq_len,
            "required_features": runner.feature_cols,
            "model_root": MODEL_ROOT,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/predict")
def predict(req: PredictRequest):
    """클라이언트가 feature row를 직접 전달하는 수동 예측 API."""
    req_started_at = datetime.now()
    logger.info(f"[AI 예측] 수동 예측 시작 device_id={req.device_id}, rows={len(req.rows)}")
    try:
        runner = store.get_runner(req.device_id)
        preds, warns = runner.predict(req.rows)
        elapsed_ms = int((datetime.now() - req_started_at).total_seconds() * 1000)
        logger.info(f"[AI 예측] 수동 예측 결과 응답: 200 device_id={req.device_id}, elapsed_ms={elapsed_ms}")
        return {
            "device_id": req.device_id,
            "best_model": runner.best_model,
            "preds": preds,
            "missing_feature_count": runner.last_missing_count,
            "missing_features": runner.last_missing_features,
            "warnings": warns,
        }
    except FileNotFoundError as e:
        logger.warning(f"[AI 예측] 수동 예측 결과 응답: 404 device_id={req.device_id}, reason={e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.warning(f"[AI 예측] 수동 예측 결과 응답: 400 device_id={req.device_id}, reason={e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[AI 예측] 수동 예측 결과 응답: 500 device_id={req.device_id}, reason={e}")
        raise HTTPException(status_code=500, detail=f"predict_failed: {e}")


@router.get("/predict/{device_id}")
def predict_by_device(device_id: str, lookback_hours: int = 24, max_data_age_hours: int = 24):
    """device_id만으로 자동 예측.

    1) DB에서 기간 데이터 조회
    2) 전처리 수행
    3) 모델 타입(XGB/DL)에 맞춰 예측
    """
    req_started_at = datetime.now()
    logger.info(
        f"[AI 예측] 자동 예측 시작 device_id={device_id}, lookback_hours={lookback_hours}, "
        f"max_data_age_hours={max_data_age_hours}"
    )
    try:
        result = _predict_one_device(device_id, lookback_hours, max_data_age_hours)
        elapsed_ms = int((datetime.now() - req_started_at).total_seconds() * 1000)
        logger.info(f"[AI 예측] 자동 예측 결과 응답: 200 device_id={device_id}, elapsed_ms={elapsed_ms}")
        return result
    except HTTPException as e:
        # 이미 상태코드가 정의된 예외는 그대로 전달
        logger.warning(
            f"[AI 예측] 자동 예측 결과 응답: {e.status_code} device_id={device_id}, reason={e.detail}"
        )
        raise
    except FileNotFoundError as e:
        # 장비/모델 파일 누락
        logger.warning(f"[AI 예측] 자동 예측 결과 응답: 404 device_id={device_id}, reason={e}")
        raise HTTPException(status_code=404, detail=str(e))
    except DataNotFoundError as e:
        # DB 데이터 미존재/전처리 결과 없음
        logger.warning(f"[AI 예측] 자동 예측 결과 응답: 404 device_id={device_id}, reason={e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        # 입력/형식 오류 처리
        msg = str(e)
        if "Unrecognized keyword arguments passed to LSTM" in msg:
            logger.error(f"[AI 예측] 자동 예측 결과 응답: 500 device_id={device_id}, reason=DL model compatibility")
            raise HTTPException(
                status_code=500,
                detail="DL 모델 로딩 호환성 오류입니다. 학습 당시 Keras 버전과 현재 서버 버전을 맞추거나 모델을 재저장해야 합니다",
            )
        logger.warning(f"[AI 예측] 자동 예측 결과 응답: 400 device_id={device_id}, reason={e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 예외 누락 방지용 최종 가드
        logger.error(f"[AI 예측] 자동 예측 결과 응답: 500 device_id={device_id}, reason={e}")
        raise HTTPException(status_code=500, detail=f"predict_auto_failed: {e}")


@router.post("/predict/batch")
def predict_batch(req: MultiPredictRequest):
    """여러 장비를 한 번에 자동 예측한다.

    - 장비별 성공/실패를 분리해 반환
    - 일부 장비 실패 시에도 나머지는 계속 수행
    """
    started_at = datetime.now()

    if not req.device_ids:
        raise HTTPException(status_code=400, detail="device_ids는 1개 이상이어야 합니다")
    if len(req.device_ids) > 50:
        raise HTTPException(status_code=400, detail="한 번에 최대 50대까지 요청할 수 있습니다")

    logger.info(
        f"[AI 예측] 배치 예측 시작 device_count={len(req.device_ids)}, "
        f"lookback_hours={req.lookback_hours}, max_data_age_hours={req.max_data_age_hours}"
    )

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    # 장비별 독립 처리: 일부 실패해도 전체 요청은 계속 진행
    for device_id in req.device_ids:
        try:
            result = _predict_one_device(device_id, req.lookback_hours, req.max_data_age_hours)
            results.append(result)
        except HTTPException as e:
            errors.append({
                "device_id": device_id,
                "status_code": e.status_code,
                "detail": e.detail,
            })
        except FileNotFoundError as e:
            errors.append({"device_id": device_id, "status_code": 404, "detail": str(e)})
        except DataNotFoundError as e:
            errors.append({"device_id": device_id, "status_code": 404, "detail": str(e)})
        except ValueError as e:
            errors.append({"device_id": device_id, "status_code": 400, "detail": str(e)})
        except Exception as e:
            errors.append({"device_id": device_id, "status_code": 500, "detail": f"predict_auto_failed: {e}"})

    elapsed_ms = int((datetime.now() - started_at).total_seconds() * 1000)
    logger.info(
        f"[AI 예측] 배치 예측 결과 응답: 200 total={len(req.device_ids)}, "
        f"success={len(results)}, failed={len(errors)}, elapsed_ms={elapsed_ms}"
    )

    return {
        "total": len(req.device_ids),
        "success": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }


@router.post("/reload-models")
def reload_models():
    """모델 러너 캐시를 비워 최신 파일 상태로 재로딩."""
    store.clear_cache()
    return {"status": "reloaded", "model_root": MODEL_ROOT}