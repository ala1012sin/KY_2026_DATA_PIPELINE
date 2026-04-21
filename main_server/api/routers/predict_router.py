"""예측 관련 API 라우터."""

from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from Logger import Logger as logger
from api.schemas.predict import MultiPredictRequest, PredictRequest
from api.schemas.responses import AutoPredictResponse, FeaturePredictResponse, ManualPredictResponse, PredictBatchResponse
from service.feature_prediction_service import predict_features_one_device
from service.monitoring_service import record_api_event
from service.processing.pipeline import DataNotFoundError
from service.prediction_service import predict_manual, predict_one_device

router = APIRouter()


@router.post("/predict", response_model=ManualPredictResponse)
def predict(req: PredictRequest):
    """클라이언트가 feature row를 직접 전달하는 수동 예측 API."""
    # 라우터는 요청/응답과 로깅만 담당, 비즈니스 로직은 service로 위임
    req_started_at = datetime.now()
    logger.info(f"[AI 예측] 수동 예측 시작 device_id={req.device_id}, rows={len(req.rows)}")
    try:
        result = predict_manual(req.device_id, req.rows)
        record_api_event(endpoint="/predict", device_id=req.device_id, status_code=200)
        elapsed_ms = int((datetime.now() - req_started_at).total_seconds() * 1000)
        logger.info(f"[AI 예측] 수동 예측 결과 응답: 200 device_id={req.device_id}, elapsed_ms={elapsed_ms}")
        return result
    except FileNotFoundError as e:
        record_api_event(endpoint="/predict", device_id=req.device_id, status_code=404, detail=str(e))
        logger.warning(f"[AI 예측] 수동 예측 결과 응답: 404 device_id={req.device_id}, reason={e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        record_api_event(endpoint="/predict", device_id=req.device_id, status_code=400, detail=str(e))
        logger.warning(f"[AI 예측] 수동 예측 결과 응답: 400 device_id={req.device_id}, reason={e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        record_api_event(endpoint="/predict", device_id=req.device_id, status_code=500, detail=str(e))
        logger.error(f"[AI 예측] 수동 예측 결과 응답: 500 device_id={req.device_id}, reason={e}")
        raise HTTPException(status_code=500, detail=f"predict_failed: {e}")


@router.get("/predict/feature/{device_id}", response_model=FeaturePredictResponse)
def predict_feature_by_device(device_id: str, lookback_hours: int = Query(24, ge=1, le=24 * 31)):
    """피처별 전용 모델로 15·30분 예측을 수행한다."""
    req_started_at = datetime.now()
    logger.info(f"[AI 피처예측] 시작 device_id={device_id}, lookback_hours={lookback_hours}")
    try:
        result = predict_features_one_device(device_id, lookback_hours)
        elapsed_ms = int((datetime.now() - req_started_at).total_seconds() * 1000)
        logger.info(f"[AI 피처예측] 완료 device_id={device_id}, elapsed_ms={elapsed_ms}")
        return result
    except HTTPException as e:
        logger.warning(f"[AI 피처예측] {e.status_code} device_id={device_id}, reason={e.detail}")
        raise
    except Exception as e:
        logger.error(f"[AI 피처예측] 500 device_id={device_id}, reason={e}")
        raise HTTPException(status_code=500, detail=f"feature_predict_failed: {e}")


@router.get("/predict/{device_id}", response_model=AutoPredictResponse)
def predict_by_device(device_id: str, lookback_hours: int = Query(24, ge=1, le=24 * 31)):
    """device_id만으로 자동 예측."""
    # 자동 예측: 최신 데이터 조회/전처리/예측을 service에서 일괄 수행
    req_started_at = datetime.now()
    logger.info(
        f"[AI 예측] 자동 예측 시작 device_id={device_id}, lookback_hours={lookback_hours}, "
        "max_data_age_hours=24(fixed)"
    )
    try:
        result = predict_one_device(device_id, lookback_hours, 24)
        record_api_event(endpoint="/predict/{device_id}", device_id=device_id, status_code=200)
        elapsed_ms = int((datetime.now() - req_started_at).total_seconds() * 1000)
        logger.info(f"[AI 예측] 자동 예측 결과 응답: 200 device_id={device_id}, elapsed_ms={elapsed_ms}")
        return result
    except HTTPException as e:
        record_api_event(
            endpoint="/predict/{device_id}",
            device_id=device_id,
            status_code=e.status_code,
            detail=str(e.detail),
        )
        logger.warning(
            f"[AI 예측] 자동 예측 결과 응답: {e.status_code} device_id={device_id}, reason={e.detail}"
        )
        raise
    except FileNotFoundError as e:
        record_api_event(endpoint="/predict/{device_id}", device_id=device_id, status_code=404, detail=str(e))
        logger.warning(f"[AI 예측] 자동 예측 결과 응답: 404 device_id={device_id}, reason={e}")
        raise HTTPException(status_code=404, detail=str(e))
    except DataNotFoundError as e:
        record_api_event(endpoint="/predict/{device_id}", device_id=device_id, status_code=404, detail=str(e))
        logger.warning(f"[AI 예측] 자동 예측 결과 응답: 404 device_id={device_id}, reason={e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        msg = str(e)
        if "Unrecognized keyword arguments passed to LSTM" in msg:
            record_api_event(
                endpoint="/predict/{device_id}",
                device_id=device_id,
                status_code=500,
                detail="DL model compatibility",
            )
            logger.error(f"[AI 예측] 자동 예측 결과 응답: 500 device_id={device_id}, reason=DL model compatibility")
            raise HTTPException(
                status_code=500,
                detail="DL 모델 로딩 호환성 오류입니다. 학습 당시 Keras 버전과 현재 서버 버전을 맞추거나 모델을 재저장해야 합니다",
            )
        record_api_event(endpoint="/predict/{device_id}", device_id=device_id, status_code=400, detail=str(e))
        logger.warning(f"[AI 예측] 자동 예측 결과 응답: 400 device_id={device_id}, reason={e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        record_api_event(endpoint="/predict/{device_id}", device_id=device_id, status_code=500, detail=str(e))
        logger.error(f"[AI 예측] 자동 예측 결과 응답: 500 device_id={device_id}, reason={e}")
        raise HTTPException(status_code=500, detail=f"predict_auto_failed: {e}")


@router.post("/predict/batch", response_model=PredictBatchResponse)
def predict_batch(req: MultiPredictRequest):
    """여러 장비를 한 번에 자동 예측한다."""
    # 배치 예측: 장비별 오류를 분리해 부분 성공 허용
    started_at = datetime.now()

    if not req.device_ids:
        raise HTTPException(status_code=400, detail="device_ids는 1개 이상이어야 합니다")
    if len(req.device_ids) > 50:
        raise HTTPException(status_code=400, detail="한 번에 최대 50대까지 요청할 수 있습니다")

    logger.info(
        f"[AI 예측] 배치 예측 시작 device_count={len(req.device_ids)}, "
        f"lookback_hours={req.lookback_hours}, max_data_age_hours=24(fixed)"
    )

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for device_id in req.device_ids:
        try:
            result = predict_one_device(device_id, req.lookback_hours, 24)
            record_api_event(endpoint="/predict/batch-item", device_id=device_id, status_code=200)
            results.append(result)
        except HTTPException as e:
            record_api_event(
                endpoint="/predict/batch-item",
                device_id=device_id,
                status_code=e.status_code,
                detail=str(e.detail),
            )
            errors.append({
                "device_id": device_id,
                "status_code": e.status_code,
                "detail": e.detail,
            })
        except FileNotFoundError as e:
            record_api_event(endpoint="/predict/batch-item", device_id=device_id, status_code=404, detail=str(e))
            errors.append({"device_id": device_id, "status_code": 404, "detail": str(e)})
        except DataNotFoundError as e:
            record_api_event(endpoint="/predict/batch-item", device_id=device_id, status_code=404, detail=str(e))
            errors.append({"device_id": device_id, "status_code": 404, "detail": str(e)})
        except ValueError as e:
            record_api_event(endpoint="/predict/batch-item", device_id=device_id, status_code=400, detail=str(e))
            errors.append({"device_id": device_id, "status_code": 400, "detail": str(e)})
        except Exception as e:
            record_api_event(endpoint="/predict/batch-item", device_id=device_id, status_code=500, detail=str(e))
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
