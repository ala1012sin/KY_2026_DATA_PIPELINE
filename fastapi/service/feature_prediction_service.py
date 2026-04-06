"""피처별 예측 비즈니스 로직 서비스."""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from fastapi import HTTPException

from service.model_store import FeatureModelStore
from service.prediction_service import predict_from_preprocessed
from service.processing.config import PreprocessConfig
from service.processing.pipeline import DataNotFoundError, preprocess_pems_pro_from_db_in_memory

# 피처 폴더명 → 응답에 사용할 대문자 레이블 매핑
_FOLDER_TO_LABEL: Dict[str, str] = {
    "pressure": "PRESSURE",
    "temperature": "TEMPERATURE",
    "hz": "HZ",
    "avgcurrent": "AVGCURRENT",
    "avgvoltage": "AVGVOLTAGE",
    "factor": "FACTOR",
}

_FEATURE_MODEL_ROOT = str(
    Path(os.environ.get("FEATURE_MODEL_ROOT", "./ai_models/feature")).expanduser().resolve()
)
feature_store = FeatureModelStore(_FEATURE_MODEL_ROOT)


def predict_features_one_device(device_id: str, lookback_hours: int) -> Dict[str, Any]:
    """장비 한 대에 대해 피처별 모델로 15·30분 예측을 수행한다."""
    if lookback_hours <= 0:
        raise HTTPException(status_code=400, detail="lookback_hours는 1 이상이어야 합니다")
    if lookback_hours > 24 * 31:
        raise HTTPException(status_code=400, detail="lookback_hours가 너무 큽니다(최대 744시간)")

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(hours=lookback_hours)

    # 전처리는 한 번만 수행 후 각 피처 모델에 공유
    try:
        meta = preprocess_pems_pro_from_db_in_memory(
            start_dt=start_dt,
            end_dt=end_dt,
            pcfg=PreprocessConfig(),
            device_ids=[device_id],
        )
    except DataNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    available_features = feature_store.list_features()
    if not available_features:
        raise HTTPException(status_code=503, detail="피처 모델 디렉터리가 비어 있습니다")

    preds: Dict[str, Dict[str, float]] = {}
    best_model_name = "XGBoost"
    found_any = False

    for folder in available_features:
        label = _FOLDER_TO_LABEL.get(folder, folder.upper())

        try:
            runner = feature_store.get_runner(folder, device_id)
        except FileNotFoundError:
            # 이 장비에 해당 피처 모델이 없으면 스킵
            continue

        try:
            result = predict_from_preprocessed(
                device_id=device_id,
                runner=runner,
                meta=meta,
                max_data_age_hours=24,
                enforce_freshness=True,
            )
        except HTTPException:
            # 데이터 부족·신선도 실패 등의 개별 피처 오류는 무시하고 계속
            continue

        if result.get("preds"):
            p = result["preds"][-1]
            # FACTOR는 -1~1 범위로 클리핑, 나머지는 0 이상으로 클리핑
            if label == "FACTOR":
                y15 = max(-1.0, min(1.0, p["y_15_pred"]))
                y30 = max(-1.0, min(1.0, p["y_30_pred"]))
            else:
                y15 = max(0.0, p["y_15_pred"])
                y30 = max(0.0, p["y_30_pred"])
            preds[label] = {"y_15_pred": y15, "y_30_pred": y30}
            best_model_name = result.get("best_model", best_model_name)
            found_any = True

    if not found_any:
        raise HTTPException(
            status_code=404,
            detail="해당 장비의 피처 예측 결과를 얻을 수 없습니다. 모델 파일 또는 데이터를 확인하세요.",
        )

    return {
        "device_id": device_id,
        "best_model": best_model_name,
        "preds": preds,
    }
