"""모델 메타/관리 관련 API 라우터."""

from fastapi import APIRouter, HTTPException

from api.schemas.responses import ModelInfoResponse, ReloadModelsResponse
from service.prediction_service import MODEL_ROOT, clear_model_caches, store

router = APIRouter()


@router.get("/model-info/{device_id}", response_model=ModelInfoResponse)
def model_info(device_id: str):
    """장비별 모델 메타정보 조회."""
    # 모델 종류/필수 feature 등 화면 진단용 메타 반환
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


@router.post("/reload-models", response_model=ReloadModelsResponse)
def reload_models():
    """모델 러너 캐시를 비워 최신 파일 상태로 재로딩."""
    # 모델 파일 교체/갱신 이후 수동 캐시 초기화 용도
    clear_model_caches()
    return {"status": "reloaded", "model_root": MODEL_ROOT}
