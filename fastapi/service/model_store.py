# fastapi/app/model_store.py
import os, json, shutil, tempfile, warnings
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import joblib

# DL 포맷 판별용
import zipfile
import h5py


def _resolve_model_root() -> str:
    """
    MODEL_ROOT 우선:
      1) ENV MODEL_ROOT
      2) 기본 ./models/current  (docker-compose에서 /app 기준이면 /app/models/current가 됨)
    """
    env = os.environ.get("MODEL_ROOT")
    if env:
        return str(Path(env).expanduser().resolve())
    return str(Path("./models/current").resolve())


class BestModelRunner:
    """
    device=<DEVICE_ID>/best_model 아래 아티팩트를 읽어서
    XGB or DL 모델을 로드하고 predict() 제공
    """

    def __init__(self, device_dir: str):
        self.device_dir = str(device_dir)
        self.best_dir = os.path.join(self.device_dir, "best_model")

        meta_path = os.path.join(self.best_dir, "meta.json")
        feat_path = os.path.join(self.best_dir, "feature_cols.json")
        if not (os.path.exists(meta_path) and os.path.exists(feat_path)):
            raise FileNotFoundError(f"Missing meta/feature_cols in {self.best_dir}")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(feat_path, "r", encoding="utf-8") as f:
            self.feature_cols: List[str] = json.load(f)

        self.best_model = str(self.meta.get("best_model"))
        self.model_type = "XGB" if self.best_model.upper().startswith("XGBOOST") else "DL"

        # scaler: XGB는 2D(StandardScaler), DL은 3D flatten 학습용(StandardScaler)일 가능성
        scaler_path = os.path.join(self.best_dir, "scaler.joblib")
        self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        self.model_obj: Any = None
        self.dl_seq_len: Optional[int] = None
        self.last_missing_features: List[str] = []
        self.last_missing_count: int = 0

        if self.model_type == "XGB":
            self.model_obj = self._load_xgb_any()
        else:
            self.model_obj = self._load_dl_any()
            ish = getattr(self.model_obj, "input_shape", None)
            # (batch, seq_len, n_feat) 형태를 기대
            self.dl_seq_len = int(ish[1]) if ish and isinstance(ish, (tuple, list)) and len(ish) >= 3 else None

    # -----------------------
    # Loaders
    # -----------------------
    def _load_xgb_any(self) -> Dict[str, Any]:
        """
        XGB 로딩 우선순위:
          1) save_model 포맷: model_xgb_15.json / model_xgb_30.json (또는 .ubj)
          2) 기존 joblib: model_xgb_horizons.joblib
        """
        # 1) json/ubj 우선
        for ext in ("json", "ubj"):
            p15 = os.path.join(self.best_dir, f"model_xgb_15.{ext}")
            p30 = os.path.join(self.best_dir, f"model_xgb_30.{ext}")
            if os.path.exists(p15) and os.path.exists(p30):
                try:
                    from xgboost import XGBRegressor
                    m15 = XGBRegressor()
                    m30 = XGBRegressor()
                    m15.load_model(p15)
                    m30.load_model(p30)
                    return {"15": m15, "30": m30}
                except Exception as e:
                    raise ValueError(f"Failed to load XGB save_model files ({ext}): {e}")

        # 2) joblib fallback
        mpath = os.path.join(self.best_dir, "model_xgb_horizons.joblib")
        if os.path.exists(mpath):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*If you are loading a serialized model.*",
                )
                obj = joblib.load(mpath)  # {"15": model, "30": model}
            if not isinstance(obj, dict) or ("15" not in obj or "30" not in obj):
                raise ValueError(f"Invalid xgb joblib format: {mpath}")
            return obj

        raise FileNotFoundError(
            f"XGB model not found. expected model_xgb_15.json/.ubj and model_xgb_30.json/.ubj OR model_xgb_horizons.joblib in {self.best_dir}"
        )

    def _pick_dl_path(self) -> str:
        """
        DL 파일 경로 결정 우선순위:
          1) meta["dl_file"] (있으면)
          2) meta["saved_model_files"]에서 *.keras/*.h5 첫 번째
          3) 기본: model_{best_model}.keras 또는 .h5
        """
        # 1) meta에 명시된 경우
        dl_file = self.meta.get("dl_file")
        if isinstance(dl_file, str) and dl_file.strip():
            cand = os.path.join(self.best_dir, dl_file)
            if os.path.exists(cand):
                return cand

        # 2) saved_model_files (학습 코드에서 기록하게 한 경우)
        smf = self.meta.get("saved_model_files")
        if isinstance(smf, list):
            for fn in smf:
                if isinstance(fn, str) and (fn.endswith(".keras") or fn.endswith(".h5")):
                    cand = os.path.join(self.best_dir, fn)
                    if os.path.exists(cand):
                        return cand

        # 3) 기본 규칙
        for ext in ("keras", "h5"):
            cand = os.path.join(self.best_dir, f"model_{self.best_model}.{ext}")
            if os.path.exists(cand):
                return cand

        # 없으면 기존 경로(.keras)로 반환해 상위에서 FileNotFoundError 처리
        return os.path.join(self.best_dir, f"model_{self.best_model}.keras")

    def _load_dl_any(self):
        """
        DL 로딩:
          - zip .keras => 그대로 load_model
          - hdf5 시그니처인데 확장자만 .keras => 임시 .h5로 복사해서 load_model
                    - 1차 파일 포맷이 깨졌으면 .h5/.keras 대체 파일 자동 폴백
        """
        import tensorflow as tf

        class CompatLSTM(tf.keras.layers.LSTM):
            def __init__(self, *args, **kwargs):
                kwargs.pop("time_major", None)
                super().__init__(*args, **kwargs)

        class CompatGRU(tf.keras.layers.GRU):
            def __init__(self, *args, **kwargs):
                kwargs.pop("time_major", None)
                super().__init__(*args, **kwargs)

        class CompatSimpleRNN(tf.keras.layers.SimpleRNN):
            def __init__(self, *args, **kwargs):
                kwargs.pop("time_major", None)
                super().__init__(*args, **kwargs)

        def _load_with_compat(path: str):
            try:
                return tf.keras.models.load_model(path, compile=False)
            except Exception as e:
                msg = str(e)
                # 구버전 학습 모델에서 남아있는 time_major 인자를 무시하고 재로딩
                if "time_major" in msg or "Unrecognized keyword arguments" in msg:
                    return tf.keras.models.load_model(
                        path,
                        compile=False,
                        custom_objects={
                            "LSTM": CompatLSTM,
                            "GRU": CompatGRU,
                            "SimpleRNN": CompatSimpleRNN,
                        },
                    )
                raise

        primary = self._pick_dl_path()
        candidates = [primary]

        # 1차 파일이 깨졌을 때 같은 prefix의 대체 확장자로 폴백
        if primary.endswith(".keras"):
            alt = primary[:-6] + ".h5"
            if alt not in candidates:
                candidates.append(alt)
        elif primary.endswith(".h5"):
            alt = primary[:-3] + ".keras"
            if alt not in candidates:
                candidates.append(alt)

        # 후보 파일을 순차 시도하고 실패 원인을 누적해 최종 에러 메시지로 반환
        errors: List[str] = []
        for path in candidates:
            if not os.path.exists(path):
                errors.append(f"missing: {path}")
                continue

            # zip .keras (정상)
            if zipfile.is_zipfile(path):
                try:
                    return _load_with_compat(path)
                except Exception as e:
                    errors.append(f"zip-load-failed({path}): {e}")
                    continue

            is_hdf5 = False
            try:
                is_hdf5 = h5py.is_hdf5(path)
            except Exception as e:
                errors.append(f"hdf5-signature-check-failed({path}): {e}")

            # hdf5면 로드 시도 (.keras 확장자면 임시 .h5로 복사)
            if is_hdf5:
                try:
                    if path.endswith(".keras"):
                        with tempfile.TemporaryDirectory() as td:
                            tmp_h5 = os.path.join(td, "tmp_model.h5")
                            shutil.copyfile(path, tmp_h5)
                            return _load_with_compat(tmp_h5)
                    return _load_with_compat(path)
                except Exception as e:
                    errors.append(f"hdf5-load-failed({path}): {e}")
                    continue

            # 포맷 판별 실패 케이스 대비: Keras 직접 로딩 최종 시도
            try:
                return _load_with_compat(path)
            except Exception as e:
                errors.append(f"direct-load-failed({path}): {e}")

        # 모든 후보가 실패하면 누적 사유를 함께 반환해 디버깅 쉽게 함
        raise ValueError("DL model load failed. " + " | ".join(errors))

    # -----------------------
    # Input prep
    # -----------------------
    def _prepare_rows_to_X2(self, rows: List[Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
        # 최근 예측에서의 누락 피처 요약 정보를 초기화
        self.last_missing_features = []
        self.last_missing_count = 0

        warnings = []
        n = len(rows)
        X = np.zeros((n, len(self.feature_cols)), dtype=np.float32)
        missing_features_set = set()

        for i, r in enumerate(rows):
            missing = 0
            for j, c in enumerate(self.feature_cols):
                v = r.get(c, None)
                if v is None:
                    missing += 1
                    missing_features_set.add(c)
                    X[i, j] = 0.0
                else:
                    try:
                        X[i, j] = float(v)
                    except Exception:
                        X[i, j] = 0.0
                        missing += 1
                        missing_features_set.add(c)
            if missing > 0:
                warnings.append(f"row[{i}] missing {missing} features -> filled 0")

        self.last_missing_features = sorted(missing_features_set)
        self.last_missing_count = len(self.last_missing_features)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # XGB는 2D scaler 적용
        if self.scaler is not None and self.model_type == "XGB":
            X = self.scaler.transform(X).astype(np.float32)

        return X, warnings

    def _scale_3d(self, X3: np.ndarray) -> np.ndarray:
        # DL은 3D를 (N*seq_len, feat)로 펼쳐 scaler 적용 후 다시 reshape
        if self.scaler is None:
            return X3
        shp = X3.shape
        X2 = X3.reshape(-1, shp[-1])
        X2s = self.scaler.transform(X2)
        return X2s.reshape(shp).astype(np.float32)

    # -----------------------
    # Predict
    # -----------------------
    def predict(self, rows: List[Dict[str, float]]):
        if self.model_type == "XGB":
            X2, warns = self._prepare_rows_to_X2(rows)
            m15 = self.model_obj["15"]
            m30 = self.model_obj["30"]
            p15 = m15.predict(X2)
            p30 = m30.predict(X2)
            preds = [{"y_15_pred": float(a), "y_30_pred": float(b)} for a, b in zip(p15, p30)]
            return preds, warns

        # DL
        if not self.dl_seq_len or self.dl_seq_len < 2:
            raise ValueError("DL model input seq_len is invalid")
        if len(rows) < self.dl_seq_len:
            raise ValueError(f"DL requires >= seq_len rows. need={self.dl_seq_len}, got={len(rows)}")

        tail = rows[-self.dl_seq_len:]
        X2, warns = self._prepare_rows_to_X2(tail)      # (seq_len, n_feat)
        X3 = X2.reshape(1, self.dl_seq_len, -1)         # (1, seq_len, n_feat)
        X3 = self._scale_3d(X3)

        y = self.model_obj.predict(X3, verbose=0)
        preds = [{"y_15_pred": float(y[0, 0]), "y_30_pred": float(y[0, 1])}]
        return preds, warns


class ModelStore:
    def __init__(self, model_root: str):
        self.model_root = str(Path(model_root).resolve())

    def _device_dir(self, device_id: str) -> str:
        return os.path.join(self.model_root, f"device={device_id}")

    @lru_cache(maxsize=256)
    def get_runner(self, device_id: str) -> BestModelRunner:
        ddir = self._device_dir(device_id)
        if not os.path.exists(ddir):
            raise FileNotFoundError(f"device not found: {ddir}")
        return BestModelRunner(ddir)

    def clear_cache(self):
        self.get_runner.cache_clear()