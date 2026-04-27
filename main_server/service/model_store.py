import json
import os
import shutil
import sys
import tempfile
import warnings
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from service.model_constants import AGG_RULES, DEVICE, LAG_STEPS, LOOKBACK, ROLL_WINDOWS
from service.model_input_utils import add_current_model_aliases


# 일부 학습 환경에서 joblib pickle이 numpy._core 경로를 기록하는데,
# 서버 런타임의 NumPy 버전에 따라 해당 모듈 경로가 없을 수 있어 alias를 맞춘다.
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core


PREDICTION_FRAME_COLUMNS = [
    "DEVICE_ID",
    "LOG_DT",
    "CUR_VOLTAGE",
    "CLUSTER_LABEL",
    "CUR_VOLTAGE_PRED",
    "CUR_VOLTAGE_PRED_15",
    "CUR_VOLTAGE_PRED_30",
]


def _resolve_model_root() -> str:
    """현재 전력 예측 모델 루트 디렉터리를 결정한다."""
    env = os.environ.get("MODEL_ROOT")
    if env:
        return str(Path(env).expanduser().resolve())
    return str(Path("./ai_models/current").resolve())


def _match_input_format(estimator: Any, feature_frame: pd.DataFrame) -> Any:
    """학습 시 사용한 입력 형식에 맞춰 DataFrame 또는 ndarray를 반환한다."""
    if estimator is not None and hasattr(estimator, "feature_names_in_"):
        return feature_frame
    return feature_frame.to_numpy(dtype=np.float32)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, num_classes: int = 2, dropout: float = 0.2):
        """클러스터 분류용 LSTM 분류기 레이어를 초기화한다."""
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """마지막 시점 은닉 상태로 클러스터 분류 logits를 계산한다."""
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        """전력 회귀용 LSTM 회귀기 레이어를 초기화한다."""
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """마지막 시점 은닉 상태로 단일 회귀값을 계산한다."""
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class BestModelRunner:
    """
    device=<DEVICE_ID>/best_model 아래 아티팩트를 읽어서
    XGB or DL 모델을 로드하고 predict() 제공
    """

    def __init__(self, device_dir: str):
        """장비별 best_model 아티팩트를 읽어 예측 러너를 준비한다."""
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
            self.dl_seq_len = int(ish[1]) if ish and isinstance(ish, (tuple, list)) and len(ish) >= 3 else None

    def _load_xgb_any(self) -> Dict[str, Any]:
        """15분/30분 XGBoost 모델을 가능한 저장 형식에서 로드한다."""
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

        mpath = os.path.join(self.best_dir, "model_xgb_horizons.joblib")
        if os.path.exists(mpath):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r".*If you are loading a serialized model.*")
                obj = joblib.load(mpath)
            if not isinstance(obj, dict) or ("15" not in obj or "30" not in obj):
                raise ValueError(f"Invalid xgb joblib format: {mpath}")
            return obj

        raise FileNotFoundError(
            f"XGB model not found. expected model_xgb_15.json/.ubj and model_xgb_30.json/.ubj OR model_xgb_horizons.joblib in {self.best_dir}"
        )

    def _pick_dl_path(self) -> str:
        """메타데이터와 파일 존재 여부를 바탕으로 DL 모델 파일 경로를 고른다."""
        dl_file = self.meta.get("dl_file")
        if isinstance(dl_file, str) and dl_file.strip():
            cand = os.path.join(self.best_dir, dl_file)
            if os.path.exists(cand):
                return cand

        smf = self.meta.get("saved_model_files")
        if isinstance(smf, list):
            for fn in smf:
                if isinstance(fn, str) and (fn.endswith(".keras") or fn.endswith(".h5")):
                    cand = os.path.join(self.best_dir, fn)
                    if os.path.exists(cand):
                        return cand

        for ext in ("keras", "h5"):
            cand = os.path.join(self.best_dir, f"model_{self.best_model}.{ext}")
            if os.path.exists(cand):
                return cand

        return os.path.join(self.best_dir, f"model_{self.best_model}.keras")

    def _load_dl_any(self):
        """호환성 보정을 적용해 Keras/HDF5 기반 DL 모델을 로드한다."""
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
            """구버전 레이어 인자 차이를 흡수하면서 모델을 읽는다."""
            try:
                return tf.keras.models.load_model(path, compile=False)
            except Exception as e:
                msg = str(e)
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

        def _build_dl_candidates(primary_path: str) -> List[str]:
            """주 모델 경로와 대체 확장자를 포함한 후보 목록을 만든다."""
            candidates = [primary_path]
            if primary_path.endswith(".keras"):
                alt = primary_path[:-6] + ".h5"
                if alt not in candidates:
                    candidates.append(alt)
            elif primary_path.endswith(".h5"):
                alt = primary_path[:-3] + ".keras"
                if alt not in candidates:
                    candidates.append(alt)
            return candidates

        def _try_load_candidate(path: str, errors: List[str]):
            """하나의 후보 경로를 zip/hdf5/direct 순서로 시도해 로드한다."""
            if not os.path.exists(path):
                errors.append(f"missing: {path}")
                return None

            if zipfile.is_zipfile(path):
                try:
                    return _load_with_compat(path)
                except Exception as e:
                    errors.append(f"zip-load-failed({path}): {e}")
                    return None

            is_hdf5 = False
            try:
                is_hdf5 = h5py.is_hdf5(path)
            except Exception as e:
                errors.append(f"hdf5-signature-check-failed({path}): {e}")

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
                    return None

            try:
                return _load_with_compat(path)
            except Exception as e:
                errors.append(f"direct-load-failed({path}): {e}")
                return None

        primary = self._pick_dl_path()
        candidates = _build_dl_candidates(primary)
        errors: List[str] = []
        for path in candidates:
            loaded = _try_load_candidate(path, errors)
            if loaded is not None:
                return loaded

        raise ValueError("DL model load failed. " + " | ".join(errors))

    def _prepare_rows_to_X2(self, rows: List[Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
        """입력 row 목록을 2차원 특징 행렬로 바꾸고 누락 피처를 기록한다."""
        self.last_missing_features = []
        self.last_missing_count = 0

        warns = []
        n = len(rows)
        X = np.zeros((n, len(self.feature_cols)), dtype=np.float32)
        missing_features_set = set()

        for i, row in enumerate(rows):
            missing = 0
            for j, col in enumerate(self.feature_cols):
                val = row.get(col, None)
                if val is None:
                    missing += 1
                    missing_features_set.add(col)
                    X[i, j] = 0.0
                else:
                    try:
                        X[i, j] = float(val)
                    except Exception:
                        X[i, j] = 0.0
                        missing += 1
                        missing_features_set.add(col)
            if missing > 0:
                warns.append(f"row[{i}] missing {missing} features -> filled 0")

        self.last_missing_features = sorted(missing_features_set)
        self.last_missing_count = len(self.last_missing_features)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if self.scaler is not None and self.model_type == "XGB":
            X = self.scaler.transform(X).astype(np.float32)

        return X, warns

    def _scale_3d(self, X3: np.ndarray) -> np.ndarray:
        """시퀀스 입력을 스케일러에 맞게 3차원 형태로 변환·복원한다."""
        if self.scaler is None:
            return X3
        shp = X3.shape
        X2 = X3.reshape(-1, shp[-1])
        X2s = self.scaler.transform(X2)
        return X2s.reshape(shp).astype(np.float32)

    def predict(self, rows: List[Dict[str, float]]):
        """로드된 best model로 15분/30분 예측을 수행한다."""
        if self.model_type == "XGB":
            X2, warns = self._prepare_rows_to_X2(rows)
            m15 = self.model_obj["15"]
            m30 = self.model_obj["30"]
            p15 = m15.predict(X2)
            p30 = m30.predict(X2)
            preds = [{"y_15_pred": float(a), "y_30_pred": float(b)} for a, b in zip(p15, p30)]
            return preds, warns

        if not self.dl_seq_len or self.dl_seq_len < 2:
            raise ValueError("DL model input seq_len is invalid")
        if len(rows) < self.dl_seq_len:
            raise ValueError(f"DL requires >= seq_len rows. need={self.dl_seq_len}, got={len(rows)}")

        tail = rows[-self.dl_seq_len:]
        X2, warns = self._prepare_rows_to_X2(tail)
        X3 = X2.reshape(1, self.dl_seq_len, -1)
        X3 = self._scale_3d(X3)

        y = self.model_obj.predict(X3, verbose=0)
        preds = [{"y_15_pred": float(y[0, 0]), "y_30_pred": float(y[0, 1])}]
        return preds, warns


def _coerce_numeric_frame(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """지정한 컬럼을 숫자형으로 강제 변환한다."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _make_sequences(X_arr: np.ndarray, lookback: int) -> np.ndarray:
    """LSTM 입력용 고정 길이 시퀀스 배열을 생성한다."""
    xs = []
    for i in range(lookback, len(X_arr)):
        xs.append(X_arr[i - lookback:i])
    return np.asarray(xs, dtype=np.float32)


def _empty_prediction_frame() -> pd.DataFrame:
    """예측 결과가 비었을 때 공통으로 사용할 빈 DataFrame 스키마를 만든다."""
    return pd.DataFrame(columns=PREDICTION_FRAME_COLUMNS)


class TwoStagePredictionRunner:
    def __init__(self, model_root: str, device_id: str):
        """장비별 분류기와 클러스터별 회귀 모델을 준비한다."""
        self.model_root = str(Path(model_root).resolve())
        self.device_id = str(device_id)
        self.classification_dir = os.path.join(self.model_root, "classification", self.device_id)
        self.regression_root = os.path.join(self.model_root, "regression", self.device_id)
        self._regression_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}

        if not os.path.isdir(self.regression_root):
            raise FileNotFoundError(f"regression model not found: {self.regression_root}")

        self.available_clusters = sorted(
            int(path.name.split("_", 1)[1])
            for path in Path(self.regression_root).iterdir()
            if path.is_dir() and path.name.startswith("cluster_") and path.name.split("_", 1)[1].isdigit()
        )
        if not self.available_clusters:
            raise FileNotFoundError(f"no regression clusters found: {self.regression_root}")

        self.has_classification = os.path.isdir(self.classification_dir)
        if self.has_classification:
            with open(os.path.join(self.classification_dir, "meta.json"), "r", encoding="utf-8") as f:
                self.cls_meta = json.load(f)
            with open(os.path.join(self.classification_dir, "feature_cols.json"), "r", encoding="utf-8") as f:
                self.feature_cols = json.load(f)

            scaler_path = os.path.join(self.classification_dir, "scaler.joblib")
            self.cls_scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            self.cls_model = joblib.load(os.path.join(self.classification_dir, "model.pkl"))
            self.best_model = f"{self.cls_meta.get('best_model', 'classifier')}+cluster"
            self.model_type = "TWO_STAGE"
        else:
            if len(self.available_clusters) != 1:
                raise FileNotFoundError(
                    f"classification model not found and regression cluster is not singular: {self.device_id}"
                )
            only_cluster = self.available_clusters[0]
            reg_bundle = self._load_regression_bundle(only_cluster)
            self.cls_meta = {
                "best_model": "NO_CLASSIFICATION_SINGLE_CLUSTER",
                "optimal_k": 1,
            }
            self.feature_cols = reg_bundle["feature_cols"]
            self.cls_scaler = None
            self.cls_model = None
            self.best_model = f"{reg_bundle['meta'].get('best_model', 'regressor')}+cluster"
            self.model_type = "SINGLE_CLUSTER"

        self.dl_seq_len = LOOKBACK
        self.last_missing_features: List[str] = []
        self.last_missing_count = 0
        self.required_raw_columns = ["DEVICE_ID", "LOG_DT"] + list(AGG_RULES.keys())

    def _normalize_raw(self, raw_df: pd.DataFrame, device_id: Optional[str] = None) -> pd.DataFrame:
        """원본 로그 컬럼명을 표준화하고 기본 타입을 정리한다."""
        df = raw_df.copy()
        if df.empty:
            return df

        if "DEVICE_ID" not in df.columns:
            df["DEVICE_ID"] = str(device_id or self.device_id)
        else:
            df["DEVICE_ID"] = df["DEVICE_ID"].fillna(str(device_id or self.device_id)).astype(str)

        if "LOG_DT" not in df.columns:
            end = pd.Timestamp.now().floor("15min")
            df["LOG_DT"] = pd.date_range(end=end, periods=len(df), freq="15min")
        else:
            df["LOG_DT"] = pd.to_datetime(df["LOG_DT"], errors="coerce")

        df = add_current_model_aliases(df)

        numeric_candidates = [col for col in AGG_RULES if col in df.columns]
        df = _coerce_numeric_frame(df, numeric_candidates)
        if "OP_STATUS" in df.columns:
            df["OP_STATUS"] = pd.to_numeric(df["OP_STATUS"], errors="coerce")

        self.last_missing_features = sorted(col for col in self.required_raw_columns if col not in df.columns)
        self.last_missing_count = len(self.last_missing_features)
        return df.dropna(subset=["LOG_DT"]).sort_values(["DEVICE_ID", "LOG_DT"]).reset_index(drop=True)

    def _preprocess(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """원본 로그를 15분 단위 예측 입력 형태로 집계·보정한다."""
        df = self._normalize_raw(raw_df, device_id=self.device_id)
        if df.empty:
            return df

        if "CUR_VOLTAGE" in df.columns:
            df["CUR_VOLTAGE"] = df["CUR_VOLTAGE"].clip(lower=0)
        if "FACTOR" in df.columns:
            df["FACTOR"] = df["FACTOR"].clip(0, 1)

        agg = {col: rule for col, rule in AGG_RULES.items() if col in df.columns}
        if not agg:
            return pd.DataFrame(columns=["DEVICE_ID", "LOG_DT"])

        def _resample_per_device(grouped_df: pd.DataFrame) -> pd.DataFrame:
            """장비별 15분 집계를 수행한다."""
            parts = []
            for device_id, group in grouped_df.groupby("DEVICE_ID"):
                g2 = (
                    group.set_index("LOG_DT")
                    .resample("15min")[list(agg.keys())]
                    .agg(agg)
                    .reset_index()
                )
                g2.insert(0, "DEVICE_ID", device_id)
                parts.append(g2)
            return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["DEVICE_ID", "LOG_DT"])

        def _zero_stopped_rows(frame: pd.DataFrame) -> pd.DataFrame:
            """정지 상태로 판단된 bin의 전력/전류/전압 값을 0으로 맞춘다."""
            if "OP_STATUS" in frame.columns:
                frame["IS_STOPPED"] = (frame["OP_STATUS"] == 0).astype(int)
            else:
                frame["IS_STOPPED"] = 0
            stopped = frame["IS_STOPPED"] == 1
            for col in ["CUR_VOLTAGE", "AVGVOLTAGE", "AVGCURRENT"]:
                if col in frame.columns:
                    frame.loc[stopped, col] = 0.0
            return frame

        df = _resample_per_device(df)
        if df.empty:
            return pd.DataFrame(columns=["DEVICE_ID", "LOG_DT"])

        df = _zero_stopped_rows(df)

        numeric_cols = list(df.select_dtypes(include="number").columns)
        for col in numeric_cols:
            df[col] = df.groupby("DEVICE_ID")[col].transform(
                lambda s: s.interpolate(method="linear", limit=3)
            )

        if "CUR_VOLTAGE" in df.columns:
            def clip_iqr(series: pd.Series) -> pd.Series:
                """IQR 기준으로 전력 이상치를 완만하게 클리핑한다."""
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                return series.clip(lower=max(0, q1 - 1.5 * iqr), upper=q3 + 1.5 * iqr)

            df["CUR_VOLTAGE"] = df.groupby("DEVICE_ID")["CUR_VOLTAGE"].transform(clip_iqr)

        return df.sort_values(["DEVICE_ID", "LOG_DT"]).reset_index(drop=True)

    def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """분류·회귀 공통으로 쓰는 시계열 파생 피처를 생성한다."""
        if df.empty:
            return df

        out = []
        for device_id, group in df.groupby("DEVICE_ID"):
            g = group.sort_values("LOG_DT").copy()

            if "CUR_VOLTAGE" not in g.columns:
                g["CUR_VOLTAGE"] = 0.0

            for window in ROLL_WINDOWS:
                rolled = g["CUR_VOLTAGE"].shift(1).rolling(window, min_periods=1)
                g[f"CUR_VOLTAGE_roll_mean_{window}"] = rolled.mean()
                g[f"CUR_VOLTAGE_roll_std_{window}"] = rolled.std().fillna(0)
                g[f"CUR_VOLTAGE_roll_max_{window}"] = rolled.max()
                g[f"CUR_VOLTAGE_roll_min_{window}"] = rolled.min()

            for lag in LAG_STEPS:
                g[f"CUR_VOLTAGE_lag_{lag}"] = g["CUR_VOLTAGE"].shift(lag)

            if "AVGVOLTAGE" in g.columns and "AVGCURRENT" in g.columns:
                g["APPARENT_POWER"] = g["AVGVOLTAGE"] * g["AVGCURRENT"]
            else:
                g["APPARENT_POWER"] = 0.0
            if "FACTOR" in g.columns:
                g["ACTIVE_POWER"] = g["APPARENT_POWER"] * g["FACTOR"]
            else:
                g["ACTIVE_POWER"] = 0.0

            if "OP_STATUS" in g.columns:
                g["OP_STATUS_CHANGE"] = g["OP_STATUS"].diff().abs().fillna(0)
            else:
                g["OP_STATUS_CHANGE"] = 0.0

            g["HOUR"] = g["LOG_DT"].dt.hour
            g["DAY_OF_WEEK"] = g["LOG_DT"].dt.dayofweek
            g["IS_WEEKEND"] = (g["DAY_OF_WEEK"] >= 5).astype(int)
            g["IS_PEAK_HOUR"] = g["HOUR"].apply(lambda hour: 1 if (9 <= hour <= 12 or 13 <= hour <= 17) else 0)
            out.append(g)

        return pd.concat(out, ignore_index=True)

    def _load_regression_bundle(self, cluster_k: int, horizon_min: int = 15) -> Dict[str, Any]:
        """클러스터/예측시점별 회귀 모델 번들을 로드하고 캐시한다."""
        cache_key = (cluster_k, horizon_min)
        if cache_key in self._regression_cache:
            return self._regression_cache[cache_key]

        cluster_dir = os.path.join(self.regression_root, f"cluster_{cluster_k}")
        if not os.path.isdir(cluster_dir):
            raise FileNotFoundError(f"regression model not found: {cluster_dir}")

        reg_dir = os.path.join(cluster_dir, f"{int(horizon_min)}min")
        if not os.path.isdir(reg_dir):
            reg_dir = cluster_dir

        with open(os.path.join(reg_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(os.path.join(reg_dir, "feature_cols.json"), "r", encoding="utf-8") as f:
            feature_cols = json.load(f)

        bundle = {
            "dir": reg_dir,
            "meta": meta,
            "feature_cols": feature_cols,
            "target_scaler": joblib.load(os.path.join(reg_dir, "scaler.joblib")),
            "feat_scaler": joblib.load(os.path.join(reg_dir, "feat_scaler.joblib"))
            if os.path.exists(os.path.join(reg_dir, "feat_scaler.joblib"))
            else None,
            "model": joblib.load(os.path.join(reg_dir, "model.pkl")),
            "horizon_min": int(meta.get("horizon_min", horizon_min)),
        }
        self._regression_cache[cache_key] = bundle
        return bundle

    def _predict_cluster_labels(self, df_dev: pd.DataFrame) -> np.ndarray:
        """장비 시계열 각 행의 클러스터 라벨을 예측한다."""
        if not self.has_classification:
            return np.full(len(df_dev), self.available_clusters[0], dtype=int)

        feature_cols = self.feature_cols
        feature_frame = df_dev.reindex(columns=feature_cols).fillna(0).astype(np.float32)
        best_name = str(self.cls_meta.get("best_model", "")).upper()

        if best_name == "MLP" and self.cls_scaler is not None:
            scaled_input = self.cls_scaler.transform(_match_input_format(self.cls_scaler, feature_frame))
            return np.asarray(self.cls_model.predict(scaled_input), dtype=int)

        if best_name == "LSTM":
            lookback = int(self.cls_meta.get("lookback") or LOOKBACK)
            base_input = _match_input_format(self.cls_scaler, feature_frame) if self.cls_scaler is not None else feature_frame
            X_scaled = (
                self.cls_scaler.transform(base_input)
                if self.cls_scaler is not None
                else feature_frame.to_numpy(dtype=np.float32)
            )
            Xs = _make_sequences(X_scaled, lookback)
            if len(Xs) == 0:
                return np.zeros(len(df_dev), dtype=int)

            blob = self.cls_model
            input_size = int(blob.get("input_size", feature_frame.shape[1])) if isinstance(blob, dict) else feature_frame.shape[1]
            num_classes = int(self.cls_meta.get("optimal_k", 2))
            state_dict = blob.get("state_dict") if isinstance(blob, dict) else (
                blob.state_dict() if hasattr(blob, "state_dict") else blob
            )

            model = LSTMClassifier(input_size=input_size, num_classes=num_classes).to(DEVICE)
            model.load_state_dict(state_dict)
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(Xs, dtype=torch.float32, device=DEVICE)).cpu().numpy()
            preds_seq = logits.argmax(axis=1)
            labels = np.full(len(df_dev), preds_seq[0], dtype=int)
            labels[lookback:] = preds_seq
            return labels

        predict_input = _match_input_format(self.cls_model, feature_frame)
        return np.asarray(self.cls_model.predict(predict_input), dtype=int)

    def _predict_voltage_for_cluster(self, cluster_k: int, seg: pd.DataFrame, horizon_min: int = 15) -> np.ndarray:
        """특정 클러스터 구간에 대해 horizon별 전력 예측값을 계산한다."""
        try:
            bundle = self._load_regression_bundle(cluster_k, horizon_min=horizon_min)
        except FileNotFoundError:
            return np.full(len(seg), np.nan, dtype=np.float32)

        feature_cols = bundle["feature_cols"]
        feature_frame = seg.reindex(columns=feature_cols).fillna(0).astype(np.float32)
        best_name = str(bundle["meta"].get("best_model", "")).upper()
        model = bundle["model"]
        feat_scaler = bundle["feat_scaler"]
        target_scaler = bundle["target_scaler"]

        if best_name == "LSTM":
            lookback = int(bundle["meta"].get("lookback") or LOOKBACK)
            base_input = _match_input_format(feat_scaler, feature_frame) if feat_scaler is not None else feature_frame
            X_scaled = (
                feat_scaler.transform(base_input).astype(np.float32)
                if feat_scaler is not None
                else feature_frame.to_numpy(dtype=np.float32)
            )
            Xs = _make_sequences(X_scaled, lookback)
            if len(Xs) == 0:
                return np.full(len(seg), np.nan, dtype=np.float32)

            blob = model
            input_size = int(blob.get("input_size", feature_frame.shape[1])) if isinstance(blob, dict) else feature_frame.shape[1]
            state_dict = blob.get("state_dict") if isinstance(blob, dict) else (
                blob.state_dict() if hasattr(blob, "state_dict") else blob
            )

            lstm_model = LSTMRegressor(input_size=input_size).to(DEVICE)
            lstm_model.load_state_dict(state_dict)
            lstm_model.eval()

            preds_seq = []
            with torch.no_grad():
                for i in range(0, len(Xs), 2048):
                    chunk = torch.tensor(Xs[i:i + 2048], dtype=torch.float32, device=DEVICE)
                    preds_seq.append(lstm_model(chunk).cpu().numpy())
            preds_seq = np.concatenate(preds_seq)
            y_pred_s = np.full(len(seg), preds_seq[0], dtype=np.float32)
            y_pred_s[lookback:] = preds_seq
        elif best_name == "MLP":
            base_input = _match_input_format(feat_scaler, feature_frame) if feat_scaler is not None else feature_frame
            X_scaled = (
                feat_scaler.transform(base_input)
                if feat_scaler is not None
                else feature_frame.to_numpy(dtype=np.float32)
            )
            y_pred_s = model.predict(X_scaled)
        else:
            predict_input = _match_input_format(model, feature_frame)
            y_pred_s = model.predict(predict_input)

        y_pred = target_scaler.inverse_transform(np.asarray(y_pred_s).reshape(-1, 1)).flatten()
        return np.clip(y_pred, 0, None)

    def predict_frame(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """원본 로그 전체 구간에 대한 행 단위 예측 프레임을 만든다."""
        df_clean = self._preprocess(raw_df)
        if df_clean.empty:
            return _empty_prediction_frame()

        df_feat = self._feature_engineer(df_clean)
        def _predict_device_frame(dev_df: pd.DataFrame) -> pd.DataFrame:
            """장비 하나의 cluster/horizon 예측 컬럼을 모두 채운다."""
            dev_df = dev_df.sort_values("LOG_DT").reset_index(drop=True).copy()
            cluster_labels = self._predict_cluster_labels(dev_df)
            dev_df["CLUSTER_LABEL"] = cluster_labels
            dev_df["CUR_VOLTAGE_PRED_15"] = np.nan
            dev_df["CUR_VOLTAGE_PRED_30"] = np.nan

            for cluster_k in sorted(np.unique(cluster_labels)):
                seg_idx = dev_df.index[dev_df["CLUSTER_LABEL"] == cluster_k]
                seg = dev_df.loc[seg_idx]
                dev_df.loc[seg_idx, "CUR_VOLTAGE_PRED_15"] = self._predict_voltage_for_cluster(int(cluster_k), seg, horizon_min=15)
                dev_df.loc[seg_idx, "CUR_VOLTAGE_PRED_30"] = self._predict_voltage_for_cluster(int(cluster_k), seg, horizon_min=30)

            dev_df["CUR_VOLTAGE_PRED"] = dev_df["CUR_VOLTAGE_PRED_15"]
            return dev_df[PREDICTION_FRAME_COLUMNS]

        results = [_predict_device_frame(dev_df) for _, dev_df in df_feat.groupby("DEVICE_ID")]
        return pd.concat(results, ignore_index=True) if results else _empty_prediction_frame()

    def predict_latest(self, raw_df: pd.DataFrame, reference_timestamp: Optional[Any] = None) -> Dict[str, Any]:
        """가장 최근 기준시각의 15분/30분 전력 예측만 추려 반환한다."""
        pred_df = self.predict_frame(raw_df)
        if pred_df.empty:
            raise ValueError("예측 가능한 전처리 결과가 없습니다")

        pred_df["LOG_DT"] = pd.to_datetime(pred_df["LOG_DT"], errors="coerce")
        pred_df = pred_df.dropna(subset=["LOG_DT"]).sort_values("LOG_DT").reset_index(drop=True)
        if pred_df.empty:
            raise ValueError("유효한 예측 시각이 없습니다")

        target_df = pred_df
        if reference_timestamp is not None:
            ref_ts = pd.Timestamp(reference_timestamp)
            if ref_ts.tzinfo is not None:
                ref_ts = ref_ts.tz_convert("UTC").tz_localize(None)
            target_df = pred_df[pred_df["LOG_DT"] <= ref_ts].copy()
            if target_df.empty:
                raise ValueError("기준시각 이전 예측 결과가 없습니다")

        non_null = target_df[
            target_df["CUR_VOLTAGE_PRED_15"].notna() | target_df["CUR_VOLTAGE_PRED_30"].notna()
        ].copy()
        if non_null.empty:
            raise ValueError("예측값이 생성되지 않았습니다")

        row = non_null.iloc[-1]
        pred_15 = row["CUR_VOLTAGE_PRED_15"]
        pred_30 = row["CUR_VOLTAGE_PRED_30"]
        pred_15_value = float(pred_15) if pd.notna(pred_15) else float(pred_30)
        pred_30_value = float(pred_30) if pd.notna(pred_30) else pred_15_value
        return {
            "timestamp": pd.Timestamp(row["LOG_DT"]).isoformat(),
            "cluster_label": int(row["CLUSTER_LABEL"]),
            "cur_voltage_pred": pred_15_value,
            "preds": [{
                "y_15_pred": pred_15_value,
                "y_30_pred": pred_30_value,
            }],
        }


class ModelStore:
    def __init__(self, model_root: str):
        """장비별 TwoStagePredictionRunner 스토어를 초기화한다."""
        self.model_root = str(Path(model_root).resolve())

    @lru_cache(maxsize=256)
    def get_runner(self, device_id: str) -> TwoStagePredictionRunner:
        """장비별 예측 러너를 캐시해서 반환한다."""
        return TwoStagePredictionRunner(self.model_root, device_id)

    def clear_cache(self):
        """장비 예측 러너 캐시를 비운다."""
        self.get_runner.cache_clear()


def _resolve_feature_model_root() -> str:
    """피처별 예측 모델 루트 디렉터리를 결정한다."""
    env = os.environ.get("FEATURE_MODEL_ROOT")
    if env:
        return str(Path(env).expanduser().resolve())
    return str(Path("./ai_models/feature").resolve())


class FeatureModelStore:
    """
    ai_models/feature/{feature_name}/device={device_id}/best_model/ 구조에서
    피처별 모델 러너를 로드·캐시하는 스토어.
    """

    def __init__(self, feature_root: str):
        """피처별 모델 루트 경로를 기준으로 스토어를 초기화한다."""
        self.feature_root = str(Path(feature_root).resolve())

    def list_features(self) -> List[str]:
        """저장소에 존재하는 피처 모델 목록을 반환한다."""
        p = Path(self.feature_root)
        if not p.exists():
            return []
        return sorted([d.name for d in p.iterdir() if d.is_dir()])

    def _device_dir(self, feature: str, device_id: str) -> str:
        """피처/장비 조합의 모델 디렉터리 경로를 만든다."""
        return os.path.join(self.feature_root, feature, f"device={device_id}")

    @lru_cache(maxsize=1024)
    def get_runner(self, feature: str, device_id: str) -> BestModelRunner:
        """피처/장비별 best model 러너를 캐시해서 반환한다."""
        ddir = self._device_dir(feature, device_id)
        if not os.path.exists(ddir):
            raise FileNotFoundError(f"feature model not found: feature={feature}, device={device_id}")
        return BestModelRunner(ddir)

    def clear_cache(self):
        """피처 모델 러너 캐시를 비운다."""
        self.get_runner.cache_clear()
