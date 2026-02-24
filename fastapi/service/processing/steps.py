"""Reusable preprocessing step functions."""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import TuneConfig


# =========================
# Column inference
# =========================
def infer_col(df: pd.DataFrame, candidates: Tuple[str, ...], required=True) -> Optional[str]:
    """컬럼 자동 찾기 함수
        후보 이름 중 실제 DF에 존재하는 컬럼을 찾아서 반환"""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Required column not found. candidates={candidates}")
    return None


# =========================
# Preprocess helpers
# =========================
def resample_15m_per_device(
    df: pd.DataFrame,
    time_col: str,
    device_col: str,
    rule: str,
    fill_method: str,
    fill_limit=None
) -> pd.DataFrame:
    """
    - 15분 bin으로 mean 집계
    - n_obs(해당 bin에 원본 관측치 개수) 생성
    - fill_method="none"이면 공백(NaN) 그대로 유지
    """
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col, device_col]).sort_values([device_col, time_col])

    num_cols = [c for c in d.columns if c not in [time_col, device_col] and pd.api.types.is_numeric_dtype(d[c])]
    if not num_cols:
        raise ValueError("No numeric columns to resample.")

    g = d.set_index(time_col).groupby(device_col)

    mean_df = g[num_cols].resample(rule).mean()
    cnt_df = g.resample(rule).size().rename("n_obs")  # bin별 관측 개수

    out = pd.concat([mean_df, cnt_df], axis=1).reset_index().sort_values([device_col, time_col])

    # 공백 유지(추천): fill_method="none"
    if fill_method and str(fill_method).lower() in ("ffill", "bfill"):
        fill_cols = num_cols  # n_obs는 채우지 않음
        out[fill_cols] = out.groupby(device_col, group_keys=False)[fill_cols].apply(
            lambda gg: gg.fillna(method=fill_method, limit=fill_limit)
        )

    return out


def add_time_features(df_15: pd.DataFrame, time_col: str, start: int = 10, end: int = 19) -> pd.DataFrame:
    """시간 피쳐 생성
        - 요일/ 시간대 패턴 학습을 위해"""
    d = df_15.copy()
    t = pd.to_datetime(d[time_col])
    d["hour"] = t.dt.hour
    d["dayofweek"] = t.dt.dayofweek
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)
    
    # 평일 여부
    d["is_business_day"] = (d["dayofweek"] < 5).astype(int)

    # 업무시간(10~19) 여부
    d["is_business_hour"] = ((d["hour"] >= start) & (d["hour"] <= end)).astype(int)
    
    return d


def pick_base_cols(df_15: pd.DataFrame, time_col: str, device_col: str, target_col: str, max_base_cols: Optional[int]) -> List[str]:
    """ 기본 컬럼 설정 함수
        컬럼 수 과다로 폭발 방지"""
    numeric_cols = [c for c in df_15.columns if c not in [time_col, device_col] and pd.api.types.is_numeric_dtype(df_15[c])]
    others = [c for c in numeric_cols if c != target_col]
    if max_base_cols is None:
        return [target_col] + others
    return [target_col] + others[:max_base_cols]

def add_feature_engineering(
    df_15: pd.DataFrame, time_col: str, device_col: str,
    base_cols: List[str],
    lag_steps: Tuple[int, ...], roll_windows: Tuple[int, ...], diff_steps: Tuple[int, ...],
    add_pct_change: bool, pct_eps: float,
    add_roll_std: bool = False,   
    session_col: Optional[str] = "session_id", 
) -> pd.DataFrame:
    d = df_15.copy().sort_values([device_col, time_col])

    # 세션 단위로 groupby (session_col 없으면 device 단위로 fallback)
    if session_col and session_col in d.columns:
        keys = [device_col, session_col]
        gb = d.groupby(keys, group_keys=False)
        reset_lv = [0, 1]
    else:
        gb = d.groupby(device_col, group_keys=False)
        reset_lv = 0

    # 여러 feature 생성
    for col in base_cols:
        # 변화량
        for k in diff_steps:
            d[f"{col}_diff{k}"] = gb[col].diff(k)
        # 0 계산 대비
        if add_pct_change:
            for k in diff_steps:
                prev = gb[col].shift(k)
                d[f"{col}_pct{k}"] = (d[col] - prev) / (np.abs(prev) + pct_eps)
        # Rolling 
        for w in roll_windows:
            d[f"{col}_roll{w}_mean"] = gb[col].rolling(w).mean().reset_index(level=reset_lv, drop=True)
            if add_roll_std:
                d[f"{col}_roll{w}_std"]  = gb[col].rolling(w).std().reset_index(level=reset_lv, drop=True)
        # 과거값
        for k in lag_steps:
            d[f"{col}_lag{k}"] = gb[col].shift(k)

    return d

def add_gap_features(df_15: pd.DataFrame, time_col: str, device_col: str, rule: str = "15min", n_obs_col: str = "n_obs") -> pd.DataFrame:
    """
    lag feature를 의미 있게 만들기 위해서 운영시간(10:00 ~ 19:00) 외의 시간에서의 갭을 feature로 추가
    """
    d = df_15.copy().sort_values([device_col, time_col])
    step = pd.to_timedelta(rule)

    if n_obs_col not in d.columns:
        d[n_obs_col] = 1

    is_obs = d[n_obs_col].fillna(0).astype(float) > 0
    obs_time = d[time_col].where(is_obs)

    last_obs_time = obs_time.groupby(d[device_col]).ffill()
    prev_obs_time = last_obs_time.groupby(d[device_col]).shift(1)

    gap_bins = (d[time_col] - prev_obs_time) / step  # float
    missing_bins = (gap_bins - 1).clip(lower=0)

    # 여기서 NaN/inf를 먼저 처리하고 int로 변환 (np.where로 astype(int) 먼저 실행되는 문제 제거)
    mb = missing_bins.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["gap_prev_obs_bins"] = np.round(mb).astype("int64")

    return d

def add_session_features(
    df_15: pd.DataFrame,
    time_col: str,
    device_col: str,
    rule: str = "15min",
    gap_col: str = "gap_prev_obs_bins",
    break_gap_bins: int = 2,   # gap_prev_obs_bins >= 2면 새 세션(30분 이상 끊김)
) -> pd.DataFrame:
    """
    - 세션(연속 운영 구간) 단위로 lag/rolling/target이 계산되도록 session_id 생성
    - 다음날 10:00 같은 긴 공백 뒤 첫 관측은 세션 시작점으로 처리됨
    """
    d = df_15.copy().sort_values([device_col, time_col])

    step_min = int(pd.to_timedelta(rule).total_seconds() // 60)

    if gap_col not in d.columns:
        # gap 없으면 time diff로 추정(최소 안전장치)
        dtm = d.groupby(device_col)[time_col].diff().dt.total_seconds() / 60.0
        bins = np.rint(dtm / step_min).fillna(0).astype(int)
        d[gap_col] = np.clip(bins - 1, 0, None).astype(int)
    else:
        d[gap_col] = pd.to_numeric(d[gap_col], errors="coerce").fillna(0).astype(int)

    # 세션 분리 플래그
    d["is_session_break"] = (d[gap_col] >= break_gap_bins).astype(int)

    # 세션 id
    d["session_id"] = d.groupby(device_col)["is_session_break"].cumsum().astype(int)

    # 세션 내 위치
    g = d.groupby([device_col, "session_id"], sort=False)
    d["is_session_start"] = (g.cumcount() == 0).astype(int)
    d["session_pos"] = g.cumcount().astype(int)
    d["minutes_from_session_start"] = d["session_pos"] * step_min

    # 직전 관측과의 간격(분) - 첫 행은 0으로 보정
    d["minutes_since_prev_obs"] = (d[gap_col] + 1) * step_min
    first = (d.groupby(device_col).cumcount() == 0)
    d.loc[first, "minutes_since_prev_obs"] = 0

    return d


def add_targets(df_feat: pd.DataFrame, time_col: str, device_col: str, target_col: str, horizons_steps: Tuple[int, int]) -> pd.DataFrame:
    """장비별 타겟값 생성 (세션 경계 넘지 않도록 session_id 기준 포함)"""
    d = df_feat.copy().sort_values([device_col, time_col])

    if "session_id" in d.columns:
        gb = d.groupby([device_col, "session_id"], group_keys=False)
    else:
        gb = d.groupby(device_col, group_keys=False)

    h1, h2 = horizons_steps
    d["y_15"] = gb[target_col].shift(-h1)
    d["y_30"] = gb[target_col].shift(-h2)
    return d


def add_splits_per_device(df: pd.DataFrame, time_col: str, device_col: str, val_ratio: float, test_ratio: float) -> pd.DataFrame:
    """장비별 시간 순 정렬 후 test/val/train으로 분할하는 함수
        시계열 데이터의 특성을 고려해 랜덤으로 뽑지 않고 시간순대로 선택"""
    d = df.copy().sort_values([device_col, time_col])

    def _split_one(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(time_col).copy()
        n = len(g)
        if n < 10:
            g["split"] = "train"
            return g
        test_start = int((1.0 - test_ratio) * n)
        val_start  = int((1.0 - test_ratio - val_ratio) * n)
        val_start = max(0, min(val_start, test_start - 1))
        g["split"] = "train"
        g.iloc[val_start:test_start, g.columns.get_loc("split")] = "val"
        g.iloc[test_start:, g.columns.get_loc("split")] = "test"
        return g

    return d.groupby(device_col, group_keys=False).apply(_split_one)


def fill_feature_nans_devicewise(df_sup, time_col, device_col, fill_limit=None):
    """각 컬럼의 lag/rolling 때문에 생기는 결측을 정리하는 함수"""
    d = df_sup.copy().sort_values([device_col, time_col])
    exclude = {time_col, device_col, "split", "y_15", "y_30"}
    feat_cols = [c for c in d.columns if c not in exclude and pd.api.types.is_numeric_dtype(d[c])]
    if not feat_cols:
        return d
    d[feat_cols] = d[feat_cols].replace([np.inf, -np.inf], np.nan)
#    d[feat_cols] = d.groupby(device_col, group_keys=False)[feat_cols].apply(
#        lambda g: g.ffill(limit=fill_limit)
#    )   주말, 평일 공백이 있다면 유지하기 위해 제외
    d[feat_cols] = d[feat_cols].fillna(0.0)
    return d



# =========================
# Training helpers (robust)
# =========================
def _op_status_to_bool(x: pd.Series) -> pd.Series:
    # True/False, 1/0, "TRUE"/"FALSE" 등 섞여도 처리
    if x.dtype == bool:
        return x
    if pd.api.types.is_numeric_dtype(x):
        return x.fillna(0).astype(float) > 0.05  # resample mean이면 0.5 기준
    # 문자열이면
    s = x.astype(str).str.strip().str.lower()
    return s.isin(["true", "1", "t", "y", "yes"])

def _inactive_by_op_status(d0: pd.DataFrame, cfg: TuneConfig):
    op = cfg.op_col_name
    if op not in d0.columns:
        return False, ""  # op가 없으면 이 기준으로는 스킵 안 함(다른 기준으로 판단 가능)

    # train 구간만 판단(없으면 전체)
    if "split" in d0.columns:
        dtr = d0[d0["split"] == "train"].copy()
    else:
        dtr = d0.copy()

    if len(dtr) < max(10, cfg.min_rows_per_device_tabular):
        return True, f"too_few_train_rows<{len(dtr)}>"

    on = _op_status_to_bool(dtr[op])
    true_ratio = float(on.mean())

    if true_ratio < cfg.op_true_ratio_min:
        return True, f"OP_STATUS_true_ratio={true_ratio:.3f} < {cfg.op_true_ratio_min}"
    return False, ""

def _build_feature_cols_robust(df: pd.DataFrame, time_col: str, device_col: str) -> List[str]:
    """학습에 사용될 Feature 리스트 생성"""
    exclude = {time_col, device_col, "split", "y_15", "y_30","session_id"}
    feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols:
        return []
    tmp = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    feat_cols = [c for c in feat_cols if tmp[c].notna().sum() > 0]
    return feat_cols


def _clean_device_frame(d: pd.DataFrame, feat_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """장비별 데이터를 학습 가능한 형태로 바꾸는 함수
        타겟값 제거, Nan값 정리"""
    d = d.dropna(subset=["y_15", "y_30"]).copy()
    if len(d) == 0:
        return d, []
    d[feat_cols] = d[feat_cols].replace([np.inf, -np.inf], np.nan)
    d[feat_cols] = d[feat_cols].fillna(0.0)
    keep = []
    for c in feat_cols:
        if d[c].notna().sum() == 0:
            continue
        if d[c].nunique(dropna=False) <= 1:
            continue
        keep.append(c)
    return d, keep


def _pick_seq_len(n_rows: int, base: int = 16) -> int:
    """데이터가 적은 경우 seq_len을 자동으로 조정하는 함수
        딥러닝 모델이 학습 가능한 샘플 수 확보 목적"""
    if n_rows <= 20:
        return 4
    if n_rows <= 40:
        return 8
    return min(base, max(4, n_rows // 8))


def make_sequences(d: pd.DataFrame, feat_cols: List[str], seq_len: int, time_col: str, session_col: str = "session_id"):
    d = d.sort_values(time_col).copy()
    d = d.dropna(subset=["y_15", "y_30"]).copy()
    if len(d) < seq_len:
        return np.empty((0, seq_len, len(feat_cols)), np.float32), np.empty((0, 2), np.float32), pd.DataFrame()

    X = d[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    y = d[["y_15", "y_30"]].to_numpy(np.float32)

    # 안전: DL은 NaN 들어가면 터질 수 있으니 0으로 정리
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    sid = None
    if session_col in d.columns:
        sid = d[session_col].to_numpy()

    X_list, y_list, meta_list = [], [], []
    for i in range(seq_len - 1, len(d)):
        # 세션 경계 넘는 window는 스킵
        if sid is not None:
            s0 = sid[i]
            if not np.all(sid[i - seq_len + 1:i + 1] == s0):
                continue

        X_list.append(X[i - seq_len + 1:i + 1])
        y_list.append(y[i])
        meta_list.append({
            time_col: d.iloc[i][time_col],
            "split": d.iloc[i]["split"] if "split" in d.columns else "train",
            "y_15_true": float(d.iloc[i]["y_15"]),
            "y_30_true": float(d.iloc[i]["y_30"]),
        })

    if not X_list:
        return np.empty((0, seq_len, len(feat_cols)), np.float32), np.empty((0, 2), np.float32), pd.DataFrame()

    X3 = np.stack(X_list, axis=0).astype(np.float32)
    y2 = np.stack(y_list, axis=0).astype(np.float32)
    meta = pd.DataFrame(meta_list)
    return X3, y2, meta


def _fallback_seq_split(n: int, min_train=20, min_val=8, min_test=5):
    if n < (min_train + min_val + min_test):
        test_n = min(min_test, max(1, n // 5))
        train_n = n - test_n
        tr = np.zeros(n, dtype=bool); va = np.zeros(n, dtype=bool); te = np.zeros(n, dtype=bool)
        tr[:train_n] = True
        te[train_n:] = True
        return tr, va, te
    train_end = int(0.7 * n)
    val_end   = int(0.85 * n)
    tr = np.zeros(n, dtype=bool); va = np.zeros(n, dtype=bool); te = np.zeros(n, dtype=bool)
    tr[:train_end] = True
    va[train_end:val_end] = True
    te[val_end:] = True
    return tr, va, te
