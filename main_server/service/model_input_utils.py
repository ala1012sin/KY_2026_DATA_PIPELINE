"""모델 입력 전처리와 raw 컬럼 보정에 쓰는 공통 유틸."""

import ast
import re
from pathlib import Path
from typing import List

import pandas as pd


_FE_SUFFIX_PATTERNS = (
    r"_lag\d+$",
    r"_diff\d+$",
    r"_pct\d+$",
    r"_roll\d+_(mean|std)$",
)

_CURRENT_MODEL_ALIAS_PAIRS = (
    ("CURVOLTAGE", "CUR_VOLTAGE"),
    ("AVG_VOLTAGE", "AVGVOLTAGE"),
    ("AVG_CURRENT", "AVGCURRENT"),
    ("CSUSAGETIME", "CS_USAGE"),
    ("MGREFILLTIME", "MG_REFILL"),
    ("URVOLT", "UR_VOLT"),
)


def to_base_feature_name(feature_name: str) -> str:
    """lag/rolling/diff 같은 파생 suffix를 제거해 원본 피처명을 반환한다."""
    base = str(feature_name)
    for pattern in _FE_SUFFIX_PATTERNS:
        base = re.sub(pattern, "", base)
    return base


def expand_feature_name_tokens(feature_name: str) -> List[str]:
    """tuple-string 형태를 포함해 raw 컬럼 후보 토큰 목록으로 펼친다."""
    base = to_base_feature_name(feature_name)
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
    """시뮬레이션 화면에 노출할 핵심 raw 입력 컬럼만 추린다."""
    del feature_cols
    allowed_fields = [
        "PRESSURE",
        "TEMPERATURE",
        "HZ",
        "AVGVOLTAGE",
        "AVGCURRENT",
        "FACTOR",
    ]
    raw_set = set(raw_columns)
    return [col for col in allowed_fields if col in raw_set]


def list_model_device_ids(model_root: str) -> List[str]:
    """current 분류 모델 폴더 기준으로 장비 목록을 반환한다."""
    cls_root = Path(model_root) / "classification"
    if not cls_root.is_dir():
        return []

    return sorted(
        {
            path.name.strip()
            for path in cls_root.iterdir()
            if path.is_dir() and path.name.strip()
        }
    )


def add_current_model_aliases(raw: pd.DataFrame) -> pd.DataFrame:
    """현재 모델이 기대하는 raw 컬럼 alias를 양방향으로 보정한다."""
    df = raw.copy()
    for src, dst in _CURRENT_MODEL_ALIAS_PAIRS:
        if dst not in df.columns and src in df.columns:
            df[dst] = df[src]
        if src not in df.columns and dst in df.columns:
            df[src] = df[dst]
    return df
