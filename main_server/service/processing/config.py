from dataclasses import dataclass
from typing import Optional, Tuple



# =========================
# Configs
# =========================
@dataclass
class PreprocessConfig: # 전처리 옵션 설정
    cutoff_datetime: Optional[str] = None
    time_col_candidates: Tuple[str, ...] = ("LOG_DT", "log_dt", "timestamp", "time", "dt")
    device_col_candidates: Tuple[str, ...] = ("DEVICE_ID", "device_id", "SERIAL_NO", "serial_no", "device", "id")
    # 타깃(전력) 후보
    target_col_candidates: Tuple[str, ...] = ("P_AVG_15M_W", "CURVOLTAGE_W", "CURVOLTAGE", "POWER", "power")

    resample_rule: str = "15min"
    fill_method: str = "none"  # 결측을 채우는 방법: ffill(직전 값)/bfill(다음 값)/none
    fill_limit: Optional[int] = None

    add_time_features: bool = True
    add_business_hour_flag: bool = False
        
    #업무시간/평일 피쳐 설정
    business_hour_start: int = 10
    business_hour_end: int = 19
        
    # 시뮬레이터 기본: 학습 파이프라인에서 지정한 입력 컬럼을 우선 사용
    fe_cols: Optional[Tuple[str, ...]] = (
        "HZ",
        "AVGVOLTAGE",
        "AVGCURRENT",
        "load_factor",
        "CSUSAGETIME_DELTA",
        "MGREFILLTIME_DELTA",
        "PRESSURE",
        "TEMPERATURE",
        "FACTOR",
        "AVG_VOLTAGE",
        "AVG_CURRENT",
        "CUR_VOLTAGE",
    )

    # rolling std 생성 여부 (피쳐 폭발/노이즈 방지용)
    add_roll_std: bool = False

    # feature engineering
    # 시뮬레이터용 기본값: 과거 시점 의존 피쳐 제거
    lag_steps: Tuple[int, ...] = ()
    roll_windows: Tuple[int, ...] = ()
    diff_steps: Tuple[int, ...] = ()
    add_pct_change: bool = False # 변화율 피쳐 생성 여부
    pct_eps: float = 1e-6 # 0 나눗셈 방지 용 

    horizons_steps: Tuple[int, int] = (1, 2)      # y_15, y_30

    # validation, test 데이터 나누는 비율 
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    use_date_split: bool = True
    test_days: int = 3
    val_days: int = 3
    min_train_days: int = 5

    # 너무 많은 원본 numeric 컬럼이 있으면 폭발하니까 제한 옵션
    # None이면 전부 사용, 숫자면 "타깃 + 상위 N개 numeric"만 사용
    max_base_cols: Optional[int] = None

    add_missing_flags: bool = True
    base_fill_method_after_fe: str = "ffill"
    base_fill_limit_after_fe: Optional[int] = 2
    engineered_fill_value: float = 0.0

    out_dir: str = "./out_preprocess"


@dataclass
class TuneConfig: # 튜닝 옵션 설정
    out_dir: str = "./out_train"
    select_by: str = "MAE_avg"

    random_seed: int = 42 # 재현성을 위해
        
    # DL, lag, rolling, diff, pct 사용 유무
    dl_use_engineered_features: bool = False

    # device 최소행
    min_rows_per_device_tabular: int = 30

    # DL
    enable_dl: bool = True
    seq_len: int = 16
    min_seq_samples: int = 20
    batch_size: int = 64
    max_epochs: int = 40
    early_stop_patience: int = 5
        
    #OP_STATUS 기반 미 가동 장비 제외 
    skip_inactive_by_op: bool = True
    op_col_name: str = "OP_STATUS"      
    op_true_ratio_min: float = 0.05     # train에서 True 비율 5% 미만이면 스킵

    # Optuna 예산
    xgb_trials: int = 30
    dl_trials: int = 20

    # Pruner(ASHA 스타일) 강도: 너무 빡세면 잘라버릴 수 있음
    pruner_min_resource: int = 5      # 최소 epoch
    pruner_reduction_factor: int = 3  # 3이면 ASHA 느낌 강함

    # 저장
    save_predictions_test: bool = True
    save_all_models: bool = False     # True면 best 아닌 모델 파일도 저장(비추)
