"""최적화 서비스 모듈."""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from Logger import Logger as logger
from db.public.models import TB_PEAK_DISPATCH_DEVICE_RESULT, TB_PEAK_DISPATCH_RUN
from infrastructure.queryFactory.base_orm import BaseQueryFactory
from setting.database_orm import db_connection_pool
from service.prediction_service import (
    MODEL_ROOT,
    list_model_device_ids,
    predict_from_preprocessed,
    store,
)
from service.processing.config import PreprocessConfig
from service.processing.pipeline import fetch_pems_pro_log_df, preprocess_pems_pro_from_db_in_memory


def solve_test_milp(
    gains_kw: List[float],
    costs: List[float],
    budget: float,
    max_actions: Optional[int] = None,
    mandatory_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """간단한 0/1 MILP 테스트 문제를 푼다."""
    if len(gains_kw) != len(costs):
        raise ValueError("gains_kw와 costs 길이는 같아야 합니다")
    if len(gains_kw) == 0:
        raise ValueError("최소 1개 이상의 항목이 필요합니다")
    if budget < 0:
        raise ValueError("budget은 0 이상이어야 합니다")

    gains = np.asarray(gains_kw, dtype=float)
    cost_values = np.asarray(costs, dtype=float)

    if np.any(gains < 0):
        raise ValueError("gains_kw 값은 음수가 될 수 없습니다")
    if np.any(cost_values < 0):
        raise ValueError("costs 값은 음수가 될 수 없습니다")

    n = len(gains_kw)
    must_indices = mandatory_indices or []
    for idx in must_indices:
        if idx < 0 or idx >= n:
            raise ValueError(f"mandatory_indices 범위를 벗어났습니다: {idx}")

    if max_actions is not None and max_actions > n:
        raise ValueError("max_actions는 항목 개수보다 클 수 없습니다")

    c = -gains
    integrality = np.ones(n, dtype=int)
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))

    constraints: List[LinearConstraint] = [
        LinearConstraint(cost_values.reshape(1, -1), -np.inf, np.array([float(budget)])),
    ]

    if max_actions is not None:
        constraints.append(
            LinearConstraint(np.ones((1, n)), -np.inf, np.array([float(max_actions)]))
        )

    for idx in must_indices:
        row = np.zeros((1, n))
        row[0, idx] = 1.0
        constraints.append(LinearConstraint(row, np.array([1.0]), np.array([1.0])))

    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"disp": False},
    )

    solution = np.asarray(result.x if result.x is not None else np.zeros(n), dtype=float)
    selected_indices = [i for i, v in enumerate(solution) if v >= 0.5]

    objective_gain_kw = float(gains[selected_indices].sum()) if selected_indices else 0.0
    total_cost = float(cost_values[selected_indices].sum()) if selected_indices else 0.0

    selected_items = [
        {
            "index": i,
            "gain_kw": float(gains[i]),
            "cost": float(cost_values[i]),
        }
        for i in selected_indices
    ]

    return {
        "status": "optimal" if bool(result.success) else "failed",
        "success": bool(result.success),
        "message": str(result.message),
        "objective_gain_kw": objective_gain_kw,
        "total_cost": total_cost,
        "selected_indices": selected_indices,
        "selected_items": selected_items,
        "raw_solution": [float(v) for v in solution.tolist()],
    }


def _load_device_threshold(device_id: str) -> Optional[float]:
    """장비별 threshold.json에서 피크 임계치를 읽어 반환한다.

    우선순위:
    1) threshold_p95
    2) threshold_meta.threshold
    """
    threshold_path = Path(MODEL_ROOT) / f"device={device_id}" / "best_model" / "threshold.json"
    if not threshold_path.exists():
        return None

    try:
        with open(threshold_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    raw = payload.get("threshold_p95")
    if raw is None:
        raw = (payload.get("threshold_meta") or {}).get("threshold")

    try:
        return float(raw)
    except Exception:
        return None


def _mean_op_status_by_device(device_ids: List[str], lookback_hours: int) -> Dict[str, float]:
    """최근 lookback 구간의 장비별 평균 OP_STATUS를 계산한다."""
    if not device_ids:
        return {}

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(hours=lookback_hours)
    raw = fetch_pems_pro_log_df(start_dt=start_dt, end_dt=end_dt, device_ids=device_ids)
    if raw.empty or "DEVICE_ID" not in raw.columns or "OP_STATUS" not in raw.columns:
        return {}

    values = raw[["DEVICE_ID", "OP_STATUS"]].copy()
    values["OP_STATUS"] = pd.to_numeric(values["OP_STATUS"], errors="coerce")
    values = values.dropna(subset=["OP_STATUS"])
    if values.empty:
        return {}

    grouped = values.groupby("DEVICE_ID")["OP_STATUS"].mean()
    return {str(k): float(v) for k, v in grouped.to_dict().items()}


def _format_kw_from_w(value_w: float) -> str:
    """W 단위 값을 kW 문자열(소수 2자리)로 변환한다."""
    return f"{(float(value_w) / 1000.0):.2f}"


def _predict_devices_batched(
    device_ids: List[str],
    lookback_hours: int,
) -> (Dict[str, Dict[str, Any]], Dict[str, str]):
    """장비별 예측을 위한 전처리를 1회만 수행하고, 장비별로 추론을 분리 실행한다."""
    if not device_ids:
        return {}, {}

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(hours=lookback_hours)

    meta = preprocess_pems_pro_from_db_in_memory(
        start_dt=start_dt,
        end_dt=end_dt,
        pcfg=PreprocessConfig(),
        device_ids=device_ids,
    )

    output: Dict[str, Dict[str, Any]] = {}
    failed: Dict[str, str] = {}
    for device_id in device_ids:
        try:
            runner = store.get_runner(device_id)
            output[device_id] = predict_from_preprocessed(
                device_id=device_id,
                runner=runner,
                meta=meta,
                max_data_age_hours=24,
                enforce_freshness=True,
            )
        except Exception as e:
            failed[device_id] = str(e)

    return output, failed


def _build_distribution_text_for_device(device_row: Dict[str, Any], allocation_plan: List[Dict[str, Any]]) -> str:
    """장비별 15/30분 추천 분배 문구를 생성한다."""
    recs: List[str] = []
    donor_id = str(device_row.get("device_id", ""))

    horizons = [
        {
            "minute": 15,
            "base": float(device_row.get("baseline_15", 0.0)),
            "threshold": float(device_row.get("threshold", 0.0)),
            "out": float(device_row.get("shift_out_15", 0.0)),
        },
        {
            "minute": 30,
            "base": float(device_row.get("baseline_30", 0.0)),
            "threshold": float(device_row.get("threshold", 0.0)),
            "out": float(device_row.get("shift_out_30", 0.0)),
        },
    ]

    for h in horizons:
        if h["base"] <= h["threshold"] or h["out"] <= 1e-9:
            continue

        allocations = [
            x for x in allocation_plan
            if int(x.get("minute", 0)) == h["minute"] and str(x.get("from_device_id", "")) == donor_id
        ]
        allocations.sort(key=lambda x: float(x.get("power_w", 0.0)), reverse=True)

        over_w = h["base"] - h["threshold"]
        if allocations:
            main_receiver = allocations[0]
            recs.append(
                f"{h['minute']}분 후 임계치 초과 예상량은 {_format_kw_from_w(over_w)}kW입니다. "
                f"{main_receiver.get('to_device_id')} 장비로 약 {_format_kw_from_w(float(main_receiver.get('power_w', 0.0)))}kW를 분배하면 "
                f"피크 완화가 가능합니다."
            )
        else:
            recs.append(
                f"{h['minute']}분 후 임계치 초과 예상량은 {_format_kw_from_w(over_w)}kW이며, 현재 최적화에서는 "
                f"수신 장비가 확보되지 않아 분배 효과가 제한적입니다."
            )

    if not recs:
        return "현재 조건에서는 임계치 초과 예상이 없어 분배 문구가 없습니다."
    return "\n".join(recs)


def _save_peak_dispatch_result(result_payload: Dict[str, Any], customer_id: Optional[str] = None) -> None:
    """피크 분배 MILP 실행 결과를 헤더/장비 상세 테이블에 저장한다."""
    db_gen = db_connection_pool()
    db = next(db_gen)
    now_dt = datetime.now()

    try:
        run_row = BaseQueryFactory(db, TB_PEAK_DISPATCH_RUN).insert_single_row(
            customer_id=customer_id,
            status=str(result_payload.get("status", "unknown")),
            success=bool(result_payload.get("success", False)),
            message=result_payload.get("message"),
            lookback_hours=int(result_payload.get("lookback_hours", 0)),
            top_k=int(result_payload.get("top_k", 0)),
            idle_op_status_threshold=float(result_payload.get("idle_op_status_threshold", 0.0)),
            force_exceed_demo=bool(result_payload.get("force_exceed_demo", False)),
            force_exceed_margin_ratio=float(result_payload.get("force_exceed_margin_ratio", 0.05)),
            device_count=int(result_payload.get("device_count", 0)),
            peak_15_before=float(result_payload.get("peak_15_before", 0.0)),
            peak_15_after=float(result_payload.get("peak_15_after", 0.0)),
            peak_30_before=float(result_payload.get("peak_30_before", 0.0)),
            peak_30_after=float(result_payload.get("peak_30_after", 0.0)),
            objective_peak_sum=float(result_payload.get("objective_peak_sum", 0.0)),
            total_slack=float(result_payload.get("total_slack", 0.0)),
            donor_device_ids=result_payload.get("donor_device_ids") or [],
            idle_device_ids=result_payload.get("idle_device_ids") or [],
            allocation_plan=result_payload.get("allocation_plan") or [],
            created_at=now_dt,
        )

        device_results = result_payload.get("devices") or []
        device_repo = BaseQueryFactory(db, TB_PEAK_DISPATCH_DEVICE_RESULT)
        for item in device_results:
            device_repo.insert_single_row(
                peak_run_id=run_row.peak_run_id,
                device_id=item.get("device_id"),
                is_donor=bool(item.get("is_donor", False)),
                is_idle=bool(item.get("is_idle", False)),
                op_status_mean=float(item.get("op_status_mean", 0.0)),
                threshold=float(item.get("threshold", 0.0)),
                baseline_15=float(item.get("baseline_15", 0.0)),
                baseline_30=float(item.get("baseline_30", 0.0)),
                optimized_15=float(item.get("optimized_15", 0.0)),
                optimized_30=float(item.get("optimized_30", 0.0)),
                delta_15=float(item.get("delta_15", 0.0)),
                delta_30=float(item.get("delta_30", 0.0)),
                shift_in_15=float(item.get("shift_in_15", 0.0)),
                shift_in_30=float(item.get("shift_in_30", 0.0)),
                shift_out_15=float(item.get("shift_out_15", 0.0)),
                shift_out_30=float(item.get("shift_out_30", 0.0)),
                required_shift_15=float(item.get("required_shift_15", 0.0)),
                required_shift_30=float(item.get("required_shift_30", 0.0)),
                slack_15=float(item.get("slack_15", 0.0)),
                slack_30=float(item.get("slack_30", 0.0)),
                distributed_targets_15=item.get("distributed_targets_15") or [],
                distributed_targets_30=item.get("distributed_targets_30") or [],
                distribution_text=item.get("distribution_text"),
                created_at=now_dt,
            )
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


def optimize_peak_dispatch_test(
    lookback_hours: int = 24,
    top_k: int = 2,
    idle_op_status_threshold: float = 0.05,
    force_exceed_demo: bool = False,
    force_exceed_margin_ratio: float = 0.05,
) -> Dict[str, Any]:
    """전체 장비를 하나의 회사로 보고 상위 사용량 장비 부하를 분배해 피크를 낮춘다."""
    total_start = time.perf_counter()

    if lookback_hours <= 0:
        raise ValueError("lookback_hours는 1 이상이어야 합니다")
    if top_k <= 0:
        raise ValueError("top_k는 1 이상이어야 합니다")
    if idle_op_status_threshold < 0 or idle_op_status_threshold > 1:
        raise ValueError("idle_op_status_threshold는 0~1 범위여야 합니다")
    if force_exceed_margin_ratio <= 0 or force_exceed_margin_ratio >= 1:
        raise ValueError("force_exceed_margin_ratio는 0~1 범위(양 끝 제외)여야 합니다")

    device_ids = list_model_device_ids(exclude_warned=False)
    if not device_ids:
        raise ValueError("모델 장비가 없습니다")

    records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    threshold_start = time.perf_counter()
    threshold_map: Dict[str, float] = {}
    for device_id in device_ids:
        threshold = _load_device_threshold(device_id)
        if threshold is None:
            skipped.append({"device_id": device_id, "reason": "threshold 없음"})
            continue
        threshold_map[device_id] = float(threshold)
    logger.info(
        f"[MILP] threshold stage done: eligible={len(threshold_map)}, skipped={len(skipped)}, "
        f"elapsed={time.perf_counter() - threshold_start:.2f}s"
    )

    predict_start = time.perf_counter()
    pred_map: Dict[str, Dict[str, Any]] = {}
    pred_failed: Dict[str, str] = {}
    try:
        pred_map, pred_failed = _predict_devices_batched(
            device_ids=list(threshold_map.keys()),
            lookback_hours=lookback_hours,
        )
    except Exception as e:
        logger.warning(f"[MILP] batch prediction failed: {e}")

    for device_id, threshold in threshold_map.items():
        if device_id in pred_failed:
            skipped.append({"device_id": device_id, "reason": f"예측 실패: {pred_failed[device_id]}"})
            continue

        pred = pred_map.get(device_id)
        if pred is None:
            skipped.append({"device_id": device_id, "reason": "예측 결과 없음"})
            continue

        try:
            first = (pred.get("preds") or [{}])[0]
            p15 = float(first.get("y_15_pred", 0.0))
            p30 = float(first.get("y_30_pred", 0.0))
        except Exception as e:
            skipped.append({"device_id": device_id, "reason": f"예측 파싱 실패: {e}"})
            continue

        records.append(
            {
                "device_id": device_id,
                "threshold": float(threshold),
                "p15": p15,
                "p30": p30,
            }
        )

    logger.info(
        f"[MILP] prediction stage done: records={len(records)}, "
        f"elapsed={time.perf_counter() - predict_start:.2f}s"
    )

    if len(records) < 3:
        raise ValueError("최적화 가능한 장비가 3대 미만입니다")

    op_mean_map = _mean_op_status_by_device([r["device_id"] for r in records], lookback_hours=lookback_hours)
    for row in records:
        op_mean = float(op_mean_map.get(row["device_id"], 0.0))
        row["op_status_mean"] = op_mean
        row["is_idle"] = op_mean <= idle_op_status_threshold

    donor_candidates = [row for row in records if not row["is_idle"]]
    if not donor_candidates:
        donor_candidates = records.copy()

    donor_candidates.sort(key=lambda r: max(r["p15"], r["p30"]), reverse=True)
    max_donor_count = max(1, min(len(donor_candidates), len(records) - 1))
    effective_top_k = min(top_k, max_donor_count)
    donor_ids = [row["device_id"] for row in donor_candidates[:effective_top_k]]

    if force_exceed_demo:
        for row in records:
            if row["device_id"] not in donor_ids:
                continue
            base_peak = max(float(row["p15"]), float(row["p30"]))
            forced_threshold = base_peak * (1.0 - float(force_exceed_margin_ratio))
            row["threshold"] = min(float(row["threshold"]), forced_threshold)

    for row in records:
        row["is_donor"] = row["device_id"] in donor_ids

    n = len(records)
    horizons = [15, 30]
    h_count = len(horizons)

    # idle 장비 인덱스(이진변수 y를 둘 장비) 추출
    idle_indices = [i for i, row in enumerate(records) if row["is_idle"]]
    idle_pos = {idx: pos for pos, idx in enumerate(idle_indices)}

    # 결정변수 벡터를 1차원으로 평탄화해서 구성:
    # [u(i,h), v(i,h), s(i,h), z(h), y(i)]
    u_base = 0
    v_base = u_base + n * h_count
    s_base = v_base + n * h_count
    z_base = s_base + n * h_count
    y_base = z_base + h_count
    var_count = y_base + len(idle_indices)

    def u_idx(i: int, h: int) -> int:
        return u_base + i * h_count + h

    def v_idx(i: int, h: int) -> int:
        return v_base + i * h_count + h

    def s_idx(i: int, h: int) -> int:
        return s_base + i * h_count + h

    def z_idx(h: int) -> int:
        return z_base + h

    def y_idx(i: int) -> int:
        return y_base + idle_pos[i]

    p = np.array([[row["p15"], row["p30"]] for row in records], dtype=float)
    th = np.array([row["threshold"] for row in records], dtype=float)
    is_donor = np.array([bool(row["is_donor"]) for row in records], dtype=bool)

    # 목적함수 계수:
    # - z: 전체 피크 최소화
    # - s: 임계치 초과(slack) 강한 패널티
    # - y/u(idle): idle 장비 사용 최소화
    c = np.zeros(var_count, dtype=float)
    w_peak = 1.0
    w_slack = 1000.0
    w_idle_flag = 5.0
    w_idle_usage = 0.05

    for h in range(h_count):
        c[z_idx(h)] = w_peak

    for i in range(n):
        for h in range(h_count):
            c[s_idx(i, h)] = w_slack
            if i in idle_pos:
                c[u_idx(i, h)] += w_idle_usage

    for i in idle_indices:
        c[y_idx(i)] = w_idle_flag

    lb = np.zeros(var_count, dtype=float)
    ub = np.full(var_count, np.inf, dtype=float)

    for i in range(n):
        for h in range(h_count):
            if is_donor[i]:
                ub[u_idx(i, h)] = 0.0
                ub[v_idx(i, h)] = max(0.0, p[i, h])
            else:
                ub[v_idx(i, h)] = 0.0

    for i in idle_indices:
        ub[y_idx(i)] = 1.0

    # 선형 제약식 리스트
    constraints: List[LinearConstraint] = []

    # (1) 수급 균형: 시점별 총 유입(u) == 총 유출(v)
    for h in range(h_count):
        row = np.zeros(var_count, dtype=float)
        for i in range(n):
            row[u_idx(i, h)] = 1.0
            row[v_idx(i, h)] = -1.0
        constraints.append(LinearConstraint(row.reshape(1, -1), np.array([0.0]), np.array([0.0])))

    for i in range(n):
        for h in range(h_count):
            # (2) 임계치 제약 완화:
            # p - v + u <= threshold + slack
            row_threshold = np.zeros(var_count, dtype=float)
            row_threshold[u_idx(i, h)] = 1.0
            row_threshold[v_idx(i, h)] = -1.0
            row_threshold[s_idx(i, h)] = -1.0
            constraints.append(
                LinearConstraint(row_threshold.reshape(1, -1), -np.inf, np.array([th[i] - p[i, h]]))
            )

            # (3) 피크 상한 제약:
            # p - v + u <= z_h
            row_peak = np.zeros(var_count, dtype=float)
            row_peak[u_idx(i, h)] = 1.0
            row_peak[v_idx(i, h)] = -1.0
            row_peak[z_idx(h)] = -1.0
            constraints.append(LinearConstraint(row_peak.reshape(1, -1), -np.inf, np.array([-p[i, h]])))

            if is_donor[i]:
                # (4) donor 유출 한계:
                # donor는 자기 초과 필요량(exceed_need)을 넘겨서 내보내지 않음
                exceed_need = max(0.0, p[i, h] - th[i])
                row_donor_cap = np.zeros(var_count, dtype=float)
                row_donor_cap[v_idx(i, h)] = 1.0
                constraints.append(LinearConstraint(row_donor_cap.reshape(1, -1), -np.inf, np.array([exceed_need])))

    for i in idle_indices:
        for h in range(h_count):
            # (5) idle 활성 연계(big-M):
            # y_i=0이면 u(i,h)=0, y_i=1일 때만 유입 허용
            headroom = max(0.0, th[i] - p[i, h])
            donor_over = sum(max(0.0, p[j, h] - th[j]) for j in range(n) if is_donor[j])
            m_value = headroom + donor_over + 1.0
            row = np.zeros(var_count, dtype=float)
            row[u_idx(i, h)] = 1.0
            row[y_idx(i)] = -m_value
            constraints.append(LinearConstraint(row.reshape(1, -1), -np.inf, np.array([0.0])))

    integrality = np.zeros(var_count, dtype=int)
    for i in idle_indices:
        integrality[y_idx(i)] = 1

    milp_start = time.perf_counter()
    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=Bounds(lb=lb, ub=ub),
        options={"disp": False},
    )
    logger.info(f"[MILP] solve stage done: elapsed={time.perf_counter() - milp_start:.2f}s")

    solution = np.asarray(result.x if result.x is not None else np.zeros(var_count), dtype=float)

    devices_out: List[Dict[str, Any]] = []
    total_slack = 0.0

    for i, row in enumerate(records):
        u15, u30 = solution[u_idx(i, 0)], solution[u_idx(i, 1)]
        v15, v30 = solution[v_idx(i, 0)], solution[v_idx(i, 1)]
        s15, s30 = solution[s_idx(i, 0)], solution[s_idx(i, 1)]
        total_slack += float(s15 + s30)

        base15, base30 = row["p15"], row["p30"]
        opt15 = base15 - v15 + u15
        opt30 = base30 - v30 + u30

        devices_out.append(
            {
                "device_id": row["device_id"],
                "is_donor": bool(row["is_donor"]),
                "is_idle": bool(row["is_idle"]),
                "op_status_mean": float(row["op_status_mean"]),
                "threshold": float(row["threshold"]),
                "baseline_15": float(base15),
                "baseline_30": float(base30),
                "optimized_15": float(opt15),
                "optimized_30": float(opt30),
                "delta_15": float(opt15 - base15),
                "delta_30": float(opt30 - base30),
                "shift_in_15": float(u15),
                "shift_in_30": float(u30),
                "shift_out_15": float(v15),
                "shift_out_30": float(v30),
                "required_shift_15": float(max(0.0, base15 - row["threshold"])),
                "required_shift_30": float(max(0.0, base30 - row["threshold"])),
                "distributed_targets_15": [],
                "distributed_targets_30": [],
                "distribution_text": None,
                "slack_15": float(s15),
                "slack_30": float(s30),
            }
        )

    allocation_plan: List[Dict[str, Any]] = []
    out_by_device = {d["device_id"]: d for d in devices_out}

    def allocate_for_horizon(h_idx: int, minute: int) -> None:
        donors = []
        receivers = []
        for i, row in enumerate(records):
            out_power = float(solution[v_idx(i, h_idx)])
            in_power = float(solution[u_idx(i, h_idx)])
            if row["is_donor"] and out_power > 1e-9:
                donors.append([row["device_id"], out_power])
            if (not row["is_donor"]) and in_power > 1e-9:
                receivers.append([row["device_id"], in_power])

        donors.sort(key=lambda x: x[1], reverse=True)
        receivers.sort(key=lambda x: x[1], reverse=True)

        receiver_idx = 0
        for donor_device_id, donor_remain in donors:
            while donor_remain > 1e-9 and receiver_idx < len(receivers):
                receiver_device_id, receiver_need = receivers[receiver_idx]
                if receiver_need <= 1e-9:
                    receiver_idx += 1
                    continue

                moved = min(donor_remain, receiver_need)
                donor_remain -= moved
                receivers[receiver_idx][1] -= moved

                allocation_plan.append(
                    {
                        "minute": minute,
                        "from_device_id": donor_device_id,
                        "to_device_id": receiver_device_id,
                        "power_w": float(moved),
                    }
                )

                target_key = "distributed_targets_15" if minute == 15 else "distributed_targets_30"
                out_by_device[donor_device_id][target_key].append(
                    {
                        "device_id": receiver_device_id,
                        "power_w": float(moved),
                    }
                )

                if receivers[receiver_idx][1] <= 1e-9:
                    receiver_idx += 1

    allocate_for_horizon(h_idx=0, minute=15)
    allocate_for_horizon(h_idx=1, minute=30)

    for item in devices_out:
        if bool(item.get("is_donor")):
            item["distribution_text"] = _build_distribution_text_for_device(item, allocation_plan)
        else:
            item["distribution_text"] = "분배원 장비가 아니므로 추천 분배 문구 대상이 아닙니다."

    peak_15_before = float(np.max(p[:, 0]))
    peak_30_before = float(np.max(p[:, 1]))
    peak_15_after = float(max(d["optimized_15"] for d in devices_out))
    peak_30_after = float(max(d["optimized_30"] for d in devices_out))

    result_payload = {
        "status": "optimal" if bool(result.success) else "failed",
        "success": bool(result.success),
        "message": str(result.message),
        "lookback_hours": int(lookback_hours),
        "top_k": int(effective_top_k),
        "idle_op_status_threshold": float(idle_op_status_threshold),
        "force_exceed_demo": bool(force_exceed_demo),
        "force_exceed_margin_ratio": float(force_exceed_margin_ratio),
        "device_count": len(records),
        "donor_device_ids": donor_ids,
        "idle_device_ids": [row["device_id"] for row in records if row["is_idle"]],
        "peak_15_before": peak_15_before,
        "peak_15_after": peak_15_after,
        "peak_30_before": peak_30_before,
        "peak_30_after": peak_30_after,
        "objective_peak_sum": float(solution[z_idx(0)] + solution[z_idx(1)]),
        "total_slack": float(total_slack),
        "allocation_plan": allocation_plan,
        "devices": devices_out,
        "skipped_devices": skipped,
    }

    # 회사 매핑 정보가 아직 없더라도 실행/결과는 항상 저장한다.
    try:
        _save_peak_dispatch_result(result_payload=result_payload, customer_id=None)
    except Exception as e:
        logger.error(f"[MILP] 피크 분배 결과 저장 실패: {e}")

    logger.info(f"[MILP] total elapsed={time.perf_counter() - total_start:.2f}s")
    return result_payload
