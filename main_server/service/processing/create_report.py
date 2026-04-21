# 각 장비별 평균 사용량, 오차값, 오차율, 제외 사유
import glob
import os
import numpy as np
import pandas as pd


def build_device_level_table(
    out_dir="./ai_models/current",
    min_usage_mean_w=10.0,
    min_nonzero_ratio=0.10,
    min_usage_std_w=1.0,
    smape_max=200.0,
    r2_min=-5.0,
):
    # ---------------------------
    # 1) metrics 로드 (best_model + MAE_15m/MAE_30m + (있으면 sMAPE/R2/RMSE))
    # ---------------------------
    p_all = os.path.join(out_dir, "report_best_models_ALL_metrics.csv")
    p_best = os.path.join(out_dir, "best_models_summary.csv")

    if os.path.exists(p_all):
        mdf = pd.read_csv(p_all)
    elif os.path.exists(p_best):
        mdf = pd.read_csv(p_best)
    else:
        raise FileNotFoundError("out_train에 report_best_models_ALL_metrics.csv 또는 best_models_summary.csv가 없습니다.")

    # device 컬럼 보정
    if "device" not in mdf.columns:
        for cand in ["DEVICE_ID", "device_id", "SERIAL_NO", "serial_no"]:
            if cand in mdf.columns:
                mdf = mdf.rename(columns={cand: "device"})
                break

    # best_model 컬럼 보정
    if "best_model" not in mdf.columns:
        for cand in ["bestModel", "model"]:
            if cand in mdf.columns:
                mdf = mdf.rename(columns={cand: "best_model"})
                break

    # 숫자형 변환
    for c in mdf.columns:
        if c not in ("device", "best_model"):
            mdf[c] = pd.to_numeric(mdf[c], errors="coerce")

    # 최소 필요 컬럼 체크
    for c in ["device", "best_model", "MAE_15m", "MAE_30m"]:
        if c not in mdf.columns:
            raise ValueError(f"필수 컬럼 누락: {c} (현재 컬럼: {list(mdf.columns)})")

    # ---------------------------
    # 2) device별 usage 로드 (best_predictions_test.csv)
    # ---------------------------
    rows = []
    pattern = os.path.join(out_dir, "device=*/best_predictions_test.csv")
    for path in glob.glob(pattern):
        dev = os.path.basename(os.path.dirname(path)).split("device=")[-1]
        dfp = pd.read_csv(path)

        y15 = pd.to_numeric(dfp.get("y_15_true"), errors="coerce") if "y_15_true" in dfp.columns else None
        y30 = pd.to_numeric(dfp.get("y_30_true"), errors="coerce") if "y_30_true" in dfp.columns else None

        if y15 is not None and y30 is not None and y15.notna().any() and y30.notna().any():
            yref = 0.5 * (y15 + y30)           # 15/30 true의 평균을 "대표 사용량"으로 사용
        elif y15 is not None and y15.notna().any():
            yref = y15
        elif y30 is not None and y30.notna().any():
            yref = y30
        else:
            continue

        yref = yref.dropna().astype(float)
        if len(yref) == 0:
            continue

        rows.append({
            "device": dev,
            "n_test_points": int(len(yref)),
            "usage_mean_w": float(yref.mean()),
            "usage_std_w": float(yref.std(ddof=1)) if len(yref) > 1 else 0.0,
            "nonzero_ratio": float((yref != 0).mean()),
        })

    udf = pd.DataFrame(rows)
    if udf.empty:
        raise ValueError("best_predictions_test.csv를 못 찾았어요. (cfg.save_predictions_test=True였는지 확인)")

    # ---------------------------
    # 3) merge + 오차율 계산
    # ---------------------------
    df = mdf.merge(udf, on="device", how="left")

    eps = 1e-9
    df["NMAE_15_pct"] = df["MAE_15m"] / (df["usage_mean_w"] + eps) * 100.0
    df["NMAE_30_pct"] = df["MAE_30m"] / (df["usage_mean_w"] + eps) * 100.0

    # ---------------------------
    # 4) (선택) 필터 PASS/EXCLUDE 플래그 만들기
    # ---------------------------
    # usage 기반
    alive = (
        df["usage_mean_w"].notna() &
        (df["usage_mean_w"] >= min_usage_mean_w) &
        (df["nonzero_ratio"] >= min_nonzero_ratio) &
        (df["usage_std_w"] >= min_usage_std_w)
    )

    # sMAPE/R2 기반(컬럼 있을 때만)
    ok_smape = pd.Series(True, index=df.index)
    smape_cols = [c for c in df.columns if c.lower().startswith("smape")]
    if smape_cols:
        ok_smape = (df[smape_cols].ge(0).all(axis=1)) & (df[smape_cols].le(smape_max).all(axis=1))

    ok_r2 = pd.Series(True, index=df.index)
    r2_cols = [c for c in df.columns if c.lower().startswith("r2")]
    if r2_cols:
        ok_r2 = (df[r2_cols].le(1.0001).all(axis=1)) & (df[r2_cols].ge(r2_min).all(axis=1))

    df["pass_filter"] = alive & ok_smape & ok_r2

    # 제외 사유(한눈에)
    reasons = []
    for i, r in df.iterrows():
        rs = []
        if not pd.notna(r.get("usage_mean_w", np.nan)):
            rs.append("no_test_usage")
        else:
            if r["usage_mean_w"] < min_usage_mean_w: rs.append("usage_mean<min")
            if r["nonzero_ratio"] < min_nonzero_ratio: rs.append("nonzero_ratio<min")
            if r["usage_std_w"] < min_usage_std_w: rs.append("usage_std<min")
        if smape_cols:
            if not ((df.loc[i, smape_cols].ge(0).all()) and (df.loc[i, smape_cols].le(smape_max).all())):
                rs.append("smape_outlier")
        if r2_cols:
            if not ((df.loc[i, r2_cols].le(1.0001).all()) and (df.loc[i, r2_cols].ge(r2_min).all())):
                rs.append("r2_outlier")
        reasons.append(",".join(rs) if rs else "")
    df["exclude_reason"] = reasons

    # ---------------------------
    # 5) 보기 좋은 컬럼만 정리
    # ---------------------------
    base_cols = [
        "device", "best_model",
        "usage_mean_w", "usage_std_w", "nonzero_ratio", "n_test_points",
        "MAE_15m", "MAE_30m", "NMAE_15_pct", "NMAE_30_pct",
    ]

    # 있으면 같이 보여주면 좋은 지표들
    extra_candidates = ["RMSE_15m", "RMSE_30m", "R2_15m", "R2_30m", "sMAPE_15m", "sMAPE_30m"]
    extra_cols = [c for c in extra_candidates if c in df.columns]

    tail_cols = ["pass_filter", "exclude_reason"]

    out_cols = [c for c in (base_cols + extra_cols + tail_cols) if c in df.columns]
    out = df[out_cols].copy()

    # 정렬: 일단 pass 먼저, 그 다음 오차율 작은 순
    out = out.sort_values(["pass_filter", "NMAE_15_pct"], ascending=[False, True])

    # 요약도 같이
    n_total = out["device"].nunique()
    n_pass = out.loc[out["pass_filter"] == True, "device"].nunique()
    print(f"Devices: total={n_total}, pass_filter={n_pass}, excluded={n_total - n_pass}")

    return out


if __name__ == "__main__":
    device_table = build_device_level_table("./ai_models/current")
    print(device_table.head(30))