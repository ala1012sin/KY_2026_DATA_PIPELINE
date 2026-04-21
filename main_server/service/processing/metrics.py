import math
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




# =========================
# 평가 함수들
# =========================
def _rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))

def _smape(y_true, y_pred, eps=1e-6) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)

def calc_metrics_2h(y_true_2d: np.ndarray, y_pred_2d: np.ndarray, names=("15", "30")) -> Dict[str, float]:
    out = {}
    maes, rmses, smapes, r2s = [], [], [], []
    for j, nm in enumerate(names):
        yt = y_true_2d[:, j]
        yp = y_pred_2d[:, j]
        mae_ = float(mean_absolute_error(yt, yp))
        rmse_ = _rmse(yt, yp)
        smape_ = _smape(yt, yp)
        r2_ = float(r2_score(yt, yp))
        out[f"MAE_{nm}m"] = mae_
        out[f"RMSE_{nm}m"] = rmse_
        out[f"sMAPE_{nm}m"] = smape_
        out[f"R2_{nm}m"] = r2_
        maes.append(mae_); rmses.append(rmse_); smapes.append(smape_); r2s.append(r2_)
    out["MAE_avg"] = float(np.mean(maes))
    out["RMSE_avg"] = float(np.mean(rmses))
    out["sMAPE_avg"] = float(np.mean(smapes))
    out["R2_avg"] = float(np.mean(r2s))
    return out
