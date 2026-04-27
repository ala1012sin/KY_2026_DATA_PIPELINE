"""Model prediction shared constants."""

try:
    import torch
except Exception:  # pragma: no cover - fallback when torch is unavailable
    torch = None


LOOKBACK = 12
DEVICE = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
ROLL_WINDOWS = [4, 12, 48, 96]
LAG_STEPS = list(range(1, 21))

AGG_RULES = {
    "CUR_VOLTAGE": "max",
    "AVGVOLTAGE": "mean",
    "AVGCURRENT": "mean",
    "FACTOR": "mean",
    "PRESSURE": "mean",
    "TEMPERATURE": "mean",
    "HZ": "mean",
    "OP_TIME": "sum",
    "OP_STATUS": "max",
    "CS_USAGE": "sum",
    "MG_REFILL": "sum",
    "UR_VOLT": "mean",
}