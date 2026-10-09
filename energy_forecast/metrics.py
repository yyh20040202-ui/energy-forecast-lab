import numpy as np


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    if truth.shape != prediction.shape or truth.size == 0:
        raise ValueError("Truth/prediction must have equal, nonempty shapes")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require finite values")
    error = prediction.astype(np.float64) - truth
    denominator = np.abs(truth) + np.abs(prediction)
    ratio = np.divide(2 * np.abs(error), denominator,
                      out=np.zeros_like(error), where=denominator > 1e-8)
    return {"mae_wh": float(np.abs(error).mean()),
            "rmse_wh": float(np.sqrt((error ** 2).mean())),
            "smape_percent": float(100 * ratio.mean())}
