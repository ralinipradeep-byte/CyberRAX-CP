"""Validation-only probability calibration and reliability-adaptive fusion."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.special import expit, logit


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)

class PlattCalibrator:
    """Platt scaling fitted exclusively on validation predictions."""
    def __init__(self):
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self.fitted = False
    def fit(self, raw_probability, y_validation):
        p = _clip(raw_probability)
        x = logit(p).reshape(-1, 1)
        y = np.asarray(y_validation, dtype=int)
        if np.unique(y).size < 2:
            raise ValueError("Platt calibration requires both classes in validation data")
        self.model.fit(x, y)
        self.fitted = True
        return self
    def predict(self, raw_probability):
        if not self.fitted:
            raise RuntimeError("Calibrator has not been fitted")
        p = _clip(raw_probability)
        return self.model.predict_proba(logit(p).reshape(-1, 1))[:, 1]
    def to_dict(self):
        if not self.fitted:
            return {"fitted": False}
        return {"fitted": True, "coef": self.model.coef_.ravel().tolist(), "intercept": self.model.intercept_.tolist()}


def expected_calibration_error(y_true, p, n_bins: int = 15):
    y = np.asarray(y_true, dtype=int); p = _clip(p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p >= edges[i]) & (p < edges[i+1] if i < n_bins-1 else p <= edges[i+1])
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def normalized_binary_confidence(p):
    p = _clip(p)
    entropy = -(p*np.log(p) + (1-p)*np.log(1-p)) / np.log(2.0)
    return np.clip(1.0-entropy, 0.0, 1.0)


def fit_branch_reliability(y_validation, calibrated_p, n_bins: int = 15, epsilon: float = 1e-6):
    ece = expected_calibration_error(y_validation, calibrated_p, n_bins=n_bins)
    reliability = 1.0/(ece+epsilon)
    return {"ece_validation": float(ece), "reliability": float(reliability), "epsilon": float(epsilon)}


def adaptive_fusion(p_t, p_d, reliability_t: float, reliability_d: float, epsilon: float = 1e-6):
    p_t, p_d = _clip(p_t), _clip(p_d)
    c_t, c_d = normalized_binary_confidence(p_t), normalized_binary_confidence(p_d)
    raw_t, raw_d = float(reliability_t)*c_t, float(reliability_d)*c_d
    denom = raw_t + raw_d
    w_t = np.where(denom <= epsilon, 0.5, raw_t/np.maximum(denom, epsilon))
    w_d = np.where(denom <= epsilon, 0.5, raw_d/np.maximum(denom, epsilon))
    p_f = w_t*p_t + w_d*p_d
    return p_f, w_t, w_d, c_t, c_d


def select_threshold(y_validation, p_validation):
    """Select F1-maximizing threshold on validation only; test labels are never passed here."""
    from sklearn.metrics import f1_score
    y = np.asarray(y_validation, dtype=int); p = _clip(p_validation)
    candidates = np.unique(np.r_[np.linspace(.05, .95, 91), np.quantile(p, np.linspace(0.0, 1.0, min(101, len(p))))])
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        score = f1_score(y, p >= t, zero_division=0)
        if score > best_f1 or (score == best_f1 and abs(t-.5) < abs(best_t-.5)):
            best_t, best_f1 = float(t), float(score)
    return best_t, best_f1
