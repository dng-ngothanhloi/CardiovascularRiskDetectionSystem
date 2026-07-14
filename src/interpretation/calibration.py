"""
Calibration utilities for L2 interpretability layer.
"""

from __future__ import annotations

import logging
from typing import Literal, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.utils.metrics import brier_score, ece_score

LOGGER = logging.getLogger(__name__)

CalibMethod = Literal["platt", "isotonic"]


def fit_calibrator(p_val: np.ndarray, y_val: np.ndarray, method: CalibMethod) -> Tuple[str, object]:
    """Fit a calibration model on validation predictions."""
    y_val = np.asarray(y_val, dtype=float).reshape(-1)
    p_val = np.asarray(p_val, dtype=float).reshape(-1)
    if y_val.shape != p_val.shape:
        raise ValueError("Shapes of y_val and p_val must match for calibration.")

    if method == "platt":
        estimator = LogisticRegression(random_state=42, solver="lbfgs")
        estimator.fit(p_val.reshape(-1, 1), y_val)
    elif method == "isotonic":
        estimator = IsotonicRegression(out_of_bounds="clip")
        estimator.fit(p_val, y_val)
    else:
        raise ValueError(f"Unsupported calibration method: {method}")

    LOGGER.info("Calibrator fitted using %s method on %d samples.", method, len(p_val))
    return method, estimator


def apply_calibrator(p: np.ndarray, calibrator: object) -> np.ndarray:
    """Apply a fitted calibrator to probability predictions."""
    p = np.asarray(p, dtype=float).reshape(-1)
    if hasattr(calibrator, "predict_proba"):
        calibrated = calibrator.predict_proba(p.reshape(-1, 1))[:, 1]
    elif hasattr(calibrator, "predict"):
        calibrated = calibrator.predict(p)
    else:
        raise TypeError(f"Calibrator of type {type(calibrator)} is unsupported.")

    return np.clip(calibrated, 0.0, 1.0)


def compute_brier_ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> dict:
    """Compute calibration metrics."""
    score_brier = brier_score(y, p)
    score_ece = ece_score(y, p, n_bins=n_bins)
    return {"brier": float(score_brier), "ece": float(score_ece)}


