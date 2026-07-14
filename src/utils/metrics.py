"""
Metrics utilities for calibration evaluation.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _clip_probs(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, 0.0, 1.0)
    if not np.allclose(clipped, p):
        LOGGER.warning("Probabilities clipped to the [0, 1] range.")
    return clipped


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    """Compute the Brier score for probabilistic predictions."""
    p = _clip_probs(np.asarray(p, dtype=float).reshape(-1))
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.shape != p.shape:
        raise ValueError("Shapes of y and p must match.")
    return float(np.mean((p - y) ** 2))


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error using equal-width probability bins."""
    bins = reliability_bins(y, p, n_bins=n_bins)
    weighted_abs_diff = bins["bin_count"] * np.abs(bins["bin_accuracy"] - bins["bin_confidence"])
    total = bins["bin_count"].sum()
    return float((weighted_abs_diff.sum() / max(total, 1)).item())


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    """Compute reliability diagram bin statistics."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    p = _clip_probs(np.asarray(p, dtype=float).reshape(-1))
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.shape != p.shape:
        raise ValueError("Shapes of y and p must match.")

    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = np.minimum((df["p"] * n_bins).astype(int), n_bins - 1)

    grouped = df.groupby("bin", observed=True)
    summaries = grouped.agg(
        bin_count=("y", "size"),
        bin_accuracy=("y", "mean"),
        bin_confidence=("p", "mean"),
    ).reset_index()
    summaries["bin_lower"] = summaries["bin"] / n_bins
    summaries["bin_upper"] = (summaries["bin"] + 1) / n_bins
    return summaries[["bin", "bin_lower", "bin_upper", "bin_count", "bin_accuracy", "bin_confidence"]]


