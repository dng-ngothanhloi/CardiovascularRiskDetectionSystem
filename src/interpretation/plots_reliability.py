"""
Reliability plotting helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.metrics import reliability_bins

LOGGER = logging.getLogger(__name__)


def _plot_reliability(ax, bins: pd.DataFrame, label: str, color: str) -> None:
    ax.plot(bins["bin_confidence"], bins["bin_accuracy"], marker="o", label=label, color=color)
    ax.bar(
        bins["bin_confidence"],
        bins["bin_accuracy"],
        width=1.0 / len(bins),
        color=color,
        alpha=0.15,
        align="center",
    )


def save_reliability(
    y: np.ndarray,
    p_pre: np.ndarray,
    p_post: np.ndarray,
    out_pre_path: Path,
    out_post_path: Path,
    n_bins: int = 15,
) -> None:
    """Save reliability plots for pre and post calibration probabilities."""
    out_pre_path = Path(out_pre_path)
    out_post_path = Path(out_post_path)
    out_pre_path.parent.mkdir(parents=True, exist_ok=True)
    out_post_path.parent.mkdir(parents=True, exist_ok=True)

    bins_pre = reliability_bins(y, p_pre, n_bins=n_bins)
    bins_post = reliability_bins(y, p_post, n_bins=n_bins)
    diagonal = np.linspace(0, 1, 100)

    for bins, path, title in [
        (bins_pre, out_pre_path, "Reliability (Pre)"),
        (bins_post, out_post_path, "Reliability (Post)"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 6))
        _plot_reliability(ax, bins, label="Empirical", color="#1f77b4")
        ax.plot(diagonal, diagonal, linestyle="--", color="#444444", label="Ideal")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        LOGGER.info("Saved reliability plot to %s", path)


def save_reliability_overlay(
    model_to_df: Dict[str, pd.DataFrame],
    out_path: Path,
    n_bins: int = 15,
) -> Optional[Path]:
    """Create overlay reliability plot comparing models."""
    if not model_to_df:
        LOGGER.warning("No models supplied for reliability overlay plot.")
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diagonal = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap("tab10")

    for idx, (model, df) in enumerate(sorted(model_to_df.items())):
        if df.empty:
            LOGGER.warning("Model %s dataframe empty; skipping overlay.", model)
            continue
        bins = reliability_bins(df["y_true"].to_numpy(), df["y_pred"].to_numpy(), n_bins=n_bins)
        color = cmap(idx % 10)
        ax.plot(bins["bin_confidence"], bins["bin_accuracy"], marker="o", label=model, color=color)

    if not ax.lines:
        LOGGER.warning("No valid model data for overlay plot.")
        plt.close(fig)
        return None

    ax.plot(diagonal, diagonal, linestyle="--", color="#444444", label="Ideal")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability Overlay")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    LOGGER.info("Saved reliability overlay plot to %s", out_path)
    return out_path


