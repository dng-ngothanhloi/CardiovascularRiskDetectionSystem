"""
Utility functions for IO operations used across the interpretation layer.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _normalise_oof_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to the expected schema."""
    col_map = {
        "y": "y_true",
        "target": "y_true",
        "label": "y_true",
        "oof": "y_pred",
        "pred": "y_pred",
        "prediction": "y_pred",
    }
    df = df.copy()
    lower_cols = {col.lower(): col for col in df.columns}

    for raw_col, canonical in col_map.items():
        if canonical in df.columns:
            continue
        if raw_col in df.columns:
            df.rename(columns={raw_col: canonical}, inplace=True)
            continue
        if raw_col in lower_cols:
            df.rename(columns={lower_cols[raw_col]: canonical}, inplace=True)

    if "y_pred" not in df.columns and "prob" in lower_cols:
        df.rename(columns={lower_cols["prob"]: "y_pred"}, inplace=True)

    if "y_true" not in df.columns and "outcome" in lower_cols:
        df.rename(columns={lower_cols["outcome"]: "y_true"}, inplace=True)

    if "id" not in df.columns:
        LOGGER.warning("OOF CSV missing 'id' column; filling with index.")
        df["id"] = df.index.astype(str)
    if "fold" not in df.columns:
        LOGGER.warning("OOF CSV missing 'fold' column; using -1 placeholder.")
        df["fold"] = -1
    if "model" not in df.columns:
        LOGGER.warning("OOF CSV missing 'model' column; inferring from folder.")
    return df


def read_oof_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read an out-of-fold predictions CSV file with flexible schema."""
    path = Path(path)
    if not path.exists():
        LOGGER.warning("OOF CSV not found at %s", path)
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        LOGGER.error("Failed to read OOF CSV at %s: %s", path, exc)
        return None

    if df.empty:
        LOGGER.warning("OOF CSV at %s is empty.", path)
        return None

    df = _normalise_oof_columns(df)
    missing = {"y_true", "y_pred"} - set(df.columns)
    if missing:
        LOGGER.error("OOF CSV missing required columns: %s", ", ".join(sorted(missing)))
        return None

    return df


def write_json(obj: dict, path: Path) -> None:
    """Persist a JSON serialisable object to the given path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2, sort_keys=True)


def read_json(path: Path) -> dict:
    """Read a JSON file and return its contents."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_pickle(obj: Any, path: Path) -> None:
    """Serialise an object to disk using pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file)


def load_pickle(path: Path) -> Any:
    """Load a pickled object."""
    path = Path(path)
    with path.open("rb") as file:
        return pickle.load(file)


