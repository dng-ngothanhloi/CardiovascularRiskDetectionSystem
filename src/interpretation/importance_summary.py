"""
Build feature importance summaries for TabNet attention masks and Random Forest models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TabNet utilities
# ---------------------------------------------------------------------------

def _load_global_attention(path: Path) -> Optional[pd.DataFrame]:
    path = Path(path)
    if not path.exists():
        LOGGER.warning("Global attention CSV missing: %s", path)
        return None
    return pd.read_csv(path)


def _load_per_sample_attention(
    attention_dirs: Iterable[Path],
    feature_names: Iterable[str],
) -> Optional[pd.DataFrame]:
    frames = []
    feature_names = list(feature_names)
    for fold_dir in attention_dirs:
        sample_path = fold_dir / "attention_per_sample.npy"
        if not sample_path.exists():
            LOGGER.warning("Per-sample attention file missing: %s", sample_path)
            continue
        data = np.load(sample_path, allow_pickle=True).item()
        attention = data.get("attention")
        ids = data.get("ids")
        if attention is None or ids is None:
            LOGGER.warning("Malformed attention data at %s", sample_path)
            continue
        if attention.ndim == 3:
            per_sample = attention.mean(axis=2)
        else:
            per_sample = attention
        df = pd.DataFrame(per_sample, columns=feature_names)
        df.insert(0, "id", ids)
        df.insert(1, "fold", fold_dir.name)
        frames.append(df)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        LOGGER.info("Loaded per-sample attention for %d samples.", len(combined))
        return combined
    LOGGER.warning("No per-sample attention data available.")
    return None


def build_feature_importance_summary(
    global_csv: Path,
    attention_root: Path,
    feature_names: list[str],
    out_csv: Path,
    cohort_csv: Optional[Path] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Create long-form feature importance summary CSV for TabNet."""
    global_df = _load_global_attention(global_csv)
    if global_df is None:
        return None, None

    long_rows = []
    per_sample = _load_per_sample_attention(attention_root.glob("fold_*"), feature_names)

    global_rows = global_df[global_df["fold"] == "global"].copy()
    if global_rows.empty:
        LOGGER.warning("Global attention summary missing 'global' fold row.")
        return None, None

    for _, row in global_rows.iterrows():
        long_rows.append(
            {
                "feature": row["feature"],
                "scope": "global",
                "group": "all",
                "mean_attention": row.get("mean_attention"),
                "std_attention": row.get("std_attention"),
                "n_samples": row.get("n_samples"),
            }
        )

    cohort_summary = None
    if cohort_csv and per_sample is not None:
        cohort_csv = Path(cohort_csv)
        if cohort_csv.exists():
            cohort_df = pd.read_csv(cohort_csv)
            if "id" not in cohort_df.columns:
                cohort_df = cohort_df.reset_index().rename(columns={"index": "id"})
            merged = per_sample.merge(cohort_df, on="id", how="inner")
            cohort_cols = [col for col in cohort_df.columns if col != "id"]
            cohort_records = []
            for cohort_col in cohort_cols:
                for cohort_value, group_df in merged.groupby(cohort_col):
                    group_name = f"{cohort_col}={cohort_value}"
                    stats = group_df[feature_names].mean()
                    for feature, mean_val in stats.items():
                        cohort_records.append(
                            {
                                "feature": feature,
                                "scope": cohort_col,
                                "group": group_name,
                                "mean_attention": float(mean_val),
                                "std_attention": None,
                                "n_samples": len(group_df),
                            }
                        )
            if cohort_records:
                cohort_summary = pd.DataFrame(cohort_records)
                long_rows.extend(cohort_records)
        else:
            LOGGER.warning("Cohort CSV not found: %s", cohort_csv)

    if not long_rows:
        LOGGER.warning("No importance rows generated; skipping CSV write.")
        return None, cohort_summary

    summary_df = pd.DataFrame(long_rows)
    summary_df["model"] = "tabnet"
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    LOGGER.info("Wrote feature importance summary to %s", out_csv)
    return summary_df, cohort_summary


# ---------------------------------------------------------------------------
# Random Forest utilities
# ---------------------------------------------------------------------------

def _load_rf_feature_names(fe_dir: Path) -> Optional[list[str]]:
    fe_dir = Path(fe_dir)
    json_path = fe_dir / "selected_features_rf.json"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        features = data.get("kept_features") or data.get("selected_features")
        if features:
            return list(features)
    csv_path = fe_dir / "selected_features_rf.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df.columns.tolist()
    LOGGER.warning("Could not determine RF feature list in %s", fe_dir)
    return None


def build_rf_importance_summary(
    model_root: Path,
    fe_dir_rf: Path,
    out_csv: Path,
) -> Optional[pd.DataFrame]:
    """Aggregate Random Forest feature importances across folds."""
    model_root = Path(model_root)
    if not model_root.exists():
        LOGGER.warning("Random Forest model root not found: %s", model_root)
        return None

    feature_names = _load_rf_feature_names(fe_dir_rf)
    if not feature_names:
        LOGGER.warning("Random Forest feature names unavailable; skipping importance summary.")
        return None

    records = []
    for fold_dir in sorted(model_root.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        try:
            fold_num = int(fold_dir.name.split("_")[-1])
        except ValueError:
            LOGGER.warning("Unexpected RF fold directory: %s", fold_dir)
            continue
        fi_path = fold_dir / "feature_importance.json"
        if not fi_path.exists():
            LOGGER.warning("RF feature importance missing for %s", fi_path)
            continue
        with fi_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        importances = payload.get("feature_importance")
        if not importances or len(importances) != len(feature_names):
            LOGGER.warning("RF feature importance length mismatch in %s", fi_path)
            continue
        for feature, importance in zip(feature_names, importances):
            records.append(
                {
                    "feature": feature,
                    "fold": fold_num,
                    "importance": float(importance),
                }
            )

    if not records:
        LOGGER.warning("No Random Forest importance records collected.")
        return None

    df = pd.DataFrame(records)
    unique_folds = df["fold"].nunique()

    global_stats = (
        df.groupby("feature")["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_importance", "std": "std_importance"})
    )
    global_stats["scope"] = "global"
    global_stats["group"] = "all"
    global_stats["n_samples"] = unique_folds

    fold_stats = df.copy()
    fold_stats["scope"] = "fold"
    fold_stats["group"] = fold_stats["fold"].apply(lambda f: f"fold_{f}")
    fold_stats = fold_stats.rename(columns={"importance": "mean_importance"})
    fold_stats["std_importance"] = np.nan
    fold_stats["n_samples"] = np.nan

    combined = pd.concat(
        [
            global_stats[
                ["feature", "scope", "group", "mean_importance", "std_importance", "n_samples"]
            ],
            fold_stats[
                ["feature", "scope", "group", "mean_importance", "std_importance", "n_samples"]
            ],
        ],
        ignore_index=True,
    )
    combined["model"] = "rf"

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    LOGGER.info("Wrote Random Forest importance summary to %s", out_csv)
    return combined


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def write_importance_md(
    out_dir: Path,
    global_csv: Path,
    model_label: str,
    value_column: str,
    cohort_csv: Optional[Path] = None,
) -> Path:
    """Render Markdown summary for feature importance."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.md"

    global_csv = Path(global_csv)
    if global_csv.exists():
        df = pd.read_csv(global_csv)
        global_top = (
            df[df["scope"] == "global"]
            .sort_values(value_column, ascending=False)
            .head(5)
        )
    else:
        global_top = pd.DataFrame()

    lines = [
        f"# {model_label} Feature Importance — Global",
        "",
        f"Top-k global features by {value_column.replace('_', ' ')}:",
        "| Rank | Feature | Value |",
        "|----:|---------|------:|",
    ]

    if global_top.empty:
        lines.append("| 1 | - | - |")
    else:
        for idx, (_, row) in enumerate(global_top.iterrows(), start=1):
            lines.append(f"| {idx} | {row['feature']} | {row[value_column]:.6f} |")

    if cohort_csv and Path(cohort_csv).exists():
        lines.append("")
        lines.append("(Cohort breakdown shown if cohort file provided)")
    else:
        lines.append("")
        lines.append("(Cohort breakdown shown if cohort file provided)")

    with path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    LOGGER.info("Wrote %s importance markdown to %s", model_label, path)
    return path


def write_comparison_csv(
    tabnet_summary: Optional[pd.DataFrame],
    rf_summary: Optional[pd.DataFrame],
    out_csv: Path,
) -> Optional[Path]:
    """Create comparison CSV between TabNet and Random Forest global importances."""
    if tabnet_summary is None or rf_summary is None:
        return None

    tabnet_global = tabnet_summary[tabnet_summary["scope"] == "global"].copy()
    rf_global = rf_summary[rf_summary["scope"] == "global"].copy()

    if tabnet_global.empty or rf_global.empty:
        LOGGER.warning("Insufficient data to create comparison CSV.")
        return None

    tabnet_global = tabnet_global[["feature", "mean_attention"]].rename(
        columns={"mean_attention": "tabnet_mean_attention"}
    )
    rf_global = rf_global[["feature", "mean_importance"]].rename(
        columns={"mean_importance": "rf_mean_importance"}
    )
    comparison = tabnet_global.merge(rf_global, on="feature", how="outer")

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_csv, index=False)
    LOGGER.info("Wrote TabNet vs RF comparison to %s", out_csv)
    return out_csv


