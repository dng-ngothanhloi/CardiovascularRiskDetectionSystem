"""
CLI to extract TabNet attention masks and summarise feature importance.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .attention_tabnet import extract_attention_masks, summarise_masks
from .importance_summary import (
    build_feature_importance_summary,
    build_rf_importance_summary,
    write_comparison_csv,
    write_importance_md,
)
from .report_importance import write_importance_report

LOGGER = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def _load_feature_spec(fe_dir: Path) -> List[str]:
    json_path = fe_dir / "selected_features_tabnet.json"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        features = data.get("kept_features") or data.get("selected_features") or []
        if features:
            return features
    csv_path = fe_dir / "selected_features_tabnet.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df.columns.tolist()
    raise FileNotFoundError(f"Could not determine TabNet feature list in {fe_dir}")


def _load_training_data(fe_dir: Path, feature_names: List[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_path = fe_dir / "Train_data_tabnet.csv"
    df = pd.read_csv(train_path)
    missing = [col for col in feature_names if col not in df.columns]
    if missing:
        raise ValueError(f"Train data missing expected features: {missing}")

    target_candidates = [col for col in df.columns if col not in feature_names]
    target_col = "cardio"
    if target_col not in df.columns and target_candidates:
        target_col = target_candidates[-1]
    if target_col not in df.columns:
        raise ValueError(f"Unable to determine target column in {train_path}")

    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy()
    ids = df.index.to_numpy()
    return X, y, ids


def _load_run_metadata(model_root: Path) -> tuple[int, int]:
    results_path = model_root / "results.json"
    seed = 42
    n_folds = 5
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as file:
            results = json.load(file)
        n_folds = results.get("n_folds", n_folds)
    run_info_path = model_root / "run_info.json"
    if run_info_path.exists():
        with run_info_path.open("r", encoding="utf-8") as file:
            info = json.load(file)
        seed = info.get("seed", seed)
    return int(n_folds), int(seed)


def run_attention_summary(
    fe_dir_tabnet: Path,
    model_root: Path,
    out_root: Path,
    cohort_csv: Optional[Path] = None,
    rf_model_root: Optional[Path] = None,
    fe_dir_rf: Optional[Path] = None,
    include_rf: bool = True,
) -> None:
    feature_names = _load_feature_spec(fe_dir_tabnet)
    X, y, ids = _load_training_data(fe_dir_tabnet, feature_names)
    n_folds, seed = _load_run_metadata(model_root)

    LOGGER.info("Loaded training data: samples=%d, features=%d", X.shape[0], X.shape[1])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_indices = [val_idx for _, val_idx in skf.split(X, y)]

    attention_dir = out_root / "attention" / "tabnet"
    importance_dir = out_root / "importance" / "tabnet"
    attention_dir.mkdir(parents=True, exist_ok=True)
    importance_dir.mkdir(parents=True, exist_ok=True)

    fold_records: List[dict] = []
    per_sample_blocks: List[np.ndarray] = []

    for fold, fold_dir in enumerate(sorted(model_root.glob("fold_*"))):
        try:
            fold_num = int(fold_dir.name.split("_")[-1])
        except ValueError:
            LOGGER.warning("Skipping unexpected directory: %s", fold_dir)
            continue

        if fold_num >= len(fold_indices):
            LOGGER.warning("Fold index %s exceeds available splits; skipping.", fold_num)
            continue

        val_idx = fold_indices[fold_num]
        X_val = X[val_idx]
        ids_val = ids[val_idx]

        masks = extract_attention_masks(fold_dir, X_val)
        if masks is None:
            LOGGER.warning("No masks extracted for fold %s.", fold_num)
            continue

        per_sample, mean_mask = summarise_masks(masks)
        per_sample_blocks.append(per_sample)

        fold_out = attention_dir / f"fold_{fold_num}"
        fold_out.mkdir(parents=True, exist_ok=True)

        np.save(
            fold_out / "attention_per_sample.npy",
            {"ids": ids_val, "attention": masks, "feature_names": feature_names},
            allow_pickle=True,
        )
        np.save(fold_out / "feature_mask_mean.npy", mean_mask)

        for feature, value in zip(feature_names, mean_mask):
            fold_records.append(
                {
                    "fold": f"fold_{fold_num}",
                    "feature": feature,
                    "mean_attention": float(value),
                    "std_attention": None,
                    "n_samples": len(ids_val),
                }
            )

    if not fold_records:
        LOGGER.warning("No attention masks extracted; skipping summary generation.")
        return

    combined_samples = np.vstack(per_sample_blocks)
    global_mean = combined_samples.mean(axis=0)
    global_std = combined_samples.std(axis=0)
    total_samples = combined_samples.shape[0]

    for feature, mean_val, std_val in zip(feature_names, global_mean, global_std):
        fold_records.append(
            {
                "fold": "global",
                "feature": feature,
                "mean_attention": float(mean_val),
                "std_attention": float(std_val),
                "n_samples": total_samples,
            }
        )

    global_df = pd.DataFrame(fold_records)
    global_csv_path = attention_dir / "global_attention.csv"
    global_df.to_csv(global_csv_path, index=False)
    LOGGER.info("Saved global attention summary to %s", global_csv_path)

    summary_csv = importance_dir / "feature_importance_summary.csv"
    tabnet_summary_df, cohort_summary_df = build_feature_importance_summary(
        global_csv=global_csv_path,
        attention_root=attention_dir,
        feature_names=feature_names,
        out_csv=summary_csv,
        cohort_csv=cohort_csv,
    )

    write_importance_md(
        importance_dir,
        summary_csv,
        model_label="TabNet",
        value_column="mean_attention",
        cohort_csv=cohort_csv,
    )

    rf_summary_df = None
    rf_summary_csv: Optional[Path] = None
    rf_summary_dir = out_root / "importance" / "rf"
    if include_rf:
        rf_root_resolved = Path(rf_model_root) if rf_model_root is not None else Path("artifacts/Model/rf")
        fe_dir_rf_resolved = Path(fe_dir_rf) if fe_dir_rf is not None else Path("artifacts/fe/random_forest")
        if rf_root_resolved.exists():
            rf_summary_dir.mkdir(parents=True, exist_ok=True)
            rf_summary_csv = rf_summary_dir / "feature_importance_summary.csv"
            rf_summary_df = build_rf_importance_summary(
                model_root=rf_root_resolved,
                fe_dir_rf=fe_dir_rf_resolved,
                out_csv=rf_summary_csv,
            )
            if rf_summary_df is not None:
                write_importance_md(
                    rf_summary_dir,
                    rf_summary_csv,
                    model_label="Random Forest",
                    value_column="mean_importance",
                )
        else:
            LOGGER.warning("Random Forest model root not found at %s; skipping RF importance.", rf_root_resolved)

    comparison_path = write_comparison_csv(
        tabnet_summary_df,
        rf_summary_df,
        out_root / "importance" / "comparison_tabnet_rf.csv",
    )

    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    def _rel(path: Optional[Path]) -> Optional[Path]:
        if path is None:
            return None
        try:
            return path.relative_to(out_root)
        except ValueError:
            return path

    write_importance_report(
        reports_dir,
        tabnet_global_csv=_rel(global_csv_path if global_csv_path.exists() else None),
        tabnet_summary_csv=_rel(summary_csv if summary_csv.exists() else None),
        cohort_csv=_rel(cohort_csv if cohort_csv and cohort_csv.exists() else None),
        rf_summary_csv=_rel(rf_summary_csv if rf_summary_csv and rf_summary_csv.exists() else None),
        comparison_csv=_rel(comparison_path) if comparison_path else None,
    )


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run attention-derived importance pipeline (TabNet + optional Random Forest)."
    )
    parser.add_argument("--fe_dir_tabnet", type=Path, required=True)
    parser.add_argument("--model_root", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, default=Path("artifacts/Interpretation"))
    parser.add_argument("--cohort_csv", type=Path)
    parser.add_argument("--rf_model_root", type=Path, default=Path("artifacts/Model/rf"))
    parser.add_argument("--fe_dir_rf", type=Path, default=Path("artifacts/fe/random_forest"))
    parser.add_argument("--skip_rf", action="store_true", help="Skip Random Forest importance aggregation.")
    return parser.parse_args(args=args)


def main(cli_args: Optional[List[str]] = None) -> None:
    _setup_logging()
    args = parse_args(cli_args)
    run_attention_summary(
        fe_dir_tabnet=args.fe_dir_tabnet,
        model_root=args.model_root,
        out_root=args.out_root,
        cohort_csv=args.cohort_csv,
        rf_model_root=None if args.skip_rf else args.rf_model_root,
        fe_dir_rf=None if args.skip_rf else args.fe_dir_rf,
        include_rf=not args.skip_rf,
    )


if __name__ == "__main__":
    main(sys.argv[1:])


