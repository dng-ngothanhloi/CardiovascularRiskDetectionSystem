"""
CLI entry point to run calibration and reliability reporting.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .calibration import CalibMethod, apply_calibrator, compute_brier_ece, fit_calibrator
from .plots_reliability import save_reliability, save_reliability_overlay
from .report_calibration import write_l2_summary_md, write_model_calib_report_md
from src.utils.io import read_oof_csv, save_pickle, write_json

LOGGER = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def _prepare_oof(model_name: str, model_dir: Path) -> Optional[pd.DataFrame]:
    path = model_dir / "oof_predictions.csv"
    df = read_oof_csv(path)
    if df is None:
        LOGGER.warning("Skipping calibration; OOF predictions missing for %s.", model_name)
        return None
    if "model" not in df.columns:
        df["model"] = model_name
    if df["model"].isnull().all():
        df["model"] = model_name
    return df


def _fold_directories(out_dir: Path, fold: int) -> Path:
    fold_dir = out_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    return fold_dir


def _calibrate_fold(
    df_fold: pd.DataFrame,
    method: CalibMethod,
    n_bins: int,
    fold_dir: Path,
) -> Tuple[np.ndarray, dict, dict]:
    y = df_fold["y_true"].to_numpy()
    p = df_fold["y_pred"].to_numpy()

    metrics_pre = compute_brier_ece(y, p, n_bins=n_bins)
    _, calibrator = fit_calibrator(p, y, method)
    p_post = apply_calibrator(p, calibrator)
    metrics_post = compute_brier_ece(y, p_post, n_bins=n_bins)

    save_pickle(calibrator, fold_dir / "calibrator.pkl")
    write_json(metrics_pre, fold_dir / "metrics_before.json")
    write_json(metrics_post, fold_dir / "metrics_after.json")

    save_reliability(
        y=y,
        p_pre=p,
        p_post=p_post,
        out_pre_path=fold_dir / "reliability_before.png",
        out_post_path=fold_dir / "reliability_after.png",
        n_bins=n_bins,
    )

    fold_meta = {
        "method": method,
        "rel_pre": "fold_{}/reliability_before.png".format(df_fold["fold"].iloc[0]),
        "rel_post": "fold_{}/reliability_after.png".format(df_fold["fold"].iloc[0]),
    }
    write_json(fold_meta, fold_dir / "fold_meta.json")
    return p_post, metrics_pre, metrics_post


def _aggregate_predictions(rows: List[pd.DataFrame]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["id", "fold", "y_true", "y_pred", "model"])
    combined = pd.concat(rows, ignore_index=True)
    return combined[["id", "fold", "y_true", "y_pred", "model"]]


def _collect_existing_per_fold(calibration_root: Path) -> List[dict]:
    rows: List[dict] = []
    for model_dir in calibration_root.glob("*"):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for fold_dir in model_dir.glob("fold_*"):
            if not fold_dir.is_dir():
                continue
            try:
                fold = int(fold_dir.name.split("_")[-1])
            except ValueError:
                continue
            metrics_before = fold_dir / "metrics_before.json"
            metrics_after = fold_dir / "metrics_after.json"
            meta_path = fold_dir / "fold_meta.json"

            if not metrics_before.exists() or not metrics_after.exists():
                LOGGER.warning("Metrics JSONs missing for %s fold %s", model_name, fold)
                rows.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "method": "-",
                        "brier_pre": None,
                        "ece_pre": None,
                        "brier_post": None,
                        "ece_post": None,
                        "rel_pre": "-",
                        "rel_post": "-",
                    }
                )
                continue

            with metrics_before.open("r", encoding="utf-8") as file:
                before = json.load(file)
            with metrics_after.open("r", encoding="utf-8") as file:
                after = json.load(file)
            meta = {}
            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as file:
                    meta = json.load(file)

            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "method": meta.get("method", "-"),
                    "brier_pre": before.get("brier"),
                    "ece_pre": before.get("ece"),
                    "brier_post": after.get("brier"),
                    "ece_post": after.get("ece"),
                    "rel_pre": meta.get("rel_pre", "-"),
                    "rel_post": meta.get("rel_post", "-"),
                }
            )
    return rows


def _collect_aggregates(
    calibration_root: Path,
    model_root: Path,
    n_bins: int,
) -> List[dict]:
    aggregates: List[dict] = []
    model_names = {
        d.name for d in model_root.glob("*") if d.is_dir()
    } | {d.name for d in calibration_root.glob("*") if d.is_dir()}

    for model_name in sorted(model_names):
        model_dir = calibration_root / model_name
        oof_pre = read_oof_csv(model_root / model_name / "oof_predictions.csv")
        oof_post_path = model_dir / "oof_calibrated.csv"
        if oof_pre is None or not oof_post_path.exists():
            aggregates.append(
                {
                    "model": model_name,
                    "brier_pre": None,
                    "ece_pre": None,
                    "brier_post": None,
                    "ece_post": None,
                    "delta_brier": None,
                    "delta_ece": None,
                }
            )
            continue
        oof_post = pd.read_csv(oof_post_path)
        metrics_pre = compute_brier_ece(oof_pre["y_true"].to_numpy(), oof_pre["y_pred"].to_numpy(), n_bins=n_bins)
        metrics_post = compute_brier_ece(oof_post["y_true"].to_numpy(), oof_post["y_pred"].to_numpy(), n_bins=n_bins)
        aggregates.append(
            {
                "model": model_name,
                "brier_pre": metrics_pre["brier"],
                "ece_pre": metrics_pre["ece"],
                "brier_post": metrics_post["brier"],
                "ece_post": metrics_post["ece"],
                "delta_brier": metrics_post["brier"] - metrics_pre["brier"],
                "delta_ece": metrics_post["ece"] - metrics_pre["ece"],
            }
        )
    return aggregates


def _update_overlays(
    calibration_root: Path,
    model_root: Path,
    n_bins: int,
) -> Dict[str, str]:
    model_to_pre: Dict[str, pd.DataFrame] = {}
    model_to_post: Dict[str, pd.DataFrame] = {}

    model_names = {
        d.name for d in model_root.glob("*") if d.is_dir()
    } | {d.name for d in calibration_root.glob("*") if d.is_dir()}

    for model_name in sorted(model_names):
        model_dir = calibration_root / model_name
        df_pre = read_oof_csv(model_root / model_name / "oof_predictions.csv")
        post_path = model_dir / "oof_calibrated.csv"
        if df_pre is not None and not df_pre.empty:
            model_to_pre[model_name] = df_pre[["y_true", "y_pred"]]
        if post_path.exists():
            df_post = pd.read_csv(post_path)
            if not df_post.empty:
                model_to_post[model_name] = df_post[["y_true", "y_pred"]]

    plots_dir = calibration_root.parent / "reports" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    overlay_pre = save_reliability_overlay(model_to_pre, plots_dir / "reliability_overlay_pre.png", n_bins=n_bins)
    overlay_post = save_reliability_overlay(model_to_post, plots_dir / "reliability_overlay_post.png", n_bins=n_bins)
    return {
        "pre": str(overlay_pre.relative_to(calibration_root.parent)) if overlay_pre else "-",
        "post": str(overlay_post.relative_to(calibration_root.parent)) if overlay_post else "-",
    }


def run_calibration(
    model: str,
    method: CalibMethod,
    model_root: Path,
    out_root: Path,
    n_bins: int,
) -> None:
    model_dir = model_root / model
    if not model_dir.exists():
        LOGGER.error("Model directory not found: %s", model_dir)
        return

    df = _prepare_oof(model, model_dir)
    if df is None:
        return

    out_model_dir = out_root / model
    out_model_dir.mkdir(parents=True, exist_ok=True)

    per_fold_rows: List[dict] = []
    calibrated_frames: List[pd.DataFrame] = []

    folds = sorted(df["fold"].unique())
    LOGGER.info("Processing %d folds for model %s.", len(folds), model)

    for fold in folds:
        df_fold = df[df["fold"] == fold].copy()
        if df_fold.empty:
            LOGGER.warning("Fold %s for model %s has no rows; skipping.", fold, model)
            continue
        fold_dir = _fold_directories(out_model_dir, fold)
        try:
            p_post, metrics_pre, metrics_post = _calibrate_fold(df_fold, method, n_bins, fold_dir)
        except Exception as exc:
            LOGGER.error("Failed to calibrate fold %s for model %s: %s", fold, model, exc)
            continue

        df_fold_post = df_fold.copy()
        df_fold_post["y_pred"] = p_post
        calibrated_frames.append(df_fold_post)
        per_fold_rows.append(
            {
                "model": model,
                "fold": fold,
                "method": method,
                "brier_pre": metrics_pre["brier"],
                "ece_pre": metrics_pre["ece"],
                "brier_post": metrics_post["brier"],
                "ece_post": metrics_post["ece"],
                "rel_pre": f"fold_{fold}/reliability_before.png",
                "rel_post": f"fold_{fold}/reliability_after.png",
            }
        )

    if not calibrated_frames:
        LOGGER.warning("No calibrated predictions generated for %s.", model)
        return

    calibrated_df = _aggregate_predictions(calibrated_frames)
    calibrated_df.to_csv(out_model_dir / "oof_calibrated.csv", index=False)
    LOGGER.info("Saved calibrated OOF predictions for %s.", model)

    write_model_calib_report_md(out_model_dir, model, per_fold_rows)

    aggregates = _collect_aggregates(out_root, model_root, n_bins)
    per_fold_all = _collect_existing_per_fold(out_root)
    plot_paths = _update_overlays(out_root, model_root, n_bins)
    write_l2_summary_md(out_root.parent / "reports" / "L2_summary.md", aggregates, per_fold_all, plot_paths)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run calibration workflow.")
    parser.add_argument("--model", required=True, help="Model name (e.g., rf, tabnet).")
    parser.add_argument("--model_root", type=Path, default=Path("artifacts/Model"))
    parser.add_argument("--out_root", type=Path, default=Path("artifacts/Interpretation/calibration"))
    parser.add_argument("--method", choices=["platt", "isotonic"], required=True)
    parser.add_argument("--n_bins", type=int, default=15)
    return parser.parse_args(args=args)


def main(cli_args: Optional[List[str]] = None) -> None:
    _setup_logging()
    args = parse_args(cli_args)
    run_calibration(
        model=args.model,
        method=args.method,
        model_root=args.model_root,
        out_root=args.out_root,
        n_bins=args.n_bins,
    )


if __name__ == "__main__":
    main(sys.argv[1:])


