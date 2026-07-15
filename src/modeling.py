"""
Stage 3 - Modeling: Stratified K-Fold CV with HPO, Calibration, and Evaluation.
"""
import argparse
import numpy as np
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
import traceback
import sys
from sklearn.model_selection import StratifiedKFold

from .utils import (
    load_config, set_all_seeds, setup_logging, ensure_dir,
    load_numpy, load_json, load_joblib, save_json, save_joblib,
    out_dir_fold, write_run_info,
    compute_all_metrics, fit_calibrator, find_best_threshold,
    plot_pr_roc, plot_reliability
)
from .hpo import hpo_search
from .models.random_forest import RandomForestTrainer
from .models.tabnet_model import TabNetTrainer
from .models.ensemble_joint import JointVAETabNetTrainer

logger = logging.getLogger(__name__)


_NON_HYPERPARAM_KEYS = frozenset({
    'input_dim', 'seed', 'random_state', 'n_jobs', 'verbose', 'warm_start',
})


def _sanitize_model_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Drop non-hyperparameter keys leaked into saved param dicts."""
    return {k: v for k, v in params.items() if k not in _NON_HYPERPARAM_KEYS}


def _aggregate_resolved_params(fold_params: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate fold resolved params via median (numeric) / mode (categorical).

    Fallback strategy when fold metrics are unavailable for best-fold selection.
    """
    if not fold_params:
        raise ValueError("No fold params to aggregate")

    keys = set()
    for p in fold_params:
        keys.update(p.keys())

    aggregated: Dict[str, Any] = {}
    for key in sorted(keys):
        if key in _NON_HYPERPARAM_KEYS:
            continue
        values = [p[key] for p in fold_params if key in p]
        if not values:
            continue

        if all(v is None for v in values):
            aggregated[key] = None
            continue

        non_null = [v for v in values if v is not None]
        if not non_null:
            aggregated[key] = None
            continue

        if all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) for v in non_null):
            median_val = float(np.median(np.asarray(non_null, dtype=float)))
            # Keep ints as ints when every fold used ints for that key
            if all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in non_null):
                aggregated[key] = int(round(median_val))
            else:
                aggregated[key] = median_val
        else:
            # Mode (most common), tie-break by first occurrence order
            counts: Dict[Any, int] = {}
            for v in non_null:
                counts[v] = counts.get(v, 0) + 1
            aggregated[key] = max(counts.items(), key=lambda kv: kv[1])[0]

    return aggregated


def _select_best_fold_params(
    fold_resolved_params: List[Dict[str, Any]],
    fold_metrics_list: List[Dict[str, Any]],
    metric_key: str = 'pr_auc',
) -> tuple:
    """
    Select resolved params from the CV fold with best validation metric.

    Train-only selection (no Test). Returns (params, selected_fold, selection_meta).
    Falls back to median/mode aggregate if metrics are missing.
    """
    if not fold_resolved_params:
        raise ValueError("No fold params to select from")

    n = min(len(fold_resolved_params), len(fold_metrics_list))
    scores: List[Optional[float]] = []
    for i in range(n):
        metrics = fold_metrics_list[i]
        if not metrics or metric_key not in metrics:
            scores.append(None)
            continue
        try:
            scores.append(float(metrics[metric_key]))
        except (TypeError, ValueError):
            scores.append(None)

    valid = [(i, s) for i, s in enumerate(scores) if s is not None]
    if not valid:
        params = _sanitize_model_params(_aggregate_resolved_params(fold_resolved_params))
        meta = {
            'param_strategy': 'median_mode_aggregate_fold_params',
            'param_strategy_note': (
                'Fallback: fold val metrics unavailable; used median/mode aggregate. '
                'No Test data used.'
            ),
            'selection_metric': metric_key,
            'fold_val_scores': scores,
            'selected_fold': None,
        }
        return params, None, meta

    selected_fold, best_score = max(valid, key=lambda t: t[1])
    params = _sanitize_model_params(dict(fold_resolved_params[selected_fold]))
    meta = {
        'param_strategy': 'best_fold_val_pr_auc' if metric_key == 'pr_auc' else f'best_fold_val_{metric_key}',
        'param_strategy_note': (
            f'Selected resolved params from fold with highest validation {metric_key}. '
            'No Test data used for selection.'
        ),
        'selection_metric': metric_key,
        'selected_fold': int(selected_fold),
        'selected_fold_val_score': float(best_score),
        'fold_val_scores': [None if s is None else float(s) for s in scores],
    }
    return params, int(selected_fold), meta


def _confusion_matrix_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'matrix': cm.tolist(),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }

def _resolve_fe_dir(cfg: Dict[str, Any], model_name: str) -> Path:
    """
    Resolve feature engineering directory based on model name.
    
    Args:
        cfg: Configuration dictionary
        model_name: Model name ('rf', 'tabnet', 'vae_tabnet')
    
    Returns:
        Path to FE directory
    
    Note:
        - 'tabnet' and 'vae_tabnet' use paths.fe_dir_tabnet if available, else paths.fe_dir
        - 'rf' uses paths.fe_dir
        - vae_tabnet expects FE features consistent with TabNet's feature space
    """
    paths = cfg.get('paths', {})
    
    if model_name in {'tabnet', 'vae_tabnet'}:
        # Use tabnet-specific FE directory if configured
        fe_dir = paths.get('fe_dir_tabnet') or paths.get('fe_dir', 'artifacts/fe')
    else:
        # RF uses default FE directory
        fe_dir = paths.get('fe_dir', 'artifacts/fe')
    
    return Path(fe_dir)


def load_fe_artifacts(fe_dir: str, model_name: str, target_col: str = "cardio", test_scoring: bool = False) -> tuple:
    """
    Load feature engineering artifacts from model-specific folders.
    
    Args:
        fe_dir: Base feature engineering directory
        model_name: Model name ('rf', 'tabnet', or 'vae_tabnet')
        target_col: Target column name
        test_scoring: Whether test scoring is enabled
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor, selected_features, fe_inputs_info)
        
    Note:
        - 'rf' uses artifacts from artifacts/fe/random_forest/
        - 'tabnet' and 'vae_tabnet' use artifacts from artifacts/fe/tabnet/
    """
    # Determine model-specific paths
    # Check if fe_dir is already model-specific (ends with model subdirectory)
    fe_dir_path = Path(fe_dir)
    if fe_dir_path.name in ['random_forest', 'tabnet']:
        # Already model-specific, use directly
        model_fe_dir = fe_dir_path
    else:
        # Append model-specific subdirectory
        if model_name == 'rf':
            model_fe_dir = fe_dir_path / 'random_forest'
        elif model_name in ['tabnet', 'vae_tabnet']:
            model_fe_dir = fe_dir_path / 'tabnet'
        else:
            raise ValueError(f"Model-specific FE paths not defined for model: {model_name}")
    
    # Determine file names based on model
    if model_name == 'rf':
        features_json = model_fe_dir / 'selected_features_rf.json'
        features_csv = model_fe_dir / 'selected_features_rf.csv'
        train_csv = model_fe_dir / 'Train_data_rf.csv'
        test_csv = model_fe_dir / 'Test_data_rf.csv'
    elif model_name in ['tabnet', 'vae_tabnet']:
        features_json = model_fe_dir / 'selected_features_tabnet.json'
        features_csv = model_fe_dir / 'selected_features_tabnet.csv'
        train_csv = model_fe_dir / 'Train_data_tabnet.csv'
        test_csv = model_fe_dir / 'Test_data_tabnet.csv'
    else:
        raise ValueError(f"Model-specific FE paths not defined for model: {model_name}")
    
    # Validate required files exist
    missing_files = []
    if not features_json.exists():
        missing_files.append(str(features_json))
    if not features_csv.exists():
        missing_files.append(str(features_csv))
    if not train_csv.exists():
        missing_files.append(str(train_csv))
    if test_scoring and not test_csv.exists():
        missing_files.append(str(test_csv))
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing required FE artifacts for model '{model_name}':\n" +
            "\n".join(f"  - {f}" for f in missing_files) +
            f"\n\nPlease run feature engineering with --model-type {model_name} first."
        )
    
    # Log input data contract
    logger.info("=" * 60)
    logger.info(f"Loading FE artifacts for model: {model_name}")
    logger.info("=" * 60)
    logger.info(f"Features JSON: {features_json}")
    logger.info(f"Features CSV: {features_csv}")
    logger.info(f"Train CSV: {train_csv}")
    if test_scoring:
        logger.info(f"Test CSV: {test_csv}")
    logger.info("=" * 60)
    
    # Load selected features list
    selected_features_data = load_json(str(features_json))
    if 'kept_features' in selected_features_data:
        selected_features_list = selected_features_data['kept_features']
    else:
        # Fallback for different JSON structure
        selected_features_list = selected_features_data.get('selected_features', [])
    
    n_features = len(selected_features_list)
    logger.info(f"Selected features count: {n_features}")
    
    # Load train data
    train_df = pd.read_csv(train_csv)
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {train_csv}")
    
    sample_ids = None
    if "sample_id" in train_df.columns:
        sample_ids = train_df["sample_id"].values
        train_df = train_df.drop(columns=["sample_id"])
    
    # Ensure feature columns match selected_features_list (order and presence)
    feature_cols = [col for col in train_df.columns if col != target_col]
    if set(feature_cols) != set(selected_features_list):
        missing = set(selected_features_list) - set(feature_cols)
        extra = set(feature_cols) - set(selected_features_list)
        if missing:
            logger.warning(f"Missing features in train CSV: {missing}")
        if extra:
            logger.warning(f"Extra features in train CSV: {extra}")
        # Use intersection to proceed, but warn
        valid_features = [f for f in selected_features_list if f in feature_cols]
        if len(valid_features) != n_features:
            raise ValueError(
                f"Feature mismatch: expected {n_features} features, "
                f"found {len(valid_features)} matching features in {train_csv}"
            )
        selected_features_list = valid_features
    
    # Reorder columns to match selected_features_list order
    X_train = train_df[selected_features_list].values
    y_train = train_df[target_col].values
    
    # Load test data (if available and needed)
    if test_scoring and test_csv.exists():
        test_df = pd.read_csv(test_csv)
        if target_col not in test_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in {test_csv}")
        if "sample_id" in test_df.columns:
            test_df = test_df.drop(columns=["sample_id"])
        
        # Ensure test data has same features as train
        test_feature_cols = [col for col in test_df.columns if col != target_col]
        if set(test_feature_cols) != set(selected_features_list):
            missing = set(selected_features_list) - set(test_feature_cols)
            if missing:
                raise ValueError(f"Missing features in test CSV: {missing}")
        
        X_test = test_df[selected_features_list].values
        y_test = test_df[target_col].values
    else:
        X_test = None
        y_test = None
    
    # Preprocessor is not needed for model-specific approach (data already processed)
    preprocessor = None
    
    # Build fe_inputs info for run_info.json
    fe_inputs_info = {
        "features_json": str(features_json),
        "features_csv": str(features_csv),
        "train_csv": str(train_csv),
        "test_csv": str(test_csv) if (test_scoring and test_csv.exists()) else None,
        "n_features": n_features
    }
    
    logger.info(f"Loaded FE artifacts: X_train={X_train.shape}, y_train={y_train.shape}")
    if X_test is not None:
        logger.info(f"X_test={X_test.shape}, y_test={y_test.shape}")
    logger.info(f"y_train distribution: {np.bincount(y_train.astype(int))}")
    
    return X_train, X_test, y_train, y_test, preprocessor, selected_features_data, fe_inputs_info, sample_ids


def build_model(
    model_name: str,
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    input_dim: int,
    seed: int,
    joint: bool = False
):
    """
    Build model trainer instance.
    
    Args:
        model_name: Model name ('rf', 'tabnet', 'vae_tabnet')
        params: Model parameters
        cfg: Configuration dictionary
        input_dim: Input feature dimension
        seed: Random seed
        joint: For vae_tabnet, whether to use joint training
    
    Returns:
        Model trainer instance
    """
    if model_name == 'rf':
        return RandomForestTrainer(params, seed=seed)
    elif model_name == 'tabnet':
        early_stopping = cfg.get('modeling', {}).get('early_stopping', {})
        return TabNetTrainer(
            input_dim=input_dim,
            params=params,
            seed=seed,
            monitor=early_stopping.get('monitor', 'val_pr_auc'),
            patience=early_stopping.get('patience', 15),
            min_delta=early_stopping.get('min_delta', 0.0005)
        )
    elif model_name == 'vae_tabnet':
        early_stopping = cfg.get('modeling', {}).get('early_stopping', {})
        return JointVAETabNetTrainer(
            input_dim=input_dim,
            params=params,
            seed=seed,
            joint_training=joint,
            monitor=early_stopping.get('monitor', 'val_pr_auc'),
            patience=early_stopping.get('patience', 15),
            min_delta=early_stopping.get('min_delta', 0.0005)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_experiment(
    cfg: Dict[str, Any],
    model_name: str,
    joint: bool = False
) -> None:
    """
    Run modeling experiment with Stratified K-Fold CV.
    
    Args:
        cfg: Configuration dictionary
        model_name: Model name ('rf', 'tabnet', 'vae_tabnet')
        joint: For vae_tabnet, whether to use joint training
    """
    # Setup model-specific log file
    log_dir = Path('src/logging')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file name based on model name and joint flag
    # Log files will be saved to src/logging/:
    #   - rf → rf_training.log
    #   - tabnet → tabnet_training.log
    #   - vae_tabnet → vae_tabnet_training.log
    #   - vae_tabnet --joint → vae_tabnet_joint_training.log
    if model_name == 'vae_tabnet' and joint:
        log_file = str(log_dir / 'vae_tabnet_joint_training.log')
    elif model_name == 'vae_tabnet':
        log_file = str(log_dir / 'vae_tabnet_training.log')
    elif model_name == 'rf':
        log_file = str(log_dir / 'rf_training.log')
    elif model_name == 'tabnet':
        log_file = str(log_dir / 'tabnet_training.log')
    else:
        # Fallback for any other model names
        log_file = str(log_dir / f'{model_name}_training.log')
    
    # Setup logging with model-specific log file
    setup_logging(log_level='INFO', log_file=log_file)
    
    # Initialize timing and error tracking
    start_time = datetime.now()
    timing_info = {
        'start_time': start_time.isoformat(),
        'model_name': model_name,
        'joint_training': joint,
        'status': 'running',
        'error': None,
        'error_traceback': None
    }
    
    # Log start
    logger.info("=" * 70)
    logger.info(f"STARTING MODELING EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Joint Training: {joint}")
    logger.info(f"Log File: {log_file}")
    logger.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    try:
        # Set seeds
        seed = cfg.get('random_seed', 42)
        set_all_seeds(seed, deterministic_tf=False)
        
        # Get target column
        target_col = cfg.get('preprocessing', {}).get('target_col') or cfg.get('preprocessing', {}).get('target', 'cardio')
        
        # Get test_scoring flag
        test_scoring = cfg.get('modeling', {}).get('test_scoring', False)
        
        # Resolve FE directory based on model name
        fe_dir_path = _resolve_fe_dir(cfg, model_name)
        logger.info(f"Using FE directory for {model_name}: {fe_dir_path}")
        
        # Load FE artifacts from resolved directory
        X_train, X_test, y_train, y_test, preprocessor, selected_features, fe_inputs_info, sample_ids = load_fe_artifacts(
            str(fe_dir_path), model_name, target_col, test_scoring
        )
        
        # Validate shapes
        assert X_train.shape[0] == y_train.shape[0], "X_train and y_train shape mismatch"
        if X_test is not None:
            assert X_test.shape[0] == y_test.shape[0], "X_test and y_test shape mismatch"
            assert X_train.shape[1] == X_test.shape[1], "Feature dimension mismatch"
        
        input_dim = X_train.shape[1]
        logger.info(f"Input dimension: {input_dim}")
        
        # Log shapes based on model type
        if model_name == 'rf':
            logger.info(f"RandomForest mode: X_train.shape = {X_train.shape}")
        elif model_name == 'tabnet':
            logger.info(f"TabNet mode: X_train.shape = {X_train.shape}")
        elif model_name == 'vae_tabnet':
            logger.info(f"VAE-TabNet mode: X_train.shape = {X_train.shape} -> will encode to z (latent_dim from config)")
        
        # Setup output directory
        model_dir = Path(cfg['paths']['model_dir'])
        model_output_dir = model_dir / model_name
        if joint:
            model_output_dir = model_dir / f"{model_name}_joint"
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        cv_split_dir = model_output_dir / "cv_splits"
        cv_split_dir.mkdir(parents=True, exist_ok=True)
        all_fold_histories: List[pd.DataFrame] = []
        final_epoch_records: List[Dict[str, float]] = []
        per_fold_metrics_records: List[Dict[str, float]] = []
        
        # Write run info with fe_inputs
        write_run_info(cfg, model_name, model_output_dir, fe_inputs_info=fe_inputs_info)
        
        # Stratified K-Fold CV
        cv_folds = cfg.get('evaluation', {}).get('cv_folds', 5)
        kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        
        # Storage for OOF predictions
        oof_predictions = np.full(len(y_train), np.nan, dtype=float)
        oof_records = []
        fold_metrics_list = []
        calibrators = []
        fold_resolved_params: List[Dict[str, Any]] = []
        test_preds_per_fold: List[np.ndarray] = []
        
        logger.info(f"Starting {cv_folds}-fold Stratified CV for {model_name}")
        
        # Per-fold loop
        for fold, (idx_tr, idx_va) in enumerate(kf.split(X_train, y_train)):
            fold_start_time = datetime.now()
            logger.info(f"=" * 60)
            logger.info(f"Fold {fold + 1}/{cv_folds}")
            logger.info(f"Fold Start Time: {fold_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"=" * 60)
            
            # Split data
            X_tr_fold = X_train[idx_tr]
            X_va_fold = X_train[idx_va]
            y_tr_fold = y_train[idx_tr]
            y_va_fold = y_train[idx_va]
            
            logger.info(f"Train: {len(X_tr_fold)}, Val: {len(X_va_fold)}")
            
            # Persist CV split indices for traceability
            try:
                train_split = pd.DataFrame({
                    "fold": np.full(len(idx_tr), fold, dtype=int),
                    "set": np.full(len(idx_tr), "train"),
                    "oof_index": idx_tr.astype(int),
                })
                val_split = pd.DataFrame({
                    "fold": np.full(len(idx_va), fold, dtype=int),
                    "set": np.full(len(idx_va), "val"),
                    "oof_index": idx_va.astype(int),
                })
                if sample_ids is not None:
                    train_split["sample_id"] = sample_ids[idx_tr].astype(int)
                    val_split["sample_id"] = sample_ids[idx_va].astype(int)
                split_df = pd.concat([train_split, val_split], ignore_index=True)
                split_path = cv_split_dir / f"fold_{fold}_split.csv"
                split_df.to_csv(split_path, index=False)
                logger.info(f"Saved CV split indices to {split_path}")
            except Exception as exc:
                logger.warning(f"Failed to persist CV split indices for fold {fold}: {exc}")
            
            # Output directory for this fold
            out_fold = out_dir_fold(cfg, model_name, fold)
            
            # HPO
            logger.info("Running HPO...")
            hpo_result = hpo_search(
                model_name=model_name,
                X_tr=X_tr_fold,
                y_tr=y_tr_fold,
                X_va=X_va_fold,
                y_va=y_va_fold,
                cfg=cfg,
                seed=seed + fold,  # Vary seed per fold
                joint=joint
            )
            raw_params = hpo_result['raw_optuna_params']
            resolved_params = hpo_result['resolved_model_params']
            fold_resolved_params.append(resolved_params)
            
            # Save raw + resolved params
            save_json(raw_params, out_fold / 'hpo_raw_params.json')
            save_json(resolved_params, out_fold / 'resolved_model_params.json')
            # Backward-compatible alias
            save_json(resolved_params, out_fold / 'params.json')
            logger.info(f"Raw Optuna params: {raw_params}")
            logger.info(f"Resolved model params: {resolved_params}")
            
            # Build and fit model
            logger.info("Building and training model...")
            tensorboard_dir = out_fold / 'tensorboard' if model_name in ['tabnet', 'vae_tabnet'] else None
            
            model = build_model(model_name, resolved_params, cfg, input_dim, seed + fold, joint)
            
            # Log input shapes before training
            if model_name == 'rf':
                logger.info(f"RandomForest training: X_tr.shape = {X_tr_fold.shape}, X_va.shape = {X_va_fold.shape}")
            elif model_name == 'tabnet':
                logger.info(f"TabNet training: X_tr.shape = {X_tr_fold.shape}, X_va.shape = {X_va_fold.shape}")
            elif model_name == 'vae_tabnet':
                logger.info(f"VAE-TabNet training: X_tr.shape = {X_tr_fold.shape} -> will encode to z_tr")
            
            if model_name in ['tabnet', 'vae_tabnet']:
                model.fit(X_tr_fold, y_tr_fold, X_va_fold, y_va_fold, tensorboard_dir)
            else:
                model.fit(X_tr_fold, y_tr_fold, X_va_fold, y_va_fold)
            
            # Capture per-epoch training history
            history_df = None
            history_obj = getattr(model, "history", None)
            if history_obj is not None and hasattr(history_obj, "history"):
                history_dict = history_obj.history
                if history_dict:
                    try:
                        history_df = pd.DataFrame(history_dict)
                        history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
                        history_df.insert(0, "fold", fold)
                        history_path = out_fold / "training_history.csv"
                        history_df.to_csv(history_path, index=False)
                        logger.info(f"Saved training history ({len(history_df)} epochs) to {history_path}")
                        all_fold_histories.append(history_df)
                        final_row = history_df.iloc[-1]
                        final_metrics = {
                            k: round(float(final_row[k]), 4)
                            for k in history_df.columns
                            if k not in {"fold", "epoch"}
                        }
                        logger.info(f"Fold {fold} final epoch metrics: {final_metrics}")
                        final_record = {"fold": fold, "epoch": int(final_row["epoch"])}
                        for col in history_df.columns:
                            if col not in {"fold", "epoch"}:
                                final_record[col] = float(final_row[col])
                        final_epoch_records.append(final_record)
                    except Exception as exc:
                        logger.warning(f"Failed to persist training history for fold {fold}: {exc}")
            
            # Predict on validation
            p_va = model.predict_proba(X_va_fold)
            
            # Calibration
            calibration_cfg = cfg.get('modeling', {}).get('calibration', {})
            calibrator = None
            if calibration_cfg.get('enabled', True):
                logger.info("Calibrating predictions...")
                calibrator = fit_calibrator(
                    p_va, y_va_fold,
                    method=calibration_cfg.get('method', 'platt')
                )
                p_va = calibrator.transform(p_va)
                calibrators.append(calibrator)
                
                # Save calibrator
                save_joblib(calibrator, str(out_fold / 'calibrator.pkl'))
            else:
                calibrators.append(None)
            
            # Compute metrics
            fold_metrics = compute_all_metrics(y_va_fold, p_va)
            fold_metrics_list.append(fold_metrics)
            
            # Save metrics
            save_json(fold_metrics, out_fold / 'metrics.json')
            logger.info(f"Fold {fold + 1} metrics: PR-AUC={fold_metrics['pr_auc']:.4f}, ROC-AUC={fold_metrics['roc_auc']:.4f}")
            fold_metrics_record = {"fold": fold}
            for key, value in fold_metrics.items():
                try:
                    fold_metrics_record[key] = float(value)
                except (TypeError, ValueError):
                    continue
            fold_metrics_record["n_train_fold"] = int(len(X_tr_fold))
            fold_metrics_record["n_val_fold"] = int(len(X_va_fold))
            per_fold_metrics_records.append(fold_metrics_record)
            
            # Generate plots
            logger.info("Generating plots...")
            plot_pr_roc(y_va_fold, p_va, out_fold)
            plot_reliability(y_va_fold, p_va, out_fold)
            
            # Save model
            model.save(out_fold)
            
            # Hard assertions for file existence based on model type
            if model_name == 'vae_tabnet':
                if joint:
                    # Joint training: model.h5 and scaler.pkl must exist
                    model_path = out_fold / 'model.h5'
                    scaler_path = out_fold / 'scaler.pkl'
                    assert model_path.exists(), f"model.h5 must exist in vae_tabnet joint mode at {model_path}"
                    assert scaler_path.exists(), f"scaler.pkl must exist in vae_tabnet joint mode at {scaler_path}"
                    logger.info(f"[OK] Verified joint model files exist: model.h5, scaler.pkl")
                else:
                    # Two-stage training: encoder, decoder, and classifier must exist
                    encoder_path = out_fold / 'encoder.h5'
                    decoder_path = out_fold / 'decoder.h5'
                    classifier_path = out_fold / 'classifier.h5'
                    assert encoder_path.exists(), f"encoder.h5 must exist in vae_tabnet two-stage mode at {encoder_path}"
                    assert decoder_path.exists(), f"decoder.h5 must exist in vae_tabnet two-stage mode at {decoder_path}"
                    assert classifier_path.exists(), f"classifier.h5 must exist in vae_tabnet two-stage mode at {classifier_path}"
                    logger.info(f"[OK] Verified two-stage model files exist: encoder.h5, decoder.h5, classifier.h5")
                
            # Save latent representations for downstream interpretation
            # Both two-stage and joint modes can extract latent z
            if hasattr(model, 'get_latent'):
                z_va_saved = model.get_latent(X_va_fold)
                if z_va_saved is not None:
                    z_va_path = out_fold / 'z_val.npy'
                    np.save(z_va_path, z_va_saved)
                    logger.info(f"Saved latent z_val to {z_va_path} (shape: {z_va_saved.shape}, joint={joint})")
            elif model_name == 'tabnet':
                # TabNet must NOT have encoder/decoder
                encoder_path = out_fold / 'encoder.h5'
                decoder_path = out_fold / 'decoder.h5'
                assert not encoder_path.exists(), f"encoder.h5 must NOT exist in tabnet mode (found at {encoder_path})"
                assert not decoder_path.exists(), f"decoder.h5 must NOT exist in tabnet mode (found at {decoder_path})"
                logger.info(f"[OK] Verified no VAE files in tabnet mode")
            elif model_name == 'rf':
                # RandomForest must NOT have encoder/decoder (it's not a deep learning model)
                encoder_path = out_fold / 'encoder.h5'
                decoder_path = out_fold / 'decoder.h5'
                assert not encoder_path.exists(), f"encoder.h5 must NOT exist in rf mode (found at {encoder_path})"
                assert not decoder_path.exists(), f"decoder.h5 must NOT exist in rf mode (found at {decoder_path})"
                # Verify model.joblib exists for RF
                model_path = out_fold / 'model.joblib'
                assert model_path.exists(), f"model.joblib must exist in rf mode at {model_path}"
                logger.info(f"[OK] Verified RF model.joblib exists, no VAE files")
            
            # Store OOF predictions
            oof_predictions[idx_va] = p_va
            for local_idx, global_idx in enumerate(idx_va):
                oof_records.append(
                    {
                        "id": int(global_idx),
                        "sample_id": int(sample_ids[global_idx]) if sample_ids is not None else int(global_idx),
                        "fold": int(fold),
                        "model": model_name,
                        "y_true": float(y_train[global_idx]),
                        "y_pred": float(p_va[local_idx]),
                    }
                )

            # Per-fold held-out test scoring (ensemble inputs)
            if test_scoring and X_test is not None:
                p_test_fold = model.predict_proba(X_test)
                if calibrator is not None:
                    p_test_fold = calibrator.transform(p_test_fold)
                test_preds_per_fold.append(np.asarray(p_test_fold, dtype=float))
                fold_test_df = pd.DataFrame({
                    "id": np.arange(len(y_test), dtype=int),
                    "y_true": y_test.astype(int),
                    "y_pred": p_test_fold.astype(float),
                    "fold": fold,
                    "model": model_name,
                })
                fold_test_path = out_fold / "test_predictions.csv"
                fold_test_df.to_csv(fold_test_path, index=False)
                logger.info(f"Saved fold {fold} test predictions to {fold_test_path}")
            
            fold_end_time = datetime.now()
            fold_duration = (fold_end_time - fold_start_time).total_seconds()
            logger.info(f"Fold {fold + 1} completed")
            logger.info(f"Fold Duration: {fold_duration:.2f} seconds ({fold_duration/60:.2f} minutes)")

        # OOF integrity checks
        assert not np.isnan(oof_predictions).any(), "OOF has unfilled (NaN) entries"
        assert len(oof_records) == len(y_train), (
            f"OOF records {len(oof_records)} != train size {len(y_train)}"
        )
        oof_ids = [r["id"] for r in oof_records]
        assert len(oof_ids) == len(set(oof_ids)), "Duplicate OOF sample ids detected"
        folds_present = sorted({int(r["fold"]) for r in oof_records})
        assert folds_present == list(range(cv_folds)), (
            f"Expected folds {list(range(cv_folds))}, got {folds_present}"
        )
        logger.info(
            f"OOF integrity OK: {len(oof_records)} records, folds={folds_present}, no NaNs"
        )
        if all_fold_histories:
            try:
                combined_history = pd.concat(all_fold_histories, ignore_index=True)
                combined_history_path = model_output_dir / "training_history_all_folds.csv"
                combined_history.to_csv(combined_history_path, index=False)
                logger.info(f"Saved combined training history to {combined_history_path}")
            except Exception as exc:
                logger.warning(f"Failed to save combined training history: {exc}")
        if final_epoch_records:
            try:
                final_summary_df = pd.DataFrame(final_epoch_records)
                summary_path = model_output_dir / "training_history_fold_summary.csv"
                final_summary_df.to_csv(summary_path, index=False)
                logger.info(f"Saved per-fold final metrics summary to {summary_path}")
            except Exception as exc:
                logger.warning(f"Failed to save per-fold summary: {exc}")
        if per_fold_metrics_records:
            try:
                metrics_df = pd.DataFrame(per_fold_metrics_records)
                metrics_path = model_output_dir / "metrics_per_fold.csv"
                metrics_df.to_csv(metrics_path, index=False)
                logger.info(f"Saved fold metrics summary to {metrics_path}")
            except Exception as exc:
                logger.warning(f"Failed to save metrics_per_fold.csv: {exc}")
    
        # Aggregate CV results
        logger.info("=" * 60)
        logger.info("Aggregating CV results...")
        logger.info("=" * 60)
        
        # Compute mean ± std for all metrics
        results = {}
        for metric in fold_metrics_list[0].keys():
            values = [m[metric] for m in fold_metrics_list]
            results[f'{metric}_mean'] = float(np.mean(values))
            results[f'{metric}_std'] = float(np.std(values))
        
        # Add metadata
        results['n_folds'] = cv_folds
        results['model_name'] = model_name
        results['seed'] = seed
        results['input_dim'] = input_dim
        results['n_train'] = len(X_train)
        results['n_test'] = len(X_test) if X_test is not None else 0
        
        # Save results
        save_json(results, model_output_dir / 'results.json')
        logger.info(f"Results saved to {model_output_dir / 'results.json'}")
        
        # Save OOF predictions
        if not oof_records:
            base_ids = np.arange(len(oof_predictions))
            sample_base = sample_ids if sample_ids is not None else base_ids
            oof_df = pd.DataFrame(
                {
                    "id": base_ids,
                    "sample_id": sample_base,
                    "fold": -1,
                    "model": model_name,
                    "y_true": y_train,
                    "y_pred": oof_predictions,
                }
            )
        else:
            oof_df = pd.DataFrame(oof_records)
        oof_df.sort_values(by=["id", "fold"], inplace=True)
        oof_df["oof"] = oof_df["y_pred"]
        oof_df["y"] = oof_df["y_true"]
        oof_path = model_output_dir / 'oof_predictions.csv'
        oof_df.to_csv(oof_path, index=False)
        logger.info(f"OOF predictions saved to {oof_path}")
        try:
            oof_manifest = {
                "rows": int(len(oof_df)),
                "columns": list(oof_df.columns),
                "target_positive_ratio": float(oof_df["y_true"].mean()),
                "folds_present": sorted(oof_df["fold"].unique().tolist()),
            }
            if "sample_id" in oof_df.columns:
                oof_manifest["unique_sample_ids"] = int(oof_df["sample_id"].nunique())
            (model_output_dir / "oof_manifest.json").write_text(json.dumps(oof_manifest, indent=2))
        except Exception as exc:
            logger.warning(f"Failed to write OOF manifest: {exc}")
        
        # Diagnostic: 5-fold ensemble on Test (not deploy; kept separate from final_model)
        test_ensemble_metrics = None
        if test_scoring and X_test is not None and test_preds_per_fold:
            logger.info("Scoring held-out test set via fold ensemble (diagnostic only)...")
            assert len(test_preds_per_fold) == cv_folds, (
                f"Expected {cv_folds} fold test preds, got {len(test_preds_per_fold)}"
            )
            test_ensemble = np.mean(np.vstack(test_preds_per_fold), axis=0)
            ens_df = pd.DataFrame({
                "id": np.arange(len(y_test), dtype=int),
                "y_true": y_test.astype(int),
                "y_pred": test_ensemble.astype(float),
                "model": model_name,
                "source": "fold_ensemble",
                "aggregation": "mean_calibrated_fold_proba",
            })
            ens_path = model_output_dir / "test_ensemble_predictions.csv"
            ens_df.to_csv(ens_path, index=False)

            test_ensemble_metrics = compute_all_metrics(y_test, test_ensemble)
            y_pred_05 = (test_ensemble >= 0.5).astype(int)
            test_ensemble_metrics['confusion_matrix_threshold_0.5'] = _confusion_matrix_dict(y_test, y_pred_05)
            test_ensemble_metrics['n_test'] = int(len(y_test))
            test_ensemble_metrics['n_folds_ensembled'] = int(len(test_preds_per_fold))
            test_ensemble_metrics['role'] = 'diagnostic_nested_cv_style'
            save_json(test_ensemble_metrics, model_output_dir / "test_ensemble_metrics.json")
            logger.info(
                f"Test ensemble (diagnostic): PR-AUC={test_ensemble_metrics['pr_auc']:.4f}, "
                f"ROC-AUC={test_ensemble_metrics['roc_auc']:.4f}"
            )
        elif test_scoring and X_test is None:
            logger.warning("Test scoring requested but test data not available")
        elif test_scoring and not test_preds_per_fold:
            logger.warning("Test scoring requested but no per-fold test predictions were collected")

        # Train-OOF summary metrics (for results.json parallel block)
        train_oof_metrics = compute_all_metrics(y_train, oof_predictions)
        results['train_oof'] = {
            'pr_auc': float(train_oof_metrics['pr_auc']),
            'roc_auc': float(train_oof_metrics['roc_auc']),
            'f1': float(train_oof_metrics['f1']),
            'accuracy': float(train_oof_metrics['accuracy']),
            'brier': float(train_oof_metrics['brier']),
            'ece': float(train_oof_metrics['ece']),
            'n_train': int(len(y_train)),
            'source': 'oof_predictions',
        }

        # Final deploy model: best-fold params (Train val only) → full Train retrain → Test predict
        build_final = cfg.get('modeling', {}).get('final_model', test_scoring)
        test_final_metrics = None
        if build_final and fold_resolved_params:
            logger.info("=" * 60)
            logger.info("Building final deploy model (best-fold val params, no Test leakage)...")
            logger.info("=" * 60)

            selection_metric = cfg.get('modeling', {}).get(
                'final_model_selection_metric', 'pr_auc'
            )
            strategy_cfg = cfg.get('modeling', {}).get(
                'final_model_param_strategy', 'best_fold_val_pr_auc'
            )
            if strategy_cfg == 'median_mode_aggregate_fold_params':
                final_params = _sanitize_model_params(
                    _aggregate_resolved_params(fold_resolved_params)
                )
                selected_fold = None
                selection_meta = {
                    'param_strategy': 'median_mode_aggregate_fold_params',
                    'param_strategy_note': 'Config requested median/mode aggregate. No Test used.',
                    'selection_metric': selection_metric,
                    'selected_fold': None,
                }
            else:
                final_params, selected_fold, selection_meta = _select_best_fold_params(
                    fold_resolved_params, fold_metrics_list, metric_key=selection_metric
                )

            logger.info(
                f"Final params strategy: {selection_meta.get('param_strategy')} "
                f"(selected_fold={selected_fold})"
            )
            logger.info(f"Final resolved params: {final_params}")

            final_dir = model_output_dir / "final_model"
            final_dir.mkdir(parents=True, exist_ok=True)

            from sklearn.model_selection import train_test_split as _tts

            final_model = build_model(model_name, final_params, cfg, input_dim, seed, joint)
            refit_epochs = None
            if model_name in ['tabnet', 'vae_tabnet']:
                # A: ES on Train-internal 90/10 to pick epoch budget (no Test)
                X_final_tr, X_final_va, y_final_tr, y_final_va = _tts(
                    X_train, y_train, test_size=0.1, stratify=y_train, random_state=seed
                )
                if model_name == 'tabnet':
                    final_model.fit(
                        X_final_tr, y_final_tr, X_final_va, y_final_va,
                        final_dir / 'tensorboard_es',
                        early_stopping=True,
                    )
                    refit_epochs = getattr(final_model, 'best_epoch', None) or final_params.get(
                        'epochs', cfg.get('tabnet', {}).get('epochs', 80)
                    )
                    # B: full-Train refit with fixed epochs (no ES)
                    final_model = build_model(model_name, final_params, cfg, input_dim, seed, joint)
                    final_model.fit(
                        X_train, y_train, None, None,
                        final_dir / 'tensorboard',
                        early_stopping=False,
                        max_epochs=int(refit_epochs),
                    )
                    logger.info(
                        f"TabNet final full-Train refit for {refit_epochs} epochs "
                        f"(ES epoch from Train-internal split)"
                    )
                else:
                    # VAE-TabNet: keep Train-internal val for ES (no second-pass API yet)
                    final_model.fit(
                        X_final_tr, y_final_tr, X_final_va, y_final_va,
                        final_dir / 'tensorboard'
                    )
                    logger.warning(
                        "VAE-TabNet final_model uses Train 90/10 fit (no full-Train refit API); "
                        "params still selected without Test."
                    )
            else:
                final_model.fit(X_train, y_train)

            # Calibrate using OOF predictions (no Test leakage)
            calibration_cfg = cfg.get('modeling', {}).get('calibration', {})
            final_calibrator = None
            if calibration_cfg.get('enabled', True):
                final_calibrator = fit_calibrator(
                    oof_predictions, y_train,
                    method=calibration_cfg.get('method', 'platt')
                )
                save_joblib(final_calibrator, str(final_dir / 'calibrator.pkl'))

            final_model.save(final_dir)
            save_json(final_params, final_dir / 'resolved_params.json')

            feature_schema = selected_features if isinstance(selected_features, dict) else {
                "selected_features": selected_features
            }
            save_json(feature_schema, final_dir / 'feature_schema.json')

            # Threshold policy from OOF only (deploy policy — never optimized on Test)
            thr_oof, f1_oof = find_best_threshold(y_train, oof_predictions, 'f1')
            threshold_policy = {
                "method": "maximize_f1_on_oof",
                "threshold": float(thr_oof),
                "f1_oof": float(f1_oof),
                "source": "oof_predictions",
                "anti_leak": "Threshold not selected on Test",
            }
            save_json(threshold_policy, final_dir / 'threshold_policy.json')

            training_metadata = {
                "model_name": model_name,
                "joint_training": bool(joint),
                "n_train": int(len(X_train)),
                "n_features": int(input_dim),
                "seed": int(seed),
                "calibration_source": "oof_predictions" if final_calibrator is not None else None,
                "created_at": datetime.now().isoformat(),
                "fold_params_candidates": fold_resolved_params,
                "full_train_refit_epochs": int(refit_epochs) if refit_epochs is not None else None,
                **selection_meta,
            }
            save_json(training_metadata, final_dir / 'training_metadata.json')
            logger.info(f"Final deploy model saved to {final_dir}")

            # Deploy Test prediction with final_model (single pass after freeze)
            if test_scoring and X_test is not None:
                logger.info("Predicting held-out Test with final_model (deploy path)...")
                p_test_final = np.asarray(final_model.predict_proba(X_test), dtype=float)
                if final_calibrator is not None:
                    p_test_final = final_calibrator.transform(p_test_final)

                final_test_df = pd.DataFrame({
                    "id": np.arange(len(y_test), dtype=int),
                    "y_true": y_test.astype(int),
                    "y_pred": p_test_final.astype(float),
                    "model": model_name,
                    "source": "final_model",
                })
                final_test_path = final_dir / "test_predictions.csv"
                final_test_df.to_csv(final_test_path, index=False)

                # Metrics: ranking metrics + classification at OOF threshold (and @0.5)
                test_final_metrics = compute_all_metrics(y_test, p_test_final)
                y_pred_oof_thr = (p_test_final >= float(thr_oof)).astype(int)
                y_pred_05 = (p_test_final >= 0.5).astype(int)
                from sklearn.metrics import f1_score as _f1, accuracy_score as _acc
                test_final_metrics['threshold_deploy_oof'] = float(thr_oof)
                test_final_metrics['f1_at_oof_threshold'] = float(_f1(y_test, y_pred_oof_thr))
                test_final_metrics['accuracy_at_oof_threshold'] = float(_acc(y_test, y_pred_oof_thr))
                test_final_metrics['confusion_matrix_oof_threshold'] = _confusion_matrix_dict(
                    y_test, y_pred_oof_thr
                )
                test_final_metrics['confusion_matrix_threshold_0.5'] = _confusion_matrix_dict(
                    y_test, y_pred_05
                )
                test_final_metrics['n_test'] = int(len(y_test))
                test_final_metrics['threshold_source'] = 'oof'
                test_final_metrics['anti_leak'] = (
                    'Params/calibrator/threshold from Train only; Test used once for scoring'
                )
                # Do not use Test-optimized F1 threshold as deploy policy
                test_final_metrics.pop('threshold_f1', None)
                save_json(test_final_metrics, final_dir / "test_metrics.json")

                # Root test_predictions = deploy (final_model) for evaluation.py compatibility
                root_test_path = model_output_dir / "test_predictions.csv"
                final_test_df.to_csv(root_test_path, index=False)
                save_json(test_final_metrics, model_output_dir / "test_metrics.json")

                aggregation_meta = {
                    "deploy_source": "final_model",
                    "deploy_predictions": str(final_test_path),
                    "root_test_predictions": str(root_test_path),
                    "ensemble_diagnostic": {
                        "predictions": str(model_output_dir / "test_ensemble_predictions.csv"),
                        "metrics": str(model_output_dir / "test_ensemble_metrics.json"),
                        "method": "mean_calibrated_fold_proba",
                        "n_folds": int(len(test_preds_per_fold)) if test_preds_per_fold else 0,
                    },
                    "anti_leak": {
                        "param_selection": selection_meta.get('param_strategy'),
                        "calibration": "oof_predictions",
                        "threshold_deploy": "oof_predictions",
                        "test_used_for": "scoring_only_after_final_model_frozen",
                    },
                }
                save_json(aggregation_meta, model_output_dir / "test_aggregation_meta.json")

                results['test_final_model'] = {
                    'pr_auc': float(test_final_metrics['pr_auc']),
                    'roc_auc': float(test_final_metrics['roc_auc']),
                    'f1_at_oof_threshold': float(test_final_metrics['f1_at_oof_threshold']),
                    'f1_at_0.5': float(test_final_metrics['f1']),
                    'accuracy_at_oof_threshold': float(test_final_metrics['accuracy_at_oof_threshold']),
                    'threshold_source': 'oof',
                    'n_test': int(len(y_test)),
                }
                logger.info(
                    f"Test final_model: PR-AUC={test_final_metrics['pr_auc']:.4f}, "
                    f"ROC-AUC={test_final_metrics['roc_auc']:.4f}, "
                    f"F1@OOF_thr={test_final_metrics['f1_at_oof_threshold']:.4f}"
                )

        if test_ensemble_metrics is not None:
            results['test_ensemble'] = {
                'pr_auc': float(test_ensemble_metrics['pr_auc']),
                'roc_auc': float(test_ensemble_metrics['roc_auc']),
                'f1': float(test_ensemble_metrics['f1']),
                'n_test': int(test_ensemble_metrics.get('n_test', 0)),
                'role': 'diagnostic_only',
            }

        # Persist updated results (train_oof + test blocks)
        save_json(results, model_output_dir / 'results.json')
        
        # Calculate total duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        timing_info['end_time'] = end_time.isoformat()
        timing_info['duration_seconds'] = duration
        timing_info['duration_minutes'] = duration / 60
        timing_info['duration_hours'] = duration / 3600
        timing_info['status'] = 'completed'
        
        # Print summary
        logger.info("=" * 70)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Model: {model_name}")
        logger.info(f"Joint Training: {joint}")
        logger.info(f"CV Folds: {cv_folds}")
        logger.info(f"PR-AUC: {results['pr_auc_mean']:.4f} ± {results['pr_auc_std']:.4f}")
        logger.info(f"ROC-AUC: {results['roc_auc_mean']:.4f} ± {results['roc_auc_std']:.4f}")
        logger.info(f"F1: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
        logger.info(f"Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
        logger.info("=" * 70)
        logger.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total Duration: {duration:.2f} seconds ({duration/60:.2f} minutes, {duration/3600:.2f} hours)")
        logger.info("=" * 70)
        
        # Save timing info
        save_json(timing_info, model_output_dir / 'timing.json')
        logger.info(f"Timing information saved to {model_output_dir / 'timing.json'}")
        
    except Exception as e:
        # Handle errors
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        timing_info['end_time'] = end_time.isoformat()
        timing_info['duration_seconds'] = duration
        timing_info['duration_minutes'] = duration / 60
        timing_info['duration_hours'] = duration / 3600
        timing_info['status'] = 'failed'
        timing_info['error'] = str(e)
        timing_info['error_type'] = type(e).__name__
        timing_info['error_traceback'] = traceback.format_exc()
        
        # Log error
        logger.error("=" * 70)
        logger.error("EXPERIMENT FAILED")
        logger.error("=" * 70)
        logger.error(f"Model: {model_name}")
        logger.error(f"Joint Training: {joint}")
        logger.error(f"Error Type: {type(e).__name__}")
        logger.error(f"Error Message: {str(e)}")
        logger.error(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.error(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.error(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.error("=" * 70)
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        logger.error("=" * 70)
        
        # Save timing info with error
        try:
            save_json(timing_info, model_output_dir / 'timing.json')
            logger.error(f"Error information saved to {model_output_dir / 'timing.json'}")
        except:
            # If we can't save to model_output_dir, try to save to a fallback location
            logger.error("Could not save error information to model directory")
        
        # Re-raise to maintain original behavior
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Stage 3 - Modeling')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Configuration file')
    parser.add_argument('--model', type=str, choices=['rf', 'tabnet', 'vae_tabnet'], required=True, help='Model to train')
    parser.add_argument('--joint', action='store_true', help='Use joint training for vae_tabnet')
    
    args = parser.parse_args()
    
    # Setup logging (will be overridden in run_experiment with model-specific log file)
    logger = setup_logging(log_level='INFO')
    
    # Load config
    cfg = load_config(args.config)
    
    # Run experiment
    run_experiment(cfg, args.model, joint=args.joint)


if __name__ == '__main__':
    main()
