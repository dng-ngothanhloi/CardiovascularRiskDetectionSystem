"""
Stage 4 (L1) - Evaluation Script

Reads OOF predictions from artifacts/Model/<model_name>/oof_predictions.csv
and held-out Test predictions (prefer final_model/test_predictions.csv),
computes metrics, performs paired bootstrap comparisons, generates plots,
and writes a comprehensive markdown report.

Does not fit calibrators. Primary Test operating-point metrics use the
OOF-derived threshold in final_model/threshold_policy.json when present.
F1@best retuned on Test is reported only as a helper.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
import json
import logging
from dataclasses import dataclass, asdict
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    recall_score,
)
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from .utils import setup_logging

logger = logging.getLogger('HeartDisease_RiskDiscovery')


@dataclass
class ModelEval:
    """Evaluation metrics for a single model."""
    name: str
    pr_auc: Optional[float] = None
    roc_auc: Optional[float] = None
    f1_best: Optional[float] = None
    thr_best: Optional[float] = None
    brier: Optional[float] = None
    ece: Optional[float] = None
    # Deploy operating point (threshold from Train OOF policy — not optimized on Test)
    thr_deploy: Optional[float] = None
    f1_at_deploy: Optional[float] = None
    accuracy_at_deploy: Optional[float] = None
    sensitivity_at_deploy: Optional[float] = None
    specificity_at_deploy: Optional[float] = None
    thr_deploy_source: Optional[str] = None


def expected_calibration_error(
    y_true: np.ndarray,
    p: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: True binary labels
        p: Predicted probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value
    """
    # Clip probabilities to [0, 1]
    p = np.clip(p, 0.0, 1.0)
    
    # Bin edges
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (p > bin_lower) & (p <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Average predicted probability and true label in this bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = p[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


def best_f1_threshold(
    y_true: np.ndarray,
    p: np.ndarray
) -> Tuple[float, float]:
    """
    Find threshold that maximizes F1 score.
    
    Args:
        y_true: True binary labels
        p: Predicted probabilities
        
    Returns:
        (best_threshold, best_f1_score)
    """
    # Grid search from 0.05 to 0.95, step 0.05
    thresholds = np.arange(0.05, 1.0, 0.05)
    f1_scores = []
    
    for thr in thresholds:
        y_pred = (p >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0.0)
        f1_scores.append(f1)
    
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def load_deploy_threshold(model_dir: Path) -> Optional[Tuple[float, str]]:
    """
    Load frozen deploy threshold from final_model/threshold_policy.json (OOF-derived).

    Returns (threshold, source_note) or None if missing.
    """
    candidates = [
        model_dir / 'final_model' / 'threshold_policy.json',
        model_dir / 'threshold_policy.json',
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                policy = json.load(f)
            thr = policy.get('threshold')
            if thr is None:
                continue
            source = policy.get('source', 'threshold_policy')
            note = f"{path.name}:{source}"
            return float(thr), note
        except Exception as e:
            logger.warning(f"Failed to read threshold policy {path}: {e}")
    return None


def metrics_at_threshold(
    y_true: np.ndarray,
    p: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Binary metrics at a fixed (pre-specified) threshold."""
    y_hat = (np.asarray(p, dtype=float) >= float(threshold)).astype(int)
    y_true = np.asarray(y_true).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        'f1': float(f1_score(y_true, y_hat, zero_division=0.0)),
        'accuracy': float(accuracy_score(y_true, y_hat)),
        'sensitivity': sens,
        'specificity': spec,
        'recall': float(recall_score(y_true, y_hat, zero_division=0.0)),
    }


def attach_deploy_threshold_metrics(
    eval_result: ModelEval,
    df: pd.DataFrame,
    model_dir: Path,
) -> ModelEval:
    """Attach OOF-policy threshold metrics (for Test / deploy reporting)."""
    loaded = load_deploy_threshold(model_dir)
    if loaded is None:
        logger.warning(
            f"No threshold_policy.json under {model_dir}; "
            "Test deploy operating-point metrics unavailable"
        )
        return eval_result
    thr, source_note = loaded
    m = metrics_at_threshold(df['y_true'].values, df['y_pred'].values, thr)
    eval_result.thr_deploy = thr
    eval_result.f1_at_deploy = m['f1']
    eval_result.accuracy_at_deploy = m['accuracy']
    eval_result.sensitivity_at_deploy = m['sensitivity']
    eval_result.specificity_at_deploy = m['specificity']
    eval_result.thr_deploy_source = source_note
    return eval_result


def load_oof(model_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load OOF predictions from model directory.
    
    Handles flexible column mapping:
    - Preferred: id, fold, y_true, y_pred, model
    - Fallbacks: y → y_true, oof → y_pred
    - Missing: id → row index, fold → -1, model → folder name
    
    Args:
        model_dir: Directory containing oof_predictions.csv
        
    Returns:
        DataFrame with columns: id, fold, y_true, y_pred, model
        Returns None if file doesn't exist or is invalid
    """
    oof_file = model_dir / 'oof_predictions.csv'
    
    if not oof_file.exists():
        logger.warning(f"OOF file not found: {oof_file}")
        return None
    
    try:
        df = pd.read_csv(oof_file)
        logger.info(f"Loaded OOF from {oof_file}, shape: {df.shape}")
        
        # Column mapping: {old_name: new_name}
        col_map = {}
        
        # y_true mapping
        if 'y_true' in df.columns:
            col_map['y_true'] = 'y_true'  # Already correct name
        elif 'y' in df.columns:
            col_map['y'] = 'y_true'  # Map old 'y' to new 'y_true'
            logger.info(f"  Mapped 'y' → 'y_true'")
        else:
            logger.error(f"  No y_true column found in {oof_file}")
            return None
        
        # y_pred mapping
        if 'y_pred' in df.columns:
            col_map['y_pred'] = 'y_pred'  # Already correct name
        elif 'oof' in df.columns:
            col_map['oof'] = 'y_pred'  # Map old 'oof' to new 'y_pred'
            logger.info(f"  Mapped 'oof' → 'y_pred'")
        else:
            logger.error(f"  No y_pred column found in {oof_file}")
            return None
        
        # id mapping
        if 'id' in df.columns:
            col_map['id'] = 'id'  # Already correct name
        else:
            df['id'] = df.index
            col_map['id'] = 'id'  # Map 'id' to 'id' (no change)
            logger.info(f"  Created 'id' from row index")
        
        # fold mapping
        if 'fold' in df.columns:
            col_map['fold'] = 'fold'  # Already correct name
        else:
            df['fold'] = -1
            col_map['fold'] = 'fold'  # Map 'fold' to 'fold' (no change)
            logger.info(f"  Created 'fold' = -1 (missing)")
        
        # model name
        model_name = model_dir.name
        df['model'] = model_name
        col_map['model'] = 'model'  # Map 'model' to 'model' (no change)
        
        # Select columns using old names (keys) and rename to new names (values)
        result_df = df[list(col_map.keys())].copy()
        result_df.rename(columns=col_map, inplace=True)
        
        # Validate data
        if result_df['y_true'].isna().any():
            logger.warning(f"  Found NaN in y_true, filling with 0")
            result_df['y_true'] = result_df['y_true'].fillna(0)
        
        if result_df['y_pred'].isna().any():
            logger.warning(f"  Found NaN in y_pred, filling with 0.5")
            result_df['y_pred'] = result_df['y_pred'].fillna(0.5)
        
        # Clip probabilities
        if (result_df['y_pred'] < 0).any() or (result_df['y_pred'] > 1).any():
            logger.warning(f"  Clipping probabilities to [0, 1]")
            result_df['y_pred'] = result_df['y_pred'].clip(0.0, 1.0)
        
        # Ensure integer types
        result_df['y_true'] = result_df['y_true'].astype(int)
        result_df['id'] = result_df['id'].astype(int)
        result_df['fold'] = result_df['fold'].astype(int)
        
        logger.info(f"  Final columns: {list(result_df.columns)}")
        return result_df
        
    except Exception as e:
        logger.error(f"Error loading OOF from {oof_file}: {e}")
        return None


def load_test_predictions(model_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load held-out test predictions from model directory.

    Preference (deploy-first, no leak confusion):
      1. final_model/test_predictions.csv  (final_model deploy path)
      2. test_predictions.csv              (root copy of deploy)
      3. test_ensemble_predictions.csv     (diagnostic fold ensemble)
    """
    candidates = [
        (model_dir / 'final_model' / 'test_predictions.csv', 'final_model'),
        (model_dir / 'test_predictions.csv', 'root'),
        (model_dir / 'test_ensemble_predictions.csv', 'ensemble_diagnostic'),
    ]
    test_file = None
    source_tag = None
    for path, tag in candidates:
        if path.exists():
            test_file = path
            source_tag = tag
            break

    if test_file is None:
        logger.warning(f"Test predictions not found under {model_dir}")
        return None

    try:
        df = pd.read_csv(test_file)
        logger.info(
            f"Loaded test predictions from {test_file} (source={source_tag}), shape: {df.shape}"
        )

        if 'y_true' not in df.columns or 'y_pred' not in df.columns:
            logger.error(f"  test file missing y_true/y_pred in {test_file}")
            return None

        if 'id' not in df.columns:
            df['id'] = np.arange(len(df), dtype=int)
        df['model'] = model_dir.name
        df['y_true'] = df['y_true'].astype(int)
        df['id'] = df['id'].astype(int)
        df['y_pred'] = df['y_pred'].astype(float).clip(0.0, 1.0)
        out = df[['id', 'y_true', 'y_pred', 'model']].copy()
        out.attrs['test_source'] = source_tag
        return out
    except Exception as e:
        logger.error(f"Error loading test predictions from {test_file}: {e}")
        return None


def load_test_ensemble_predictions(model_dir: Path) -> Optional[pd.DataFrame]:
    """Load diagnostic fold-ensemble test predictions if present."""
    test_file = model_dir / 'test_ensemble_predictions.csv'
    if not test_file.exists():
        return None
    try:
        df = pd.read_csv(test_file)
        if 'y_true' not in df.columns or 'y_pred' not in df.columns:
            return None
        if 'id' not in df.columns:
            df['id'] = np.arange(len(df), dtype=int)
        df['model'] = model_dir.name
        df['y_true'] = df['y_true'].astype(int)
        df['id'] = df['id'].astype(int)
        df['y_pred'] = df['y_pred'].astype(float).clip(0.0, 1.0)
        return df[['id', 'y_true', 'y_pred', 'model']].copy()
    except Exception as e:
        logger.warning(f"Failed to load ensemble test preds from {test_file}: {e}")
        return None


def bootstrap_pr_auc(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: Optional[np.ndarray] = None,
    n_boot: int = 2000,
    rng_seed: int = 42
) -> Dict[str, float]:
    """
    Compute bootstrap PR-AUC with optional paired comparison.
    
    Args:
        y_true: True binary labels
        p_a: Predictions from model A
        p_b: Optional predictions from model B (for paired comparison)
        n_boot: Number of bootstrap samples
        rng_seed: Random seed
        
    Returns:
        Dictionary with:
        - If p_b is None: mean, low, high (2.5/97.5 percentiles) of AP_A
        - If p_b is not None: above + diff_mean, diff_low, diff_high for Δ(AP_A - AP_B)
    """
    rng = np.random.RandomState(rng_seed)
    n = len(y_true)
    
    ap_a_samples = []
    ap_b_samples = []
    diff_samples = []
    
    for _ in range(n_boot):
        # Paired bootstrap: sample indices with replacement
        indices = rng.choice(n, size=n, replace=True)
        y_boot = y_true[indices]
        p_a_boot = p_a[indices]
        
        # Compute AP for model A
        try:
            ap_a = average_precision_score(y_boot, p_a_boot)
            ap_a_samples.append(ap_a)
        except ValueError:
            continue
        
        # If model B provided, compute paired difference
        if p_b is not None:
            p_b_boot = p_b[indices]
            try:
                ap_b = average_precision_score(y_boot, p_b_boot)
                ap_b_samples.append(ap_b)
                diff_samples.append(ap_a - ap_b)
            except ValueError:
                continue
    
    result = {
        'mean': float(np.mean(ap_a_samples)),
        'low': float(np.percentile(ap_a_samples, 2.5)),
        'high': float(np.percentile(ap_a_samples, 97.5))
    }
    
    if p_b is not None and len(diff_samples) > 0:
        result['diff_mean'] = float(np.mean(diff_samples))
        result['diff_low'] = float(np.percentile(diff_samples, 2.5))
        result['diff_high'] = float(np.percentile(diff_samples, 97.5))
        result['ap_b_mean'] = float(np.mean(ap_b_samples))
        result['ap_b_low'] = float(np.percentile(ap_b_samples, 2.5))
        result['ap_b_high'] = float(np.percentile(ap_b_samples, 97.5))
    
    return result


def evaluate_model_oof(df: pd.DataFrame) -> ModelEval:
    """
    Compute all metrics for a model from OOF predictions.
    
    Args:
        df: DataFrame with columns: id, fold, y_true, y_pred, model
        
    Returns:
        ModelEval dataclass with computed metrics
    """
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    
    model_name = df['model'].iloc[0]
    logger.info(f"Evaluating model: {model_name}")
    
    # PR-AUC
    try:
        pr_auc = average_precision_score(y_true, y_pred)
    except ValueError as e:
        logger.warning(f"  PR-AUC computation failed: {e}")
        pr_auc = None
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except ValueError as e:
        logger.warning(f"  ROC-AUC computation failed: {e}")
        roc_auc = None
    
    # Best F1 and threshold
    try:
        thr_best, f1_best = best_f1_threshold(y_true, y_pred)
    except Exception as e:
        logger.warning(f"  Best F1 computation failed: {e}")
        thr_best, f1_best = None, None
    
    # Brier score
    try:
        brier = brier_score_loss(y_true, y_pred)
    except Exception as e:
        logger.warning(f"  Brier score computation failed: {e}")
        brier = None
    
    # ECE
    try:
        ece = expected_calibration_error(y_true, y_pred, n_bins=15)
    except Exception as e:
        logger.warning(f"  ECE computation failed: {e}")
        ece = None
    
    return ModelEval(
        name=model_name,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        f1_best=f1_best,
        thr_best=thr_best,
        brier=brier,
        ece=ece
    )


def align_predictions(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two prediction dataframes for paired comparison.
    
    Args:
        df_a: First model predictions
        df_b: Second model predictions
        
    Returns:
        (aligned_df_a, aligned_df_b)
    """
    # Try to align by id and y_true
    if 'id' in df_a.columns and 'id' in df_b.columns:
        # Inner join on id and y_true
        merged = pd.merge(
            df_a[['id', 'y_true', 'y_pred']],
            df_b[['id', 'y_true', 'y_pred']],
            on=['id', 'y_true'],
            suffixes=('_a', '_b'),
            how='inner'
        )
        
        if len(merged) > 0:
            logger.info(f"  Aligned {len(merged)} samples by id and y_true")
            df_a_aligned = df_a[df_a['id'].isin(merged['id'])].copy()
            df_b_aligned = df_b[df_b['id'].isin(merged['id'])].copy()
            # Reorder to match
            df_a_aligned = df_a_aligned.set_index('id').loc[merged['id']].reset_index()
            df_b_aligned = df_b_aligned.set_index('id').loc[merged['id']].reset_index()
            return df_a_aligned, df_b_aligned
        else:
            logger.warning(f"  No matching samples by id, falling back to index alignment")
    
    # Fallback: align by index
    min_len = min(len(df_a), len(df_b))
    logger.warning(f"  Aligning by index, using first {min_len} samples")
    return df_a.iloc[:min_len].copy(), df_b.iloc[:min_len].copy()


def plot_pr_overlay(
    oof_map: Dict[str, pd.DataFrame],
    out_path: Path
) -> bool:
    """Plot PR curves for all models."""
    if not oof_map:
        return False
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for model_name, df in oof_map.items():
        y_true = df['y_true'].values
        y_pred = df['y_pred'].values
        
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            ap = average_precision_score(y_true, y_pred)
            ax.plot(recall, precision, label=f"{model_name} (AP={ap:.4f})", linewidth=2)
        except Exception as e:
            logger.warning(f"  Failed to plot PR curve for {model_name}: {e}")
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves (OOF)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved PR overlay to {out_path}")
    return True


def plot_roc_overlay(
    oof_map: Dict[str, pd.DataFrame],
    out_path: Path
) -> bool:
    """Plot ROC curves for all models."""
    if not oof_map:
        return False
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for model_name, df in oof_map.items():
        y_true = df['y_true'].values
        y_pred = df['y_pred'].values
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.4f})", linewidth=2)
        except Exception as e:
            logger.warning(f"  Failed to plot ROC curve for {model_name}: {e}")
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves (OOF)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved ROC overlay to {out_path}")
    return True


def plot_reliability_overlay(
    oof_map: Dict[str, pd.DataFrame],
    out_path: Path
) -> bool:
    """Plot reliability (calibration) curves for all models."""
    if not oof_map:
        return False
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for model_name, df in oof_map.items():
        y_true = df['y_true'].values
        y_pred = df['y_pred'].values
        
        try:
            CalibrationDisplay.from_predictions(
                y_true, y_pred,
                n_bins=15,
                name=model_name,
                ax=ax
            )
        except Exception as e:
            logger.warning(f"  Failed to plot reliability curve for {model_name}: {e}")
    
    ax.set_title('Reliability Curves (OOF)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved reliability overlay to {out_path}")
    return True


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    threshold: float,
    out_path: Path
) -> bool:
    """Plot confusion matrix at best threshold."""
    try:
        y_pred_binary = (y_pred >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Negative', 'Positive'],
               yticklabels=['Negative', 'Positive'],
               title=f'Confusion Matrix: {model_name} (thr={threshold:.3f})',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix to {out_path}")
        return True
    except Exception as e:
        logger.warning(f"  Failed to plot confusion matrix: {e}")
        return False


def write_markdown_report(
    out_dir: Path,
    model_evals: Dict[str, ModelEval],
    oof_map: Dict[str, pd.DataFrame],
    pairwise_ci: Dict[str, Dict[str, float]],
    make_plots: bool = True,
    test_evals: Optional[Dict[str, ModelEval]] = None,
    test_map: Optional[Dict[str, pd.DataFrame]] = None,
    test_pairwise_ci: Optional[Dict[str, Dict[str, float]]] = None,
) -> Path:
    """
    Write comprehensive markdown evaluation report.
    
    Args:
        out_dir: Output directory
        model_evals: Dictionary mapping model names to ModelEval (OOF)
        oof_map: Dictionary mapping model names to OOF DataFrames
        pairwise_ci: Dictionary mapping pair names to CI results (OOF)
        make_plots: Whether plots were generated
        test_evals: Optional model evaluations on held-out Test
        test_map: Optional test prediction DataFrames
        test_pairwise_ci: Optional pairwise CI on Test
        
    Returns:
        Path to the generated report.md
    """
    report_path = out_dir / 'report.md'
    test_evals = test_evals or {}
    test_map = test_map or {}
    test_pairwise_ci = test_pairwise_ci or {}
    
    with open(report_path, 'w') as f:
        f.write("# Stage 4 – Evaluation Report (Cardio Target)\n\n")
        f.write(
            "This report summarizes out-of-fold (OOF) CV results and, when available, "
            "held-out Test results from **`final_model`**. OOF and Test metrics are "
            "reported separately and must not be mixed as one protocol. "
            "Evaluation does **not** fit calibrators (ECE only). "
            "Primary Test operating point uses the **OOF** threshold in "
            "`final_model/threshold_policy.json`; F1@best retuned on Test is a helper only.\n\n"
        )
        
        # Aggregate Metrics Table (OOF)
        f.write("## 1. OOF / CV Results\n\n")
        f.write("### Aggregate Metrics (OOF)\n\n")
        f.write("| Model | PR-AUC | ROC-AUC | F1@best | Thr(best) | Brier | ECE |\n")
        f.write("|------|--------:|--------:|--------:|----------:|------:|----:|\n")
        
        for model_name in ['rf', 'tabnet', 'vae_tabnet']:
            eval_result = model_evals.get(model_name)
            if eval_result:
                pr_auc = f"{eval_result.pr_auc:.4f}" if eval_result.pr_auc is not None else "-"
                roc_auc = f"{eval_result.roc_auc:.4f}" if eval_result.roc_auc is not None else "-"
                f1 = f"{eval_result.f1_best:.4f}" if eval_result.f1_best is not None else "-"
                thr = f"{eval_result.thr_best:.3f}" if eval_result.thr_best is not None else "-"
                brier = f"{eval_result.brier:.4f}" if eval_result.brier is not None else "-"
                ece = f"{eval_result.ece:.4f}" if eval_result.ece is not None else "-"
                f.write(f"| {model_name} | {pr_auc} | {roc_auc} | {f1} | {thr} | {brier} | {ece} |\n")
            else:
                f.write(f"| {model_name} | - | - | - | - | - | - |\n")
        
        f.write("\n")
        
        # Pairwise Comparison Table (OOF)
        f.write("### Pairwise Comparison on OOF (PR-AUC, 95% CI via paired bootstrap)\n\n")
        f.write("| Pair | AP_A | AP_B | Δ(AP_A−AP_B) | 95% CI(Δ) |\n")
        f.write("|------|-----:|-----:|-------------:|:----------|\n")
        
        pairs = [
            ('RF vs TabNet', 'rf', 'tabnet'),
            ('TabNet vs VAE→TabNet', 'tabnet', 'vae_tabnet'),
            ('RF vs VAE→TabNet', 'rf', 'vae_tabnet')
        ]
        
        for pair_name, model_a, model_b in pairs:
            ci_result = pairwise_ci.get(pair_name)
            if ci_result:
                ap_a = f"{ci_result.get('mean', 0):.4f}"
                ap_b = f"{ci_result.get('ap_b_mean', 0):.4f}"
                diff_mean = f"{ci_result.get('diff_mean', 0):.4f}"
                diff_low = f"{ci_result.get('diff_low', 0):.4f}"
                diff_high = f"{ci_result.get('diff_high', 0):.4f}"
                ci_str = f"[{diff_low}, {diff_high}]"
                f.write(f"| {pair_name} | {ap_a} | {ap_b} | {diff_mean} | {ci_str} |\n")
            else:
                f.write(f"| {pair_name} | - | - | - | - |\n")
        
        f.write("\n")
        
        # Diagnostic Plots (OOF)
        f.write("### Diagnostic Plots (OOF)\n\n")
        plots_dir = out_dir / 'plots'
        
        plot_files = {
            'PR Curves': plots_dir / 'pr_overlay.png',
            'ROC Curves': plots_dir / 'roc_overlay.png',
            'Reliability Curves': plots_dir / 'reliability_overlay.png'
        }
        
        for plot_name, plot_path in plot_files.items():
            if plot_path.exists() and make_plots:
                f.write(f"* **{plot_name}**: `{plot_path.relative_to(out_dir)}`\n")
            else:
                f.write(f"* **{plot_name}**: -\n")
        
        # Confusion matrices (OOF)
        f.write("\n### Confusion Matrices on OOF (at best threshold)\n\n")
        for model_name in ['rf', 'tabnet', 'vae_tabnet']:
            eval_result = model_evals.get(model_name)
            if eval_result and eval_result.thr_best is not None:
                cm_path = plots_dir / f'confusion_matrix_{model_name}.png'
                if cm_path.exists():
                    f.write(f"* **{model_name}**: `{cm_path.relative_to(out_dir)}`\n")
                else:
                    f.write(f"* **{model_name}**: -\n")
            else:
                f.write(f"* **{model_name}**: -\n")

        # Test section — deploy = final_model; ensemble is diagnostic only
        f.write("\n## 2. Held-out Test Results (`final_model` deploy)\n\n")
        if not test_evals:
            f.write(
                "_No deploy test predictions found "
                "(`final_model/test_predictions.csv` or root `test_predictions.csv`). "
                "Run modeling with `test_scoring: true` and `final_model: true`._\n\n"
            )
        else:
            f.write(
                "Primary Test metrics use **`final_model`** predictions "
                "(params/calibrator/threshold selected on Train/OOF only; Test scored once after freeze). "
                "See `test_aggregation_meta.json`.\n\n"
            )
            f.write("### Aggregate Metrics (Test — deploy @ OOF threshold) **primary**\n\n")
            f.write(
                "| Model | PR-AUC | ROC-AUC | Thr(OOF) | F1@OOF | Acc@OOF | Sens@OOF | Spec@OOF | Brier | ECE |\n"
            )
            f.write(
                "|------|--------:|--------:|---------:|-------:|--------:|---------:|---------:|------:|----:|\n"
            )
            for model_name in ['rf', 'tabnet', 'vae_tabnet']:
                eval_result = test_evals.get(model_name)
                if eval_result:
                    pr_auc = f"{eval_result.pr_auc:.4f}" if eval_result.pr_auc is not None else "-"
                    roc_auc = f"{eval_result.roc_auc:.4f}" if eval_result.roc_auc is not None else "-"
                    thr_d = f"{eval_result.thr_deploy:.3f}" if eval_result.thr_deploy is not None else "-"
                    f1_d = f"{eval_result.f1_at_deploy:.4f}" if eval_result.f1_at_deploy is not None else "-"
                    acc_d = (
                        f"{eval_result.accuracy_at_deploy:.4f}"
                        if eval_result.accuracy_at_deploy is not None else "-"
                    )
                    sens_d = (
                        f"{eval_result.sensitivity_at_deploy:.4f}"
                        if eval_result.sensitivity_at_deploy is not None else "-"
                    )
                    spec_d = (
                        f"{eval_result.specificity_at_deploy:.4f}"
                        if eval_result.specificity_at_deploy is not None else "-"
                    )
                    brier = f"{eval_result.brier:.4f}" if eval_result.brier is not None else "-"
                    ece = f"{eval_result.ece:.4f}" if eval_result.ece is not None else "-"
                    f.write(
                        f"| {model_name} | {pr_auc} | {roc_auc} | {thr_d} | {f1_d} | "
                        f"{acc_d} | {sens_d} | {spec_d} | {brier} | {ece} |\n"
                    )
                else:
                    f.write(f"| {model_name} | - | - | - | - | - | - | - | - | - |\n")
            f.write(
                "\nThreshold from `final_model/threshold_policy.json` "
                "(selected on Train OOF; **not** re-tuned on Test).\n\n"
            )

            f.write("### Helper Metrics (Test — F1@best retuned on Test) **secondary**\n\n")
            f.write("| Model | F1@best* | Thr(best)* |\n")
            f.write("|------|--------:|----------:|\n")
            for model_name in ['rf', 'tabnet', 'vae_tabnet']:
                eval_result = test_evals.get(model_name)
                if eval_result:
                    f1 = f"{eval_result.f1_best:.4f}" if eval_result.f1_best is not None else "-"
                    thr = f"{eval_result.thr_best:.3f}" if eval_result.thr_best is not None else "-"
                    f.write(f"| {model_name} | {f1} | {thr} |\n")
                else:
                    f.write(f"| {model_name} | - | - |\n")
            f.write(
                "\n\\*Do **not** use as clinical / deploy operating point — "
                "optimistic relative to the frozen OOF threshold.\n\n"
            )

            f.write("### Pairwise Comparison on Test — final_model (PR-AUC, 95% CI)\n\n")
            f.write("| Pair | AP_A | AP_B | Δ(AP_A−AP_B) | 95% CI(Δ) |\n")
            f.write("|------|-----:|-----:|-------------:|:----------|\n")
            for pair_name, model_a, model_b in pairs:
                ci_result = test_pairwise_ci.get(pair_name)
                if ci_result:
                    ap_a = f"{ci_result.get('mean', 0):.4f}"
                    ap_b = f"{ci_result.get('ap_b_mean', 0):.4f}"
                    diff_mean = f"{ci_result.get('diff_mean', 0):.4f}"
                    diff_low = f"{ci_result.get('diff_low', 0):.4f}"
                    diff_high = f"{ci_result.get('diff_high', 0):.4f}"
                    ci_str = f"[{diff_low}, {diff_high}]"
                    f.write(f"| {pair_name} | {ap_a} | {ap_b} | {diff_mean} | {ci_str} |\n")
                else:
                    f.write(f"| {pair_name} | - | - | - | - |\n")

            f.write("\n### Confusion Matrices on Test (at **OOF deploy** threshold)\n\n")
            for model_name in ['rf', 'tabnet', 'vae_tabnet']:
                cm_path = plots_dir / f'confusion_matrix_test_{model_name}.png'
                if cm_path.exists():
                    f.write(f"* **{model_name}**: `{cm_path.relative_to(out_dir)}`\n")
                else:
                    f.write(f"* **{model_name}**: -\n")

            f.write(
                "\n### Test ensemble (diagnostic only)\n\n"
                "If present, `test_ensemble_predictions.csv` is the 5-fold mean "
                "(nested-CV style). **Not used for deploy.** "
                "Compare with `final_model/test_metrics.json`.\n"
            )
    

    logger.info(f"Written markdown report to {report_path}")
    return report_path


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(
        description='Stage 4 (L1) - Evaluation Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model_root',
        type=str,
        default='artifacts/Model',
        help='Root directory containing model subfolders'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='artifacts/Eval',
        help='Output directory for report and plots'
    )
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Skip plot generation'
    )
    parser.add_argument(
        '--n_boot',
        type=int,
        default=2000,
        help='Number of bootstrap samples'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level='INFO')
    logger.info("=" * 70)
    logger.info("STAGE 4 - EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Model root: {args.model_root}")
    logger.info(f"Output directory: {args.out_dir}")
    logger.info(f"Bootstrap samples: {args.n_boot}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Generate plots: {not args.no_plots}")
    
    # Setup paths
    model_root = Path(args.model_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load OOF predictions
    model_names = ['rf', 'tabnet', 'vae_tabnet']
    oof_map = {}
    model_evals = {}
    
    for model_name in model_names:
        model_dir = model_root / model_name
        df = load_oof(model_dir)
        if df is not None:
            oof_map[model_name] = df
            model_evals[model_name] = evaluate_model_oof(df)
        else:
            logger.warning(f"Skipping {model_name} (no OOF data)")
    
    if not oof_map:
        logger.error("No OOF predictions found! Exiting.")
        return
    
    # Compute pairwise PR-AUC comparisons (OOF)
    pairwise_ci = {}
    pairs = [
        ('RF vs TabNet', 'rf', 'tabnet'),
        ('TabNet vs VAE→TabNet', 'tabnet', 'vae_tabnet'),
        ('RF vs VAE→TabNet', 'rf', 'vae_tabnet')
    ]
    
    for pair_name, model_a, model_b in pairs:
        if model_a in oof_map and model_b in oof_map:
            logger.info(f"Computing paired bootstrap (OOF) for {pair_name}...")
            df_a, df_b = align_predictions(oof_map[model_a], oof_map[model_b])
            
            y_true = df_a['y_true'].values
            p_a = df_a['y_pred'].values
            p_b = df_b['y_pred'].values
            
            ci_result = bootstrap_pr_auc(
                y_true, p_a, p_b,
                n_boot=args.n_boot,
                rng_seed=args.seed
            )
            pairwise_ci[pair_name] = ci_result
            logger.info(f"  Δ(AP_A - AP_B) = {ci_result.get('diff_mean', 0):.4f} "
                       f"[{ci_result.get('diff_low', 0):.4f}, {ci_result.get('diff_high', 0):.4f}]")
        else:
            logger.warning(f"Skipping {pair_name} (missing models)")

    # Load held-out Test predictions (separate from OOF)
    test_map = {}
    test_evals = {}
    for model_name in model_names:
        model_dir = model_root / model_name
        df_test = load_test_predictions(model_dir)
        if df_test is not None:
            test_map[model_name] = df_test
            te = evaluate_model_oof(df_test)
            te = attach_deploy_threshold_metrics(te, df_test, model_dir)
            test_evals[model_name] = te
            logger.info(
                f"Test eval {model_name}: PR-AUC={te.pr_auc}, ROC-AUC={te.roc_auc}, "
                f"F1@OOF_thr={te.f1_at_deploy} (thr={te.thr_deploy})"
            )

    test_pairwise_ci = {}
    for pair_name, model_a, model_b in pairs:
        if model_a in test_map and model_b in test_map:
            logger.info(f"Computing paired bootstrap (Test) for {pair_name}...")
            df_a, df_b = align_predictions(test_map[model_a], test_map[model_b])
            # Guard: require matching y_true after align
            if not np.array_equal(df_a['y_true'].values, df_b['y_true'].values):
                logger.warning(f"  Skipping {pair_name}: y_true mismatch after align")
                continue
            ci_result = bootstrap_pr_auc(
                df_a['y_true'].values,
                df_a['y_pred'].values,
                df_b['y_pred'].values,
                n_boot=args.n_boot,
                rng_seed=args.seed
            )
            test_pairwise_ci[pair_name] = ci_result
            logger.info(
                f"  Test Δ(AP_A - AP_B) = {ci_result.get('diff_mean', 0):.4f} "
                f"[{ci_result.get('diff_low', 0):.4f}, {ci_result.get('diff_high', 0):.4f}]"
            )
    
    # Generate plots
    if not args.no_plots:
        logger.info("Generating plots...")
        plot_pr_overlay(oof_map, plots_dir / 'pr_overlay.png')
        plot_roc_overlay(oof_map, plots_dir / 'roc_overlay.png')
        plot_reliability_overlay(oof_map, plots_dir / 'reliability_overlay.png')
        
        # Confusion matrices (OOF)
        for model_name, eval_result in model_evals.items():
            if eval_result.thr_best is not None:
                df = oof_map[model_name]
                plot_confusion_matrix(
                    df['y_true'].values,
                    df['y_pred'].values,
                    model_name,
                    eval_result.thr_best,
                    plots_dir / f'confusion_matrix_{model_name}.png'
                )

        # Confusion matrices (Test) — primary = OOF deploy threshold
        for model_name, eval_result in test_evals.items():
            thr_cm = (
                eval_result.thr_deploy
                if eval_result.thr_deploy is not None
                else eval_result.thr_best
            )
            if thr_cm is not None:
                df = test_map[model_name]
                plot_confusion_matrix(
                    df['y_true'].values,
                    df['y_pred'].values,
                    f"{model_name}_test",
                    thr_cm,
                    plots_dir / f'confusion_matrix_test_{model_name}.png'
                )
    
    # Write summary JSON (OOF and Test kept separate)
    summary = {
        'oof_model_metrics': {
            name: {k: v for k, v in asdict(eval_result).items() if v is not None}
            for name, eval_result in model_evals.items()
        },
        'oof_pairwise_ci': pairwise_ci,
        'test_model_metrics': {
            name: {k: v for k, v in asdict(eval_result).items() if v is not None}
            for name, eval_result in test_evals.items()
        },
        'test_pairwise_ci': test_pairwise_ci,
    }
    
    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Written summary JSON to {summary_path}")

    # Write markdown report
    report_path = write_markdown_report(
        out_dir,
        model_evals,
        oof_map,
        pairwise_ci,
        make_plots=not args.no_plots,
        test_evals=test_evals,
        test_map=test_map,
        test_pairwise_ci=test_pairwise_ci,
    )
    
    logger.info("=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)
    print(f"Markdown report → {report_path}")


if __name__ == '__main__':
    main()
