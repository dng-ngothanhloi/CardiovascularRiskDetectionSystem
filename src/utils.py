"""
Utility functions for reproducibility, logging, and configuration management.
"""
import os
import json
import random
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Literal
import numpy as np
import tensorflow as tf
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    brier_score_loss, roc_curve, precision_recall_curve, confusion_matrix
)

# Set random seeds for reproducibility
RANDOM_STATE = 42

# Allow importing submodules from the utils/ directory (e.g., src.utils.metrics)
__path__ = [str((Path(__file__).resolve().parent / "utils").resolve())]


def set_seeds(seed: int = RANDOM_STATE) -> None:
    """
    Fix random seeds for NumPy, TensorFlow, and Python's random module.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Set deterministic operations for TensorFlow
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_all_seeds(seed: int, deterministic_tf: bool = False) -> None:
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic_tf: If True, enable TensorFlow deterministic operations
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic_tf:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Configure logging for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file (if None, uses src/logging/training.log)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('HeartDisease_RiskDiscovery')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (default to src/logging/training.log if not specified)
    if log_file is None:
        # Create src/logging directory if it doesn't exist
        log_dir = Path('src/logging')
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(log_dir / 'training.log')
    
    # Ensure log file directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, mode='a')  # Append mode
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    current_file = Path(__file__)
    # Assuming utils.py is in src/, go up one level to get project root
    return current_file.parent.parent


def get_logger() -> logging.Logger:
    """
    Get or create the default logger for the project.
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger('HeartDisease_RiskDiscovery')
    if not logger.handlers:
        return setup_logging()
    return logger


def ensure_dirs(*dirs: str) -> None:
    """
    Ensure multiple directories exist, create if they don't.
    
    Args:
        *dirs: Variable number of directory paths to create
    """
    for dir_path in dirs:
        ensure_dir(dir_path)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ==================== I/O Utilities ====================

def load_numpy(path: str) -> np.ndarray:
    """Load numpy array from .npy file."""
    return np.load(path)


def save_numpy(arr: np.ndarray, path: str) -> None:
    """Save numpy array to .npy file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_joblib(obj: Any, path: str) -> None:
    """Save object using joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: str) -> Any:
    """Load object using joblib."""
    return joblib.load(path)


def save_keras_model(model: tf.keras.Model, path: str) -> None:
    """Save Keras model."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_keras_model(path: str) -> tf.keras.Model:
    """Load Keras model."""
    return tf.keras.models.load_model(path)


def out_dir_fold(cfg: Dict[str, Any], model_name: str, fold: int) -> Path:
    """Get output directory for a specific model and fold."""
    model_dir = Path(cfg['paths']['model_dir'])
    fold_dir = model_dir / model_name / f'fold_{fold}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    return fold_dir


def write_run_info(cfg: Dict[str, Any], model_name: str, out_dir: Path, fe_inputs_info: Optional[Dict[str, Any]] = None) -> None:
    """Write run information (versions, device, FE inputs) to JSON."""
    import sys
    import platform
    
    run_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'numpy_version': np.__version__,
        'sklearn_version': None,
        'tensorflow_version': tf.__version__,
        'device': 'GPU' if tf.config.list_physical_devices('GPU') else 'CPU',
        'model_name': model_name,
        'seed': cfg.get('random_seed', 42),
    }
    
    try:
        import sklearn
        run_info['sklearn_version'] = sklearn.__version__
    except:
        pass
    
    # Add FE inputs information if provided
    if fe_inputs_info is not None:
        run_info['fe_inputs'] = fe_inputs_info
    
    save_json(run_info, out_dir / 'run_info.json')


# ==================== Metrics Utilities ====================

def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray, metric: str = 'f1') -> Tuple[float, float]:
    """
    Find best threshold for binary classification.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('f1' or 'f1_recall_08' or 'f1_recall_09')
    
    Returns:
        Tuple of (best_threshold, best_score)
    """
    from sklearn.metrics import f1_score, precision_recall_curve
    
    if metric == 'f1':
        # Standard F1 optimization
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_score = 0
        best_thresh = 0.5
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            score = f1_score(y_true, y_pred)
            if score > best_score:
                best_score = score
                best_thresh = thresh
        return best_thresh, best_score
    elif metric == 'f1_recall_08':
        # F1 with recall >= 0.8
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        # thresholds has length = len(precision) - 1
        valid = recall[:-1] >= 0.8  # Exclude last recall value (always 0)
        if not valid.any():
            return 0.5, f1_score(y_true, (y_proba >= 0.5).astype(int))
        # Use valid indices for thresholds (which matches precision/recall[:-1])
        valid_precision = precision[:-1][valid]
        valid_recall = recall[:-1][valid]
        valid_thresholds = thresholds[valid]
        f1_scores = 2 * (valid_precision * valid_recall) / (valid_precision + valid_recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_thresh = valid_thresholds[best_idx] if len(valid_thresholds) > 0 else 0.5
        return float(best_thresh), float(f1_scores[best_idx])
    elif metric == 'f1_recall_09':
        # F1 with recall >= 0.9
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        # thresholds has length = len(precision) - 1
        valid = recall[:-1] >= 0.9  # Exclude last recall value (always 0)
        if not valid.any():
            return 0.5, f1_score(y_true, (y_proba >= 0.5).astype(int))
        # Use valid indices for thresholds (which matches precision/recall[:-1])
        valid_precision = precision[:-1][valid]
        valid_recall = recall[:-1][valid]
        valid_thresholds = thresholds[valid]
        f1_scores = 2 * (valid_precision * valid_recall) / (valid_precision + valid_recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_thresh = valid_thresholds[best_idx] if len(valid_thresholds) > 0 else 0.5
        return float(best_thresh), float(f1_scores[best_idx])
    else:
        return 0.5, f1_score(y_true, (y_proba >= 0.5).astype(int))


def expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins for calibration
    
    Returns:
        ECE score
    """
    y_true = y_true.astype(int)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_proba[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


def compute_all_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities (for positive class)
    
    Returns:
        Dictionary of metrics
    """
    y_pred = (y_proba >= 0.5).astype(int)
    
    metrics = {
        'roc_auc': float(roc_auc_score(y_true, y_proba)),
        'pr_auc': float(average_precision_score(y_true, y_proba)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred)),
        'brier': float(brier_score_loss(y_true, y_proba)),
        'ece': expected_calibration_error(y_true, y_proba),
    }
    
    # Best threshold metrics
    best_thresh_f1, f1_best = find_best_threshold(y_true, y_proba, 'f1')
    metrics['f1_best'] = f1_best
    metrics['threshold_f1'] = best_thresh_f1
    
    best_thresh_r08, f1_r08 = find_best_threshold(y_true, y_proba, 'f1_recall_08')
    metrics['f1_recall_08'] = f1_r08
    metrics['threshold_recall_08'] = best_thresh_r08
    
    best_thresh_r09, f1_r09 = find_best_threshold(y_true, y_proba, 'f1_recall_09')
    metrics['f1_recall_09'] = f1_r09
    metrics['threshold_recall_09'] = best_thresh_r09
    
    return metrics


# ==================== Calibration Utilities ====================

class Calibrator:
    """Wrapper for sklearn calibration methods."""
    
    def __init__(self, method: Literal['platt', 'isotonic']):
        self.method = method
        self.calibrator = None
    
    def fit(self, p_val: np.ndarray, y_val: np.ndarray) -> None:
        """Fit calibrator on validation predictions."""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        
        if self.method == 'platt':
            self.calibrator = LogisticRegression()
            self.calibrator.fit(p_val.reshape(-1, 1), y_val)
        elif self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(p_val, y_val)
    
    def transform(self, p: np.ndarray) -> np.ndarray:
        """Transform probabilities using fitted calibrator."""
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted")
        
        if self.method == 'platt':
            return self.calibrator.predict_proba(p.reshape(-1, 1))[:, 1]
        else:  # isotonic
            return self.calibrator.transform(p)
    
    def fit_transform(self, p_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(p_val, y_val)
        return self.transform(p_val)


def fit_calibrator(p_val: np.ndarray, y_val: np.ndarray, method: Literal['platt', 'isotonic']) -> Calibrator:
    """
    Fit a calibrator on validation predictions.
    
    Args:
        p_val: Validation probabilities
        y_val: Validation labels
        method: Calibration method ('platt' or 'isotonic')
    
    Returns:
        Fitted Calibrator object
    """
    calibrator = Calibrator(method)
    calibrator.fit(p_val, y_val)
    return calibrator


# ==================== Plotting Utilities ====================

def plot_pr_roc(y_true: np.ndarray, y_proba: np.ndarray, outdir: Path) -> None:
    """Plot PR and ROC curves."""
    from sklearn.metrics import roc_curve, precision_recall_curve
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax1.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_score(y_true, y_proba):.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ax2.plot(recall, precision, label=f'PR (AUC = {average_precision_score(y_true, y_proba):.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / 'pr_roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_reliability(y_true: np.ndarray, y_proba: np.ndarray, outdir: Path) -> None:
    """Plot reliability diagram (calibration curve)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    CalibrationDisplay.from_predictions(y_true, y_proba, n_bins=10, ax=ax)
    ax.set_title('Calibration Curve (Reliability Diagram)')
    plt.tight_layout()
    plt.savefig(outdir / 'calib_reliability.png', dpi=150, bbox_inches='tight')
    plt.close()


# Initialize seeds on import
set_seeds(RANDOM_STATE)
