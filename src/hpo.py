"""
Hyperparameter optimization using Optuna.
Target metric: Composite score (weighted combination of ROC-AUC, PR-AUC, and Accuracy).
"""
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from typing import Dict, Any, Tuple, List
import logging
import tensorflow as tf

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score
)

from .models.random_forest import RandomForestTrainer
from .models.tabnet_model import TabNetTrainer
from .models.ensemble_joint import JointVAETabNetTrainer

logger = logging.getLogger(__name__)


def compute_composite_score(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    weights: List[float] = None
) -> float:
    """
    Compute composite score from ROC-AUC, PR-AUC, and Accuracy.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities (for positive class)
        weights: Weights for [roc_auc, pr_auc, accuracy]. Default: [0.4, 0.4, 0.2]
    
    Returns:
        Composite score (weighted sum)
    """
    if weights is None:
        weights = [0.4, 0.4, 0.2]
    
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    y_pred = (y_proba >= 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    
    composite = weights[0] * roc_auc + weights[1] * pr_auc + weights[2] * accuracy
    return composite


def objective_rf(
    trial: optuna.Trial,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    cfg: Dict[str, Any],
    seed: int
) -> float:
    """
    Optuna objective for RandomForest.
    
    Args:
        trial: Optuna trial
        X_tr: Training features
        y_tr: Training labels
        X_va: Validation features
        y_va: Validation labels
        cfg: Configuration dictionary
        seed: Random seed
    
    Returns:
        Composite score (weighted combination of ROC-AUC, PR-AUC, Accuracy) on validation set
    """
    # Search space - expanded for better optimization
    use_max_depth = trial.suggest_categorical('use_max_depth', [True, False])
    max_depth = trial.suggest_int('max_depth', 6, 30) if use_max_depth else None  # Expanded range
    
    use_max_features_cat = trial.suggest_categorical('use_max_features_cat', [True, False])
    if use_max_features_cat:
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    else:
        max_features = trial.suggest_float('max_features_float', 0.2, 1.0)  # Expanded range

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500, step=50),  # Expanded range with smaller step
        'max_depth': max_depth,
        'max_features': max_features,
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),  # Expanded range
        'class_weight': cfg.get('rf', {}).get('class_weight', 'balanced')
    }
    
    # Train model
    trainer = RandomForestTrainer(params, seed=seed)
    trainer.fit(X_tr, y_tr)
    
    # Predict on validation
    p_va = trainer.predict_proba(X_va)
    
    # Compute composite score
    hpo_cfg = cfg.get('modeling', {}).get('hpo', {})
    weights = hpo_cfg.get('composite_weights', [0.4, 0.4, 0.2])
    composite = compute_composite_score(y_va, p_va, weights)
    
    return composite


def objective_tabnet(
    trial: optuna.Trial,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    cfg: Dict[str, Any],
    seed: int
) -> float:
    """
    Optuna objective for TabNet.
    
    Args:
        trial: Optuna trial
        X_tr: Training features
        y_tr: Training labels
        X_va: Validation features
        y_va: Validation labels
        cfg: Configuration dictionary
        seed: Random seed
    
    Returns:
        Composite score (weighted combination of ROC-AUC, PR-AUC, Accuracy) on validation set
    """
    input_dim = X_tr.shape[1]
    
    # Search space - expanded for better optimization
    n_d = trial.suggest_int('n_d', 4, 32, step=4)  # Expanded range: 4, 8, 12, 16, 20, 24, 28, 32
    params = {
        'n_d': n_d,
        'n_a': n_d,  # keep decision and attention dimensions in sync
        'n_steps': trial.suggest_int('n_steps', 3, 5),
        'gamma': trial.suggest_float('gamma', 1.0, 2.0),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
        'epochs': cfg.get('tabnet', {}).get('epochs', 80),  # Use config epochs
        'lr': trial.suggest_float('lr', 1e-4, 3e-3, log=True),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3)
    }
    
    # Early stopping config
    early_stopping = cfg.get('modeling', {}).get('early_stopping', {})
    
    # Train model
    trainer = TabNetTrainer(
        input_dim=input_dim,
        params=params,
        seed=seed,
        monitor=early_stopping.get('monitor', 'val_pr_auc'),
        patience=early_stopping.get('patience', 15),
        min_delta=early_stopping.get('min_delta', 0.0005)
    )
    try:
        trainer.fit(X_tr, y_tr, X_va, y_va)
    except tf.errors.InvalidArgumentError as err:
        msg = (
            f"TabNet trial {trial.number} pruned due to InvalidArgumentError: {err}"
        )
        logger.warning(msg)
        raise optuna.exceptions.TrialPruned(msg) from err
    
    # Predict on validation
    p_va = trainer.predict_proba(X_va)
    
    # Compute composite score
    hpo_cfg = cfg.get('modeling', {}).get('hpo', {})
    weights = hpo_cfg.get('composite_weights', [0.4, 0.4, 0.2])
    composite = compute_composite_score(y_va, p_va, weights)
    
    return composite


def objective_vae_tabnet(
    trial: optuna.Trial,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    cfg: Dict[str, Any],
    seed: int,
    joint: bool = False
) -> float:
    """
    Optuna objective for Joint VAE-TabNet.
    
    Args:
        trial: Optuna trial
        X_tr: Training features
        y_tr: Training labels
        X_va: Validation features
        y_va: Validation labels
        cfg: Configuration dictionary
        seed: Random seed
        joint: If True, use joint training; else two-stage
    
    Returns:
        Composite score (weighted combination of ROC-AUC, PR-AUC, Accuracy) on validation set
    """
    input_dim = X_tr.shape[1]
    
    # Search space
    params = {
        'latent_dim': trial.suggest_categorical('latent_dim', [8, 16, 24, 32]),
        'width': trial.suggest_int('width', 128, 512, step=64),
        'depth': trial.suggest_int('depth', 2, 4),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'beta': trial.suggest_float('beta', 0.5, 4.0),
        'lr': trial.suggest_float('lr', 1e-4, 3e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
        'epochs': cfg.get('vae', {}).get('epochs', 50)
    }
    
    if joint:
        # Joint training: only VAE and classifier parameters
        params['lambda_cls'] = trial.suggest_float('lambda_cls', 0.5, 3.0)
    else:
        # Two-stage training: include TabNet parameters
        params.update({
            'n_d': trial.suggest_int('n_d', 8, 32, step=4),
            'n_a': trial.suggest_int('n_a', 8, 32, step=4),
            'n_steps': trial.suggest_int('n_steps', 3, 7),
            'gamma': trial.suggest_float('gamma', 1.0, 2.0),
            'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-5, 1e-3, log=True)
        })
    
    # Early stopping config
    early_stopping = cfg.get('modeling', {}).get('early_stopping', {})
    
    # Train model
    trainer = JointVAETabNetTrainer(
        input_dim=input_dim,
        params=params,
        seed=seed,
        joint_training=joint,
        monitor=early_stopping.get('monitor', 'val_pr_auc'),
        patience=early_stopping.get('patience', 15),
        min_delta=early_stopping.get('min_delta', 0.0005)
    )
    trainer.fit(X_tr, y_tr, X_va, y_va)
    
    # Predict on validation
    p_va = trainer.predict_proba(X_va)
    
    # Compute composite score
    hpo_cfg = cfg.get('modeling', {}).get('hpo', {})
    weights = hpo_cfg.get('composite_weights', [0.4, 0.4, 0.2])
    composite = compute_composite_score(y_va, p_va, weights)
    
    return composite


def resolve_params(
    model_name: str,
    raw_params: Dict[str, Any],
    cfg: Dict[str, Any],
    joint: bool = False
) -> Dict[str, Any]:
    """
    Map raw Optuna trial params to trainer-ready hyperparameters.
    
    Optuna's study.best_params keeps helper keys (use_max_depth, max_features_float,
    etc.). Trainers need the resolved values used during the objective trials.
    """
    raw = dict(raw_params or {})

    if model_name == 'rf':
        # Already-resolved config defaults / legacy resolved dicts
        if 'n_estimators' in raw and 'use_max_depth' not in raw and 'use_max_features_cat' not in raw:
            resolved = {
                'n_estimators': int(raw['n_estimators']),
                'max_depth': raw.get('max_depth'),
                'max_features': raw.get('max_features', 'sqrt'),
                'min_samples_leaf': int(raw.get('min_samples_leaf', 1)),
                'class_weight': raw.get(
                    'class_weight',
                    cfg.get('rf', {}).get('class_weight', 'balanced')
                ),
            }
            return resolved

        use_max_depth = raw.get('use_max_depth', True)
        max_depth = int(raw['max_depth']) if (use_max_depth and 'max_depth' in raw) else None

        use_max_features_cat = raw.get('use_max_features_cat', True)
        if use_max_features_cat:
            max_features = raw.get('max_features', 'sqrt')
        else:
            max_features = raw.get('max_features_float', raw.get('max_features', 0.5))

        return {
            'n_estimators': int(raw.get('n_estimators', 500)),
            'max_depth': max_depth,
            'max_features': max_features,
            'min_samples_leaf': int(raw.get('min_samples_leaf', 1)),
            'class_weight': cfg.get('rf', {}).get('class_weight', 'balanced'),
        }

    if model_name == 'tabnet':
        tabnet_cfg = cfg.get('tabnet', {})
        n_d = int(raw.get('n_d', tabnet_cfg.get('n_d', 16)))
        n_a = int(raw.get('n_a', n_d))  # objective keeps n_a == n_d
        return {
            'n_d': n_d,
            'n_a': n_a,
            'n_steps': int(raw.get('n_steps', tabnet_cfg.get('n_steps', 5))),
            'gamma': float(raw.get('gamma', tabnet_cfg.get('gamma', 1.5))),
            'lambda_sparse': float(raw.get('lambda_sparse', tabnet_cfg.get('lambda_sparse', 1e-4))),
            'batch_size': int(raw.get('batch_size', tabnet_cfg.get('batch_size', 256))),
            'epochs': int(raw.get('epochs', tabnet_cfg.get('epochs', 80))),
            'lr': float(raw.get('lr', tabnet_cfg.get('lr', 1e-3))),
            'dropout': float(raw.get('dropout', tabnet_cfg.get('dropout', 0.0))),
        }

    if model_name == 'vae_tabnet':
        vae_cfg = cfg.get('vae', {})
        resolved = {
            'latent_dim': int(raw.get('latent_dim', vae_cfg.get('latent_dim', 16))),
            'width': int(raw.get('width', vae_cfg.get('width', 256))),
            'depth': int(raw.get('depth', vae_cfg.get('depth', 3))),
            'dropout': float(raw.get('dropout', vae_cfg.get('dropout', 0.1))),
            'beta': float(raw.get('beta', vae_cfg.get('beta', 1.0))),
            'lr': float(raw.get('lr', vae_cfg.get('lr', 1e-3))),
            'batch_size': int(raw.get('batch_size', vae_cfg.get('batch_size', 128))),
            'epochs': int(raw.get('epochs', vae_cfg.get('epochs', 50))),
        }
        if joint:
            resolved['lambda_cls'] = float(
                raw.get('lambda_cls', vae_cfg.get('lambda_cls', 1.0))
            )
        else:
            n_d = int(raw.get('n_d', 16))
            resolved.update({
                'n_d': n_d,
                'n_a': int(raw.get('n_a', n_d)),
                'n_steps': int(raw.get('n_steps', 5)),
                'gamma': float(raw.get('gamma', 1.5)),
                'lambda_sparse': float(raw.get('lambda_sparse', 1e-4)),
            })
        return resolved

    raise ValueError(f"Unknown model name: {model_name}")


def hpo_search(
    model_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    cfg: Dict[str, Any],
    seed: int = 42,
    joint: bool = False
) -> Dict[str, Any]:
    """
    Run hyperparameter optimization for a model.
    
    Args:
        model_name: Model name ('rf', 'tabnet', 'vae_tabnet')
        X_tr: Training features
        y_tr: Training labels
        X_va: Validation features
        y_va: Validation labels
        cfg: Configuration dictionary
        seed: Random seed
        joint: For vae_tabnet, whether to use joint training
    
    Returns:
        Dict with raw_optuna_params and resolved_model_params
    """
    hpo_cfg = cfg.get('modeling', {}).get('hpo', {})
    if not hpo_cfg.get('enabled', True):
        # Return default params from config (already trainer-ready)
        if model_name == 'rf':
            raw_params = dict(cfg.get('rf', {}))
        elif model_name == 'tabnet':
            raw_params = dict(cfg.get('tabnet', {}))
        elif model_name == 'vae_tabnet':
            raw_params = dict(cfg.get('vae', {}))
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        resolved = resolve_params(model_name, raw_params, cfg, joint=joint)
        logger.info(f"HPO disabled. Using config defaults. Resolved params: {resolved}")
        return {
            'raw_optuna_params': raw_params,
            'resolved_model_params': resolved,
        }
    
    n_trials = hpo_cfg.get('n_trials', 30)
    timeout_min = hpo_cfg.get('timeout_min', 45)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner() if hpo_cfg.get('pruner') == 'median' else None
    )
    
    # Select objective function
    if model_name == 'rf':
        objective = lambda trial: objective_rf(trial, X_tr, y_tr, X_va, y_va, cfg, seed)
    elif model_name == 'tabnet':
        objective = lambda trial: objective_tabnet(trial, X_tr, y_tr, X_va, y_va, cfg, seed)
    elif model_name == 'vae_tabnet':
        objective = lambda trial: objective_vae_tabnet(trial, X_tr, y_tr, X_va, y_va, cfg, seed, joint)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Optimize
    logger.info(f"Starting HPO for {model_name} (n_trials={n_trials}, timeout={timeout_min}min)")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_min * 60 if timeout_min else None,
        show_progress_bar=True
    )
    
    raw_params = dict(study.best_params)
    resolved = resolve_params(model_name, raw_params, cfg, joint=joint)

    logger.info(f"HPO completed. Best composite score: {study.best_value:.4f}")
    logger.info(f"Raw Optuna params: {raw_params}")
    logger.info(f"Resolved model params: {resolved}")

    return {
        'raw_optuna_params': raw_params,
        'resolved_model_params': resolved,
    }
