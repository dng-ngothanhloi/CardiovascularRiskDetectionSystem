"""
RandomForest trainer using scikit-learn.
"""
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

from ..utils import save_joblib, save_json

# Use main logger to ensure logs are written to file
logger = logging.getLogger('HeartDisease_RiskDiscovery')


class RandomForestTrainer:
    """
    RandomForest trainer with HPO support.
    """
    
    def __init__(self, params: Dict[str, Any], seed: int = 42):
        """
        Initialize RandomForest trainer.
        
        Args:
            params: Model parameters (n_estimators, max_depth, etc.)
            seed: Random seed
        """
        self.params = params.copy()
        
        # Filter out Optuna helper parameters
        valid_params = {
            'n_estimators', 'max_depth', 'max_features', 
            'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes',
            'min_impurity_decrease', 'bootstrap', 'oob_score',
            'class_weight', 'n_jobs', 'random_state', 'verbose', 'warm_start'
        }
        
        filtered_params = {k: v for k, v in self.params.items() if k in valid_params}
        filtered_params['random_state'] = seed
        filtered_params.setdefault('n_jobs', -1)
        filtered_params.setdefault('class_weight', 'balanced')
        
        self.model = RandomForestClassifier(**filtered_params)
        self.is_fitted = False
        logger.info(f"RandomForest initialized with resolved params: {filtered_params}")
    
    def fit(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: Optional[np.ndarray] = None,
        y_va: Optional[np.ndarray] = None
    ) -> None:
        """
        Train RandomForest model.
        
        Args:
            X_tr: Training features
            y_tr: Training labels
            X_va: Optional validation features (not used for RF)
            y_va: Optional validation labels (not used for RF)
        """
        logger.info(f"Training RandomForest on {len(X_tr)} samples")
        self.model.fit(X_tr, y_tr)
        self.is_fitted = True
        logger.info("RandomForest training completed")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
        
        Returns:
            Probabilities for positive class (shape: [n_samples])
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, path: Path) -> None:
        """
        Save model to disk.
        
        Args:
            path: Directory path to save the model
        """
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / 'model.joblib'
        save_joblib(self.model, str(model_path))
        
        # Save parameters
        params_path = path / 'params.json'
        save_json(self.params, str(params_path))
        
        # Save feature importance
        if self.is_fitted:
            try:
                importance = self.get_feature_importance()
                importance_dict = {
                    'feature_importance': importance.tolist(),
                    'n_features': len(importance),
                    'top_k': int(np.sum(importance > np.mean(importance)))
                }
                importance_path = path / 'feature_importance.json'
                save_json(importance_dict, str(importance_path))
                logger.info(f"Feature importance saved to {importance_path}")
            except Exception as e:
                logger.warning(f"Failed to save feature importance: {e}")
        
        logger.info(f"RandomForest saved to {model_path}")
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        return self.model.feature_importances_