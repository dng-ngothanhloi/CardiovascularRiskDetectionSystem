"""
Base trainer protocol and utilities for all model implementations.
"""
from typing import Protocol, Optional
import numpy as np
from pathlib import Path


class BaseTrainer(Protocol):
    """
    Protocol defining the interface for all model trainers.
    All model implementations must conform to this interface.
    """
    
    def fit(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: Optional[np.ndarray] = None,
        y_va: Optional[np.ndarray] = None
    ) -> None:
        """
        Train the model.
        
        Args:
            X_tr: Training features
            y_tr: Training labels
            X_va: Optional validation features
            y_va: Optional validation labels
        """
        ...
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
        
        Returns:
            Predicted probabilities for positive class (shape: [n_samples])
        """
        ...
    
    def save(self, path: Path) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Directory path to save the model
        """
        ...