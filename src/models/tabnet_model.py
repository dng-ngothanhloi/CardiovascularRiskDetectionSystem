"""
TabNet implementation using TensorFlow/Keras.
Attention-based feature selection for tabular data.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

from ..utils import save_keras_model, save_json

# Use main logger to ensure logs are written to file
logger = logging.getLogger('HeartDisease_RiskDiscovery')


class GLU(layers.Layer):
    """Gated Linear Unit activation."""
    
    def call(self, inputs):
        return inputs * tf.nn.sigmoid(inputs)


class FeatureTransformer(layers.Layer):
    """Feature transformation block."""
    
    def __init__(self, n_d: int, n_a: int, shared: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.n_d = n_d
        self.n_a = n_a
        self.shared = shared
        
        # Decision step output dimension
        self.fc_d = layers.Dense(n_d, use_bias=False)
        self.bn_d = layers.BatchNormalization()
        self.glu_d = GLU()
        
        # Attention step output dimension
        self.fc_a = layers.Dense(n_a, use_bias=False)
        self.bn_a = layers.BatchNormalization()
        self.glu_a = GLU()
    
    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'n_d': self.n_d,
            'n_a': self.n_a,
            'shared': self.shared,
        })
        return config
    
    def call(self, inputs, training=None):
        # Decision step
        d = self.fc_d(inputs)
        d = self.bn_d(d, training=training)
        d = self.glu_d(d)
        
        # Attention step
        a = self.fc_a(inputs)
        a = self.bn_a(a, training=training)
        a = self.glu_a(a)
        
        return d, a


class AttentiveTransformer(layers.Layer):
    """Attentive transformer for feature selection."""
    
    def __init__(self, n_features: int, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.fc = layers.Dense(n_features, use_bias=False)
        self.bn = layers.BatchNormalization()
    
    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'n_features': self.n_features,
        })
        return config
    
    def call(self, inputs, prior: tf.Tensor, training=None):
        """
        Args:
            inputs: Feature transformer output (a_step, shape: [batch, n_a])
            prior: Prior attention mask (for sparsity, shape: [batch, n_features])
        Returns:
            mask: Attention mask (shape: [batch, n_features])
        """
        x = self.fc(inputs)  # [batch, n_features]
        x = self.bn(x, training=training)
        eps = tf.constant(1e-6, dtype=prior.dtype)
        prior_safe = tf.maximum(prior, eps)
        x = x * prior_safe  # Element-wise multiply with prior (stabilized)
        mask = tf.nn.softmax(x, axis=1)
        mask = tf.clip_by_value(mask, eps, 1.0)
        denom = tf.reduce_sum(mask, axis=1, keepdims=True) + eps
        mask = mask / denom
        return mask


class TabNetBlock(layers.Layer):
    """Single TabNet step block."""
    
    def __init__(self, n_d: int, n_a: int, n_steps: int, n_features: int, gamma: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.n_features = n_features
        self.gamma = gamma
        
        # Shared feature transformer
        self.shared_ft = FeatureTransformer(n_d, n_a, shared=True)
        
        # Step-specific transformers
        self.ft_steps = [FeatureTransformer(n_d, n_a, shared=False) for _ in range(n_steps)]
        self.att_steps = [AttentiveTransformer(n_features) for _ in range(n_steps)]
    
    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'n_d': self.n_d,
            'n_a': self.n_a,
            'n_steps': self.n_steps,
            'n_features': self.n_features,
            'gamma': self.gamma,
        })
        return config
    
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        n_features = inputs.shape[-1]
        
        # Initialize
        dtype = inputs.dtype
        prior = tf.ones((batch_size, n_features), dtype=dtype) / tf.cast(n_features, dtype)
        output_aggregated = tf.zeros((batch_size, self.n_d), dtype=dtype)
        masks = []
        eps = tf.constant(1e-6, dtype=inputs.dtype)
        
        # Shared feature transformation
        d_shared, a_shared = self.shared_ft(inputs, training=training)
        
        # Step-by-step processing
        for step in range(self.n_steps):
            # Feature transformation
            d_step, a_step = self.ft_steps[step](d_shared, training=training)
            
            # Attention mask
            mask = self.att_steps[step](a_step, prior, training=training)
            masks.append(mask)
            
            # Update prior (sparsity)
            prior = tf.maximum(prior, eps)
            prior = prior * (self.gamma - mask)
            prior = tf.maximum(prior, eps)
            
            # Aggregate decision output (mask is [batch, n_features], but we need to select features)
            # Use mask to weight the decision output
            # In TabNet, we use the mask to select which features contribute
            # Here we aggregate d_step (shape [batch, n_d]) weighted by mask sum
            mask_sum = tf.reduce_sum(mask, axis=1, keepdims=True) + eps  # [batch, 1]
            output_aggregated += tf.nn.relu(d_step) * mask_sum
        
        return output_aggregated, masks


class TabNetModel(keras.Model):
    """TabNet model for tabular classification."""
    
    def __init__(
        self,
        input_dim: int,
        n_d: int = 16,
        n_a: int = 16,
        n_steps: int = 5,
        gamma: float = 1.5,
        lambda_sparse: float = 1e-4,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        
        # Batch normalization
        self.bn = layers.BatchNormalization()
        
        # TabNet block
        self.tabnet = TabNetBlock(n_d, n_a, n_steps, input_dim, gamma)
        
        # Final classifier
        self.fc = layers.Dense(1, activation='sigmoid')
        
        if dropout > 0:
            self.dropout = layers.Dropout(dropout)
        else:
            self.dropout = None
        
        # Store masks for interpretability
        self.attention_masks = []
    
    def call(self, inputs, training=None):
        # Normalize inputs
        x = self.bn(inputs, training=training)
        
        # TabNet processing
        output, masks = self.tabnet(x, training=training)
        self.attention_masks = masks
        
        # Compute sparse loss (for regularization)
        if training and masks and len(masks) > 0:
            # Sparse loss: mean of entropy of masks (encourage sparsity)
            # Initialize as Tensor to avoid shape issues
            sparse_loss = tf.constant(0.0, dtype=tf.float32)
            valid_masks_count = 0
            for mask in masks:
                # Entropy: -sum(p * log(p + eps))
                # Ensure mask is a proper tensor
                if mask is not None:
                    eps = 1e-15
                    # Entropy per sample: -sum(p * log(p + eps)) over features
                    entropy = -tf.reduce_sum(mask * tf.math.log(mask + eps), axis=1)  # [batch]
                    # Mean entropy across batch
                    mean_entropy = tf.reduce_mean(entropy)  # scalar
                    sparse_loss = sparse_loss + mean_entropy
                    valid_masks_count += 1
            
            # Average over number of valid masks and add as regularization loss
            if valid_masks_count > 0:
                final_loss = self.lambda_sparse * sparse_loss / tf.cast(valid_masks_count, tf.float32)
                self.add_loss(final_loss)
        
        # Dropout
        if self.dropout:
            output = self.dropout(output, training=training)
        
        # Classification
        predictions = self.fc(output)
        
        return predictions
    
    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'n_d': self.n_d,
            'n_a': self.n_a,
            'n_steps': self.n_steps,
            'gamma': self.gamma,
            'lambda_sparse': self.lambda_sparse,
            'dropout': self.dropout.rate if self.dropout is not None else 0.0,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        # Extract dropout from config
        dropout = config.pop('dropout', 0.0)
        # Create model instance
        instance = cls(**config)
        return instance
    
    def get_attention_masks(self) -> List[np.ndarray]:
        """Get attention masks for interpretability."""
        if not self.attention_masks:
            return []
        # Convert masks to numpy arrays, handling both eager and symbolic tensors
        numpy_masks = []
        for mask in self.attention_masks:
            if hasattr(mask, 'numpy'):
                # Eager tensor - can convert directly
                try:
                    numpy_masks.append(mask.numpy())
                except (AttributeError, NotImplementedError):
                    # Symbolic tensor - skip or use alternative method
                    logger.warning("Cannot convert symbolic tensor to numpy, skipping mask")
                    continue
            elif isinstance(mask, np.ndarray):
                # Already numpy array
                numpy_masks.append(mask)
            else:
                # Try to convert using tf.keras.backend
                try:
                    import tensorflow as tf
                    if tf.executing_eagerly():
                        numpy_masks.append(mask.numpy() if hasattr(mask, 'numpy') else mask)
                    else:
                        logger.warning("Cannot convert symbolic tensor to numpy in graph mode, skipping mask")
                        continue
                except Exception as e:
                    logger.warning(f"Error converting mask to numpy: {e}, skipping mask")
                    continue
        return numpy_masks


class TabNetTrainer:
    """
    TabNet trainer with early stopping and TensorBoard logging.
    """
    
    def __init__(
        self,
        input_dim: int,
        params: Dict[str, Any],
        seed: int = 42,
        monitor: str = 'val_pr_auc',
        patience: int = 15,
        min_delta: float = 0.0005
    ):
        """
        Initialize TabNet trainer.
        
        Args:
            input_dim: Input feature dimension
            params: Model parameters (n_d, n_a, n_steps, etc.)
            seed: Random seed
            monitor: Metric to monitor for early stopping
            patience: Early stopping patience
            min_delta: Minimum change to qualify as improvement
        """
        self.input_dim = input_dim
        self.params = params
        self.seed = seed
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        
        # Set seed
        tf.random.set_seed(seed)
        
        # Build model
        self.model = TabNetModel(
            input_dim=input_dim,
            n_d=params.get('n_d', 16),
            n_a=params.get('n_a', 16),
            n_steps=params.get('n_steps', 5),
            gamma=params.get('gamma', 1.5),
            lambda_sparse=params.get('lambda_sparse', 1e-4),
            dropout=params.get('dropout', 0.0)
        )
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params.get('lr', 1e-3)),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.is_fitted = False
        self.history = None
        logger.info(f"TabNet initialized: input_dim={input_dim}, params={params}")
    
    def fit(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: Optional[np.ndarray] = None,
        y_va: Optional[np.ndarray] = None,
        tensorboard_dir: Optional[Path] = None,
        early_stopping: bool = True,
        max_epochs: Optional[int] = None,
    ) -> None:
        """
        Train TabNet model.
        
        Args:
            X_tr: Training features
            y_tr: Training labels
            X_va: Optional validation features
            y_va: Optional validation labels
            tensorboard_dir: Optional TensorBoard log directory
            early_stopping: If False, train fixed epochs (for full-Train final refit)
            max_epochs: Override params['epochs'] when set
        """
        logger.info(f"Training TabNet on {len(X_tr)} samples")
        
        # Prepare callbacks
        cb_list = []
        es_callback = None
        self.best_epoch = None
        
        # Early stopping (custom metric monitoring)
        if early_stopping and X_va is not None and y_va is not None:
            # Custom early stopping callback for PR-AUC
            class PRAUCEarlyStopping(callbacks.Callback):
                def __init__(self, monitor='val_pr_auc', patience=15, min_delta=0.0005, X_val=None, y_val=None):
                    super().__init__()
                    self.monitor = monitor
                    self.patience = patience
                    self.min_delta = min_delta
                    self.best = -np.inf
                    self.wait = 0
                    self.best_epoch = 0
                    self.X_val = X_val
                    self.y_val = y_val
                
                def on_epoch_end(self, epoch, logs=None):
                    from sklearn.metrics import average_precision_score
                    import numpy as np
                    val_pred = self.model.predict(self.X_val, verbose=0).flatten()
                    # Replace NaN/inf in predictions to prevent sklearn errors
                    val_pred = np.nan_to_num(val_pred, nan=0.5, posinf=1.0, neginf=0.0)
                    # Clip to [0, 1] for probability
                    val_pred = np.clip(val_pred, 0.0, 1.0)
                    current = average_precision_score(self.y_val, val_pred)
                    
                    if current > self.best + self.min_delta:
                        self.best = current
                        self.wait = 0
                        self.best_epoch = int(epoch)
                    else:
                        self.wait += 1
                    
                    if logs is not None:
                        logs['val_pr_auc'] = current
                    if self.wait >= self.patience:
                        self.model.stop_training = True
            
            es_callback = PRAUCEarlyStopping(
                monitor=self.monitor, 
                patience=self.patience, 
                min_delta=self.min_delta,
                X_val=X_va,
                y_val=y_va
            )
            cb_list.append(es_callback)
        
        # TensorBoard
        if tensorboard_dir:
            cb_list.append(callbacks.TensorBoard(
                log_dir=str(tensorboard_dir),
                histogram_freq=1,
                write_graph=True
            ))
        
        # Training
        use_val = early_stopping and X_va is not None and y_va is not None
        validation_data = (X_va, y_va) if use_val else None
        epochs = int(max_epochs if max_epochs is not None else self.params.get('epochs', 80))
        
        self.history = self.model.fit(
            X_tr, y_tr,
            validation_data=validation_data,
            batch_size=self.params.get('batch_size', 256),
            epochs=epochs,
            callbacks=cb_list if cb_list else None,
            verbose=1
        )
        
        if es_callback is not None:
            self.best_epoch = int(es_callback.best_epoch) + 1  # 1-based epoch count
        elif self.history is not None and hasattr(self.history, 'epoch'):
            self.best_epoch = int(len(self.history.epoch))
        
        self.is_fitted = True
        logger.info("TabNet training completed")
    
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
        # Clear previous masks before prediction
        self.model.attention_masks = []
        predictions = self.model.predict(X, verbose=0)
        # Masks are now stored in self.model.attention_masks
        return predictions.flatten()
    
    def predict_with_attention(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Predict probabilities and return attention masks.
        
        Args:
            X: Input features
        
        Returns:
            (predictions, attention_masks)
            predictions: Probabilities for positive class (shape: [n_samples])
            attention_masks: List of attention masks from each step
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        # Clear previous masks
        self.model.attention_masks = []
        predictions = self.model.predict(X, verbose=0)
        masks = self.model.get_attention_masks()
        return predictions.flatten(), masks
    
    def get_attention_masks(self) -> List[np.ndarray]:
        """Get attention masks for interpretability."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get attention masks")
        return self.model.get_attention_masks()
    
    def save(self, path: Path) -> None:
        """
        Save model to disk.
        
        Args:
            path: Directory path to save the model
        
        Note:
            Saves weights only (not full model) to avoid serialization issues.
            Model structure is defined by params.json, so we can rebuild it.
        """
        path.mkdir(parents=True, exist_ok=True)
        # Keras requires .weights.h5 extension for save_weights()
        model_path = path / 'model.weights.h5'
        
        # Save weights only (Solution 3: Best long-term approach)
        # This avoids serialization issues with custom layers
        self.model.save_weights(str(model_path))
        
        # Save attention masks (only if we have valid numpy arrays)
        masks = self.get_attention_masks()
        if masks and len(masks) > 0:
            try:
                # Ensure all masks are numpy arrays and have compatible shapes
                numpy_masks = []
                for mask in masks:
                    if isinstance(mask, np.ndarray):
                        numpy_masks.append(mask)
                    else:
                        logger.warning(f"Skipping non-numpy mask: {type(mask)}")
                
                if numpy_masks:
                    masks_path = path / 'attention_masks.npy'
                    # Save as list of arrays (not stacked) to handle variable batch sizes
                    np.save(masks_path, numpy_masks, allow_pickle=True)
                    logger.info(f"Attention masks saved to {masks_path} ({len(numpy_masks)} masks)")
                else:
                    logger.warning("No valid numpy masks to save")
            except Exception as e:
                logger.warning(f"Failed to save attention masks: {e}")
        
        # Save parameters
        # Persist input dimension so downstream loaders rebuild the model correctly
        self.params['input_dim'] = self.input_dim
        params_path = path / 'params.json'
        save_json(self.params, str(params_path))
        
        logger.info(f"TabNet weights saved to {model_path} (weights only format: .weights.h5)")