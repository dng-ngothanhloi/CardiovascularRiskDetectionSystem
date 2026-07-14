"""
Joint VAE-TabNet ensemble model.
Supports both two-stage (sequential) and joint training modes.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from sklearn.preprocessing import StandardScaler

from .vae_model import VAETrainer
from .tabnet_model import TabNetTrainer, TabNetModel
from ..utils import save_keras_model, save_json, save_joblib

# Use main logger to ensure logs are written to file
logger = logging.getLogger('HeartDisease_RiskDiscovery')


class JointVAETabNetModel(keras.Model):
    """
    Joint VAE-TabNet model with multi-output (reconstruction + classification).
    Loss: L_joint = L_rec + beta * L_KL + lambda * L_CE
    Simplified version for joint training.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        width: int = 256,
        depth: int = 3,
        beta: float = 1.0,
        lambda_cls: float = 1.0,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.lambda_cls = lambda_cls
        
        # Encoder layers
        self.encoder_layers = []
        for i in range(depth):
            self.encoder_layers.append(layers.Dense(width, activation='relu', name=f'encoder_dense_{i}'))
            if dropout > 0:
                self.encoder_layers.append(layers.Dropout(dropout, name=f'encoder_dropout_{i}'))
        
        self.z_mean_layer = layers.Dense(latent_dim, name='z_mean')
        self.z_log_var_layer = layers.Dense(latent_dim, name='z_log_var')
        
        # Sampling layer
        from .vae_model import Sampling
        self.sampling = Sampling(name='z')
        
        # Decoder layers
        self.decoder_layers = []
        for i in range(depth):
            self.decoder_layers.append(layers.Dense(width, activation='relu', name=f'decoder_dense_{i}'))
            if dropout > 0:
                self.decoder_layers.append(layers.Dropout(dropout, name=f'decoder_dropout_{i}'))
        
        self.decoder_output = layers.Dense(input_dim, name='decoder_output')
        
        # Classifier (simple MLP on latent)
        self.classifier_layers = [
            layers.BatchNormalization(name='classifier_bn'),
            layers.Dense(32, activation='relu', name='classifier_hidden'),
            layers.Dense(1, activation='sigmoid', name='classifier_output')
        ]
    
    def encode(self, x, training=None):
        """Encode input to latent representation."""
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        z = self.sampling([z_mean, z_log_var])
        return z_mean, z_log_var, z
    
    def decode(self, z, training=None):
        """Decode latent representation."""
        x = z
        for layer in self.decoder_layers:
            x = layer(x, training=training)
        return self.decoder_output(x)
    
    def classify(self, z, training=None):
        """Classify from latent representation."""
        x = z
        for layer in self.classifier_layers:
            x = layer(x, training=training)
        return x
    
    def call(self, inputs, training=None):
        """Forward pass."""
        z_mean, z_log_var, z = self.encode(inputs, training=training)
        reconstructed = self.decode(z, training=training)
        classified = self.classify(z, training=training)
        
        # Store for loss computation
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        
        return [reconstructed, classified]


class JointVAETabNetTrainer:
    """
    Joint VAE-TabNet trainer supporting both two-stage and joint training.
    """
    
    def __init__(
        self,
        input_dim: int,
        params: Dict[str, Any],
        seed: int = 42,
        joint_training: bool = False,
        monitor: str = 'val_pr_auc',
        patience: int = 15,
        min_delta: float = 0.0005
    ):
        """
        Initialize Joint VAE-TabNet trainer.
        
        Args:
            input_dim: Input feature dimension
            params: Model parameters (latent_dim, width, depth, etc.)
            seed: Random seed
            joint_training: If True, use joint training; else two-stage
            monitor: Metric to monitor for early stopping
            patience: Early stopping patience
            min_delta: Minimum change to qualify as improvement
        """
        self.input_dim = input_dim
        self.params = params
        self.seed = seed
        self.joint_training = joint_training
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        
        # Set seed
        tf.random.set_seed(seed)
        
        if joint_training:
            # Joint training: build multi-output model
            # Note: JointVAETabNetModel does not accept TabNet parameters (n_d, n_a, n_steps, gamma, lambda_sparse)
            # These are only used in two-stage training
            
            # Initialize StandardScaler for normalization (CRITICAL FIX)
            self.scaler = StandardScaler()
            
            # Log VAE parameters for joint training
            vae_params = {
                'latent_dim': params.get('latent_dim', 16),
                'width': params.get('width', 256),
                'depth': params.get('depth', 3),
                'beta': params.get('beta', 1.0),
                'lambda_cls': params.get('lambda_cls', 1.0),
                'dropout': params.get('dropout', 0.1),
                'lr': params.get('lr', 1e-3),
                'batch_size': params.get('batch_size', 256),
                'epochs': params.get('epochs', 80)
            }
            logger.info("=" * 70)
            logger.info("JOINT TRAINING MODE - VAE PARAMETERS")
            logger.info("=" * 70)
            logger.info(f"VAE Parameters passed to JointVAETabNetModel:")
            for key, value in vae_params.items():
                logger.info(f"  {key}: {value}")
            logger.info("=" * 70)
            
            self.model = JointVAETabNetModel(
                input_dim=input_dim,
                latent_dim=vae_params['latent_dim'],
                width=vae_params['width'],
                depth=vae_params['depth'],
                beta=vae_params['beta'],
                lambda_cls=vae_params['lambda_cls'],
                dropout=vae_params['dropout']
            )
            
            # Custom training step for joint loss
            # Note: This is simplified; full implementation would use custom training loop
            # For now, we use separate losses with weights
            # In practice, joint loss = L_rec + beta*L_KL + lambda*L_CE would need custom training loop
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=vae_params['lr']),
                loss=['mse', 'binary_crossentropy'],
                loss_weights=[1.0, vae_params['lambda_cls']],
                metrics=[['mse'], ['accuracy']]
            )
        else:
            # Two-stage: separate VAE and TabNet
            self.vae = VAETrainer(input_dim, params, seed)
            
            # TabNet will be built after VAE training with latent_dim as input
            self.tabnet = None
            self.tabnet_params = {
                'n_d': params.get('n_d', 16),
                'n_a': params.get('n_a', 16),
                'n_steps': params.get('n_steps', 5),
                'gamma': params.get('gamma', 1.5),
                'lambda_sparse': params.get('lambda_sparse', 1e-4),
                'dropout': params.get('dropout', 0.0),
                'batch_size': params.get('batch_size', 256),
                'epochs': params.get('epochs', 80),
                'lr': params.get('lr', 1e-3)
            }
        
        self.is_fitted = False
        self.history = None
        logger.info(f"JointVAETabNet initialized: input_dim={input_dim}, joint_training={joint_training}")
    
    def fit(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: Optional[np.ndarray] = None,
        y_va: Optional[np.ndarray] = None,
        tensorboard_dir: Optional[Path] = None
    ) -> None:
        """
        Train Joint VAE-TabNet model.
        
        Args:
            X_tr: Training features
            y_tr: Training labels
            X_va: Optional validation features
            y_va: Optional validation labels
            tensorboard_dir: Optional TensorBoard log directory
        """
        logger.info(f"Training Joint VAE-TabNet on {len(X_tr)} samples (joint={self.joint_training})")
        
        if self.joint_training:
            # Joint training
            cb_list = []
            
            # Early stopping (simplified - would need custom callback for joint loss)
            if X_va is not None and y_va is not None:
                class PRAUCEarlyStopping(callbacks.Callback):
                    def __init__(self, monitor='val_pr_auc', patience=15, min_delta=0.0005, X_val=None, y_val=None):
                        super().__init__()
                        self.monitor = monitor
                        self.patience = patience
                        self.min_delta = min_delta
                        self.best = -np.inf
                        self.wait = 0
                        self.X_val = X_val
                        self.y_val = y_val
                    
                    def on_epoch_end(self, epoch, logs=None):
                        from sklearn.metrics import average_precision_score
                        _, val_pred = self.model.predict(self.X_val, verbose=0)
                        current = average_precision_score(self.y_val, val_pred.flatten())
                        
                        if current > self.best + self.min_delta:
                            self.best = current
                            self.wait = 0
                        else:
                            self.wait += 1
                        
                        if logs is not None:
                            logs['val_pr_auc'] = current
                        if self.wait >= self.patience:
                            self.model.stop_training = True
                
                cb_list.append(PRAUCEarlyStopping(
                    monitor=self.monitor,
                    patience=self.patience,
                    min_delta=self.min_delta,
                    X_val=X_va,
                    y_val=y_va
                ))
            
            if tensorboard_dir:
                cb_list.append(callbacks.TensorBoard(
                    log_dir=str(tensorboard_dir),
                    histogram_freq=1,
                    write_graph=True
                ))
            
            # CRITICAL FIX: Normalize input data before training
            logger.info("=" * 70)
            logger.info("NORMALIZING DATA FOR JOINT TRAINING")
            logger.info("=" * 70)
            logger.info(f"Train data stats (BEFORE normalize): X_tr.shape={X_tr.shape}, min={X_tr.min():.4f}, max={X_tr.max():.4f}, mean={X_tr.mean():.4f}, std={X_tr.std():.4f}")
            
            if X_va is not None:
                logger.info(f"Val data stats (BEFORE normalize): X_va.shape={X_va.shape}, min={X_va.min():.4f}, max={X_va.max():.4f}, mean={X_va.mean():.4f}, std={X_va.std():.4f}")
                # Check for scale mismatch
                scale_ratio = X_va.std() / (X_tr.std() + 1e-10)
                if scale_ratio > 2.0 or scale_ratio < 0.5:
                    logger.warning(f"⚠️  Scale mismatch detected! Val std / Train std = {scale_ratio:.4f}")
                    logger.warning(f"   Normalizing data to fix this issue...")
            
            # Normalize input data using StandardScaler (fit on train, transform both train and val)
            logger.info("Normalizing input data with StandardScaler...")
            X_tr_scaled = self.scaler.fit_transform(X_tr)
            logger.info(f"Train data stats (AFTER normalize): min={X_tr_scaled.min():.4f}, max={X_tr_scaled.max():.4f}, mean={X_tr_scaled.mean():.4f}, std={X_tr_scaled.std():.4f}")
            
            if X_va is not None:
                X_va_scaled = self.scaler.transform(X_va)
                logger.info(f"Val data stats (AFTER normalize): min={X_va_scaled.min():.4f}, max={X_va_scaled.max():.4f}, mean={X_va_scaled.mean():.4f}, std={X_va_scaled.std():.4f}")
                validation_data = (X_va_scaled, [X_va_scaled, y_va])
            else:
                validation_data = None
            
            logger.info("=" * 70)
            
            # For joint training, we need custom loss that combines reconstruction and classification
            # Simplified: use separate losses with weights
            # Model expects single input X, outputs [reconstruction, classification]
            # Custom training (simplified - would need custom training loop for proper joint loss)
            # For now, use multi-output with weighted losses
            self.history = self.model.fit(
                X_tr_scaled, [X_tr_scaled, y_tr],  # Input: normalized X_tr, Targets: [normalized_reconstruction_target, classification_target]
                validation_data=validation_data,
                batch_size=self.params.get('batch_size', 256),
                epochs=self.params.get('epochs', 80),
                callbacks=cb_list,
                verbose=1
            )
        else:
            # Two-stage training
            # Stage 1: Train VAE
            logger.info("Stage 1: Training VAE...")
            
            # Hard assertions for runtime guards
            assert X_tr.ndim == 2, f"VAE expects 2D numeric matrix, got {X_tr.ndim}D"
            assert X_tr.dtype in [np.float32, np.float64], f"VAE expects float dtype, got {X_tr.dtype}"
            
            self.vae.fit(X_tr, y_tr, X_va, y_va)
            
            # Stage 2: Get latent representations and train TabNet
            logger.info("Stage 2: Training TabNet on latent representations...")
            
            # Log VAE parameters that were used to generate latent representations
            vae_params_used = {
                'latent_dim': self.params.get('latent_dim', 16),
                'width': self.params.get('width', 256),
                'depth': self.params.get('depth', 3),
                'beta': self.params.get('beta', 1.0),
                'dropout': self.params.get('dropout', 0.1),
                'lr': self.params.get('lr', 1e-3),
                'batch_size': self.params.get('batch_size', 128),
                'epochs': self.params.get('epochs', 50)
            }
            logger.info("=" * 70)
            logger.info("TWO-STAGE TRAINING MODE - VAE PARAMETERS PASSED TO TABNET")
            logger.info("=" * 70)
            logger.info(f"VAE Parameters used to generate latent representations:")
            for key, value in vae_params_used.items():
                logger.info(f"  {key}: {value}")
            logger.info(f"Latent dimension (TabNet input_dim): {vae_params_used['latent_dim']}")
            logger.info("=" * 70)
            
            logger.info(f"Encoding X_tr: shape {X_tr.shape} -> z_tr")
            z_tr = self.vae.get_latent(X_tr)
            # Replace NaN/inf with 0 to prevent TabNet training issues
            z_tr = np.nan_to_num(z_tr, nan=0.0, posinf=1e6, neginf=-1e6)
            assert z_tr.shape[0] == X_tr.shape[0], f"Latent rows must match: z_tr.shape[0]={z_tr.shape[0]} != X_tr.shape[0]={X_tr.shape[0]}"
            logger.info(f"Encoded z_tr: shape {z_tr.shape}, NaN count: {np.isnan(z_tr).sum()}, Inf count: {np.isinf(z_tr).sum()}")
            
            z_va = None
            if X_va is not None:
                logger.info(f"Encoding X_va: shape {X_va.shape} -> z_va")
                z_va = self.vae.get_latent(X_va)
                # Replace NaN/inf with 0
                z_va = np.nan_to_num(z_va, nan=0.0, posinf=1e6, neginf=-1e6)
                assert z_va.shape[0] == X_va.shape[0], f"Latent rows must match: z_va.shape[0]={z_va.shape[0]} != X_va.shape[0]={X_va.shape[0]}"
                logger.info(f"Encoded z_va: shape {z_va.shape}, NaN count: {np.isnan(z_va).sum()}, Inf count: {np.isinf(z_va).sum()}")
            
            # Build TabNet with latent_dim as input
            logger.info(f"Building TabNet with input_dim={vae_params_used['latent_dim']} (from VAE latent_dim)")
            self.tabnet = TabNetTrainer(
                input_dim=vae_params_used['latent_dim'],
                params=self.tabnet_params,
                seed=self.seed,
                monitor=self.monitor,
                patience=self.patience,
                min_delta=self.min_delta
            )
            
            self.tabnet.fit(z_tr, y_tr, z_va, y_va, tensorboard_dir)
        
        self.is_fitted = True
        logger.info("Joint VAE-TabNet training completed")
    
    def get_latent(self, X: np.ndarray) -> np.ndarray:
        """
        Get latent representation z from input X.
        
        Args:
            X: Input features (will be normalized if joint training)
        
        Returns:
            Latent representation z (z_mean for deterministic output)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting latent representation")
        
        if self.joint_training:
            # Normalize input data using fitted scaler
            X_scaled = self.scaler.transform(X)
            # Extract latent z from joint model using encode() method
            z_mean, _, _ = self.model.encode(X_scaled, training=False)
            z_mean_np = z_mean.numpy() if hasattr(z_mean, 'numpy') else z_mean
            # Replace NaN/inf to prevent downstream issues
            z_mean_np = np.nan_to_num(z_mean_np, nan=0.0, posinf=1e6, neginf=-1e6)
            return z_mean_np
        else:
            # Two-stage: use VAE's get_latent (already handles NaN)
            return self.vae.get_latent(X)
    
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
        
        if self.joint_training:
            # Normalize input data using fitted scaler
            X_scaled = self.scaler.transform(X)
            _, predictions = self.model.predict(X_scaled, verbose=0)
            # predictions is already flattened from sigmoid output
            return predictions.flatten() if len(predictions.shape) > 1 else predictions
        else:
            z = self.vae.get_latent(X)
            return self.tabnet.predict_proba(z)
    
    def save(self, path: Path) -> None:
        """
        Save model to disk.
        
        Args:
            path: Directory path to save the model
        """
        path.mkdir(parents=True, exist_ok=True)
        
        if self.joint_training:
            # Save joint model
            save_keras_model(self.model, str(path / 'model.h5'))
            # Extract encoder and decoder from joint model
            if hasattr(self.model, 'encoder'):
                save_keras_model(self.model.encoder, str(path / 'encoder.h5'))
            if hasattr(self.model, 'decoder'):
                save_keras_model(self.model.decoder, str(path / 'decoder.h5'))
            # Classifier is part of the joint model
            if hasattr(self.model, 'classifier_layers'):
                # Save classifier separately if accessible
                pass  # Classifier is embedded in model.h5
            # Save scaler (CRITICAL: needed for inference)
            if hasattr(self, 'scaler'):
                save_joblib(self.scaler, str(path / 'scaler.pkl'))
                logger.info(f"Saved scaler to {path / 'scaler.pkl'}")
        else:
            # Two-stage: Save VAE and TabNet separately
            # VAE saves encoder.h5, decoder.h5, vae.h5
            self.vae.save(path / 'vae')
            # Move encoder/decoder to root for consistency
            vae_dir = path / 'vae'
            if (vae_dir / 'encoder.h5').exists():
                import shutil
                shutil.copy(vae_dir / 'encoder.h5', path / 'encoder.h5')
            if (vae_dir / 'decoder.h5').exists():
                import shutil
                shutil.copy(vae_dir / 'decoder.h5', path / 'decoder.h5')
            
            # TabNet saves model.weights.h5 (new format) or model.h5 (old format)
            self.tabnet.save(path / 'tabnet')
            # Copy TabNet model to root as classifier.h5 for clarity
            tabnet_dir = path / 'tabnet'
            # Try new format first, then old format for backward compatibility
            if (tabnet_dir / 'model.weights.h5').exists():
                import shutil
                shutil.copy(tabnet_dir / 'model.weights.h5', path / 'classifier.weights.h5')
            elif (tabnet_dir / 'model.h5').exists():
                import shutil
                shutil.copy(tabnet_dir / 'model.h5', path / 'classifier.h5')
        
        # Save parameters
        params_path = path / 'params.json'
        params_to_save = self.params.copy()
        params_to_save['joint_training'] = self.joint_training
        save_json(params_to_save, str(params_path))
        
        logger.info(f"Joint VAE-TabNet saved to {path}")
