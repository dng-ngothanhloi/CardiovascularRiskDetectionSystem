"""
Variational Autoencoder (VAE) for tabular data.
Learns latent representations for downstream classification.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from sklearn.preprocessing import StandardScaler

from ..utils import save_keras_model, save_json

# Use main logger to ensure logs are written to file
logger = logging.getLogger('HeartDisease_RiskDiscovery')


class Sampling(layers.Layer):
    """Sampling layer for VAE."""
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        # Clip z_log_var before exp to prevent numerical instability
        z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        # Clip z to prevent NaN/inf
        z = tf.clip_by_value(z, -1e6, 1e6)
        return z


class VAETrainer:
    """
    Variational Autoencoder trainer for tabular data.
    """
    
    def __init__(
        self,
        input_dim: int,
        params: Dict[str, Any],
        seed: int = 42
    ):
        """
        Initialize VAE trainer.
        
        Args:
            input_dim: Input feature dimension
            params: Model parameters (latent_dim, width, depth, beta, etc.)
            seed: Random seed
        """
        self.input_dim = input_dim
        self.params = params
        self.seed = seed
        
        # Set seed
        tf.random.set_seed(seed)
        
        # Build encoder
        latent_dim = params.get('latent_dim', 16)
        width = params.get('width', 256)
        depth = params.get('depth', 3)
        dropout = params.get('dropout', 0.1)
        beta = params.get('beta', 1.0)
        
        # Encoder
        encoder_input = layers.Input(shape=(input_dim,), name='encoder_input')
        x = encoder_input
        
        for i in range(depth):
            x = layers.Dense(width, activation='relu', name=f'encoder_dense_{i}')(x)
            if dropout > 0:
                x = layers.Dropout(dropout, name=f'encoder_dropout_{i}')(x)
        
        z_mean = layers.Dense(latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
        z = Sampling(name='z')([z_mean, z_log_var])
        
        self.encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name='encoder')
        
        # Decoder
        latent_input = layers.Input(shape=(latent_dim,), name='latent_input')
        x = latent_input
        
        for i in range(depth):
            x = layers.Dense(width, activation='relu', name=f'decoder_dense_{i}')(x)
            if dropout > 0:
                x = layers.Dropout(dropout, name=f'decoder_dropout_{i}')(x)
        
        decoder_output = layers.Dense(input_dim, name='decoder_output')(x)
        
        self.decoder = keras.Model(latent_input, decoder_output, name='decoder')
        
        # VAE (combined)
        vae_output = self.decoder(z)
        self.vae = keras.Model(encoder_input, vae_output, name='vae')
        
        # Loss function
        self.beta = beta
        self.input_dim = input_dim  # Store for loss normalization
        
        def vae_loss(x_true, x_pred):
            # Reconstruction loss
            # Calculate MSE manually: (x_true - x_pred)^2 per feature
            mse_per_feature = tf.square(x_true - x_pred)  # [batch, features]
            # Normalize by number of features to make loss scale-independent
            recon_loss = tf.reduce_mean(tf.reduce_sum(mse_per_feature, axis=1)) / tf.cast(self.input_dim, tf.float32)
            
            # KL divergence
            z_mean, z_log_var, _ = self.encoder(x_true)
            # Clip z_log_var to prevent numerical instability
            z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            ))
            
            # Total loss (reconstruction + beta * KL)
            total_loss = recon_loss + self.beta * kl_loss
            # Clip loss to prevent NaN/inf (increased from 1e6 to 1e8 to allow larger losses)
            total_loss = tf.clip_by_value(total_loss, -1e8, 1e8)
            
            return total_loss
        
        self.vae.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params.get('lr', 1e-3)),
            loss=vae_loss
        )
        
        self.is_fitted = False
        self.scaler = StandardScaler()  # For normalizing input data
        logger.info(f"VAE initialized: input_dim={input_dim}, latent_dim={latent_dim}, beta={beta}")
    
    def fit(
        self,
        X_tr: np.ndarray,
        y_tr: Optional[np.ndarray] = None,
        X_va: Optional[np.ndarray] = None,
        y_va: Optional[np.ndarray] = None
    ) -> None:
        """
        Train VAE model.
        
        Args:
            X_tr: Training features
            y_tr: Training labels (not used for VAE)
            X_va: Optional validation features
            y_va: Optional validation labels (not used for VAE)
        """
        logger.info(f"Training VAE on {len(X_tr)} samples")
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
            validation_data = (X_va_scaled, X_va_scaled)
        else:
            validation_data = None
        
        self.history = self.vae.fit(
            X_tr_scaled, X_tr_scaled,  # Autoencoder: reconstruct normalized input
            validation_data=validation_data,
            batch_size=self.params.get('batch_size', 128),
            epochs=self.params.get('epochs', 50),
            verbose=1
        )
        
        self.is_fitted = True
        logger.info("VAE training completed")
    
    def get_latent(self, X: np.ndarray) -> np.ndarray:
        """
        Get latent representation.
        
        Args:
            X: Input features (will be normalized using fitted scaler)
        
        Returns:
            Latent representation (z_mean)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get latent representation")
        # Normalize input data using fitted scaler
        X_scaled = self.scaler.transform(X)
        z_mean, _, _ = self.encoder.predict(X_scaled, verbose=0)
        # Replace NaN/inf to prevent downstream issues
        z_mean = np.nan_to_num(z_mean, nan=0.0, posinf=1e6, neginf=-1e6)
        return z_mean

    def save(self, path: Path) -> None:
        """
        Save model to disk.
        
        Args:
            path: Directory path to save the model
        """
        path.mkdir(parents=True, exist_ok=True)
        save_keras_model(self.encoder, str(path / 'encoder.h5'))
        save_keras_model(self.decoder, str(path / 'decoder.h5'))
        save_keras_model(self.vae, str(path / 'vae.h5'))
        
        # Save parameters
        params_path = path / 'params.json'
        save_json(self.params, str(params_path))
        
        # Save scaler
        from ..utils import save_joblib
        save_joblib(self.scaler, str(path / 'scaler.pkl'))
        
        logger.info(f"VAE saved to {path}")