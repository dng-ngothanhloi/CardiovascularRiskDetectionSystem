"""
Utilities for extracting TabNet attention masks and summary statistics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


def _load_tf() -> Optional["object"]:
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover - tf may be absent in CI
        LOGGER.warning("TensorFlow import failed: %s", exc)
        return None
    return tf


def _load_tabnet_model(
    model_path: Path,
    input_dim: int,
) -> Optional["object"]:
    tf = _load_tf()
    if tf is None:
        return None

    from src.models.tabnet_model import AttentiveTransformer, FeatureTransformer, GLU, TabNetModel

    custom_objects = {
        "GLU": GLU,
        "FeatureTransformer": FeatureTransformer,
        "AttentiveTransformer": AttentiveTransformer,
        "TabNetModel": TabNetModel,
    }

    model_path = Path(model_path)
    if model_path.is_file():
        LOGGER.info("Loading TabNet model from %s", model_path)
        try:
            model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects, compile=False)
            return model
        except Exception as exc:
            LOGGER.error("Failed to load TabNet model %s: %s", model_path, exc)
            return None

    weights_path = model_path / "model.weights.h5"
    params_path = model_path / "params.json"
    if not weights_path.exists() or not params_path.exists():
        LOGGER.warning("Missing weights or params for TabNet model at %s", model_path)
        return None

    with params_path.open("r", encoding="utf-8") as file:
        params = json.load(file)

    model = TabNetModel(
        input_dim=input_dim,
        n_d=params.get("n_d", 16),
        n_a=params.get("n_a", 16),
        n_steps=params.get("n_steps", 5),
        gamma=params.get("gamma", 1.5),
        lambda_sparse=params.get("lambda_sparse", 1e-4),
        dropout=params.get("dropout", 0.0),
    )
    dummy = np.zeros((1, input_dim), dtype=np.float32)
    model(dummy, training=False)
    try:
        model.load_weights(str(weights_path))
    except Exception as exc:
        LOGGER.error("Failed to load weights for TabNet model at %s: %s", model_path, exc)
        return None
    LOGGER.info("Loaded TabNet weights from %s", weights_path)
    return model


def extract_attention_masks(model_path: Path, X_val: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract per-sample attention masks from a trained TabNet model.
    """
    X_val = np.asarray(X_val, dtype=np.float32)
    if X_val.size == 0:
        LOGGER.warning("Validation data empty; cannot extract attention masks.")
        return None

    model = _load_tabnet_model(Path(model_path), input_dim=X_val.shape[1])
    if model is None:
        return None

    batch_size = max(len(X_val), 1)
    tf = _load_tf()
    if tf is None:
        return None

    prev_flag = tf.config.functions_run_eagerly()
    tf.config.run_functions_eagerly(True)
    try:
        model.predict(X_val, batch_size=batch_size, verbose=0)
    except Exception as exc:
        LOGGER.error("TabNet prediction failed for %s: %s", model_path, exc)
        return None
    finally:
        tf.config.run_functions_eagerly(prev_flag)

    masks = getattr(model, "attention_masks", None)
    if not masks:
        LOGGER.warning("TabNet model at %s did not expose attention masks.", model_path)
        return None

    mask_arrays = []
    for mask in masks:
        if hasattr(mask, "numpy"):
            try:
                mask_arrays.append(mask.numpy())
                continue
            except Exception:
                pass
        try:
            mask_arrays.append(tf.convert_to_tensor(mask).numpy())
        except Exception as exc:
            LOGGER.error("Failed to convert attention mask to numpy: %s", exc)
            return None

    try:
        stacked = np.stack(mask_arrays, axis=0)  # [steps, batch, features]
    except ValueError as exc:
        LOGGER.error("Attention masks shapes incompatible: %s", exc)
        return None

    masks_per_sample = np.transpose(stacked, (1, 2, 0))  # [batch, features, steps]
    LOGGER.info(
        "Extracted attention masks: model=%s, shape=%s",
        model_path,
        masks_per_sample.shape,
    )
    return masks_per_sample


def summarise_masks(masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Summarise per-sample masks into averages over steps and samples.
    """
    if masks.ndim == 2:
        per_sample = masks
    elif masks.ndim == 3:
        per_sample = masks.mean(axis=2)
    else:
        raise ValueError(f"Unexpected mask shape: {masks.shape}")

    feature_mean = per_sample.mean(axis=0)
    return per_sample, feature_mean


