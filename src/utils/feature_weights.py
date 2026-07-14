"""
Utilities for computing feature weights from TabNet attention masks.

Supports:
- Per-patient feature weights
- Per-group feature weights (e.g., by gender, age group, risk level)
- Global feature importance
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger('HeartDisease_RiskDiscovery')


def compute_per_patient_weights(
    attention_masks: List[np.ndarray]
) -> np.ndarray:
    """
    Compute per-patient feature weights from TabNet attention masks.
    
    Formula: m^(i) = Σ_t M_t^(i)
    Where M_t^(i) is the attention mask at step t for patient i.
    
    Args:
        attention_masks: List of attention masks, each shape [batch, n_features]
                        from different TabNet steps
        
    Returns:
        Per-patient feature weights, shape [n_samples, n_features]
    """
    if not attention_masks:
        raise ValueError("No attention masks provided")
    
    # Stack masks and sum across steps
    # Each mask is [batch, n_features]
    # Result: [n_steps, batch, n_features] -> sum -> [batch, n_features]
    stacked = np.stack(attention_masks, axis=0)  # [n_steps, batch, n_features]
    per_patient_weights = np.sum(stacked, axis=0)  # [batch, n_features]
    
    logger.info(f"Computed per-patient weights: shape {per_patient_weights.shape}")
    return per_patient_weights


def compute_global_importance(
    attention_masks: List[np.ndarray]
) -> np.ndarray:
    """
    Compute global feature importance from attention masks.
    
    Formula: FI_j = E_i[m_j^(i)]
    Where m_j^(i) is the weight of feature j for patient i.
    
    Args:
        attention_masks: List of attention masks from TabNet
        
    Returns:
        Global feature importance, shape [n_features]
    """
    per_patient = compute_per_patient_weights(attention_masks)
    global_importance = np.mean(per_patient, axis=0)  # [n_features]
    
    logger.info(f"Computed global importance: shape {global_importance.shape}")
    return global_importance


def compute_group_importance(
    attention_masks: List[np.ndarray],
    group_labels: np.ndarray
) -> Dict[int, np.ndarray]:
    """
    Compute feature importance per group.
    
    Args:
        attention_masks: List of attention masks from TabNet
        group_labels: Group labels for each sample, shape [n_samples]
        
    Returns:
        Dictionary mapping group_id to feature importance array [n_features]
    """
    per_patient = compute_per_patient_weights(attention_masks)
    
    if len(per_patient) != len(group_labels):
        raise ValueError(f"Mismatch: {len(per_patient)} samples vs {len(group_labels)} labels")
    
    group_importance = {}
    unique_groups = np.unique(group_labels)
    
    for group_id in unique_groups:
        mask = group_labels == group_id
        group_weights = per_patient[mask]
        group_importance[int(group_id)] = np.mean(group_weights, axis=0)
        logger.info(f"Group {group_id}: {np.sum(mask)} samples, importance shape {group_importance[int(group_id)].shape}")
    
    return group_importance


def load_attention_masks(mask_path: Path) -> Optional[List[np.ndarray]]:
    """
    Load attention masks from saved .npy file.
    
    Args:
        mask_path: Path to attention_masks.npy file
        
    Returns:
        List of attention mask arrays, or None if file doesn't exist
    """
    if not mask_path.exists():
        logger.warning(f"Attention masks file not found: {mask_path}")
        return None
    
    try:
        masks = np.load(mask_path, allow_pickle=True)
        if isinstance(masks, np.ndarray):
            # If saved as array, convert to list
            if masks.ndim == 3:
                # Shape: [n_steps, batch, n_features]
                masks = [masks[i] for i in range(masks.shape[0])]
            else:
                masks = [masks]
        
        logger.info(f"Loaded {len(masks)} attention masks from {mask_path}")
        return masks
    except Exception as e:
        logger.error(f"Error loading attention masks from {mask_path}: {e}")
        return None


def save_feature_weights(
    per_patient_weights: np.ndarray,
    global_importance: np.ndarray,
    group_importance: Optional[Dict[int, np.ndarray]],
    feature_names: Optional[List[str]],
    output_path: Path
) -> None:
    """
    Save feature weights to JSON and CSV files.
    
    Args:
        per_patient_weights: Per-patient weights, shape [n_samples, n_features]
        global_importance: Global importance, shape [n_features]
        group_importance: Optional group importance dict
        feature_names: Optional feature names
        output_path: Output directory
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save global importance
    if feature_names:
        global_df = pd.DataFrame({
            'feature': feature_names,
            'importance': global_importance
        })
    else:
        global_df = pd.DataFrame({
            'feature_idx': range(len(global_importance)),
            'importance': global_importance
        })
    
    global_df = global_df.sort_values('importance', ascending=False)
    global_df.to_csv(output_path / 'global_feature_importance.csv', index=False)
    logger.info(f"Saved global feature importance to {output_path / 'global_feature_importance.csv'}")
    
    # Save per-patient weights (as CSV)
    per_patient_df = pd.DataFrame(per_patient_weights)
    if feature_names:
        per_patient_df.columns = feature_names
    else:
        per_patient_df.columns = [f'feature_{i}' for i in range(per_patient_df.shape[1])]
    
    per_patient_df.to_csv(output_path / 'per_patient_feature_weights.csv', index=False)
    logger.info(f"Saved per-patient weights to {output_path / 'per_patient_feature_weights.csv'}")
    
    # Save group importance if provided
    if group_importance:
        group_data = []
        for group_id, importance in group_importance.items():
            if feature_names:
                group_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                })
            else:
                group_df = pd.DataFrame({
                    'feature_idx': range(len(importance)),
                    'importance': importance
                })
            group_df = group_df.sort_values('importance', ascending=False)
            group_df['group_id'] = group_id
            group_data.append(group_df)
        
        if group_data:
            group_df_all = pd.concat(group_data, ignore_index=True)
            group_df_all.to_csv(output_path / 'group_feature_importance.csv', index=False)
            logger.info(f"Saved group importance to {output_path / 'group_feature_importance.csv'}")
    
    # Save summary JSON
    summary = {
        'global_importance': global_importance.tolist(),
        'n_features': len(global_importance),
        'n_samples': len(per_patient_weights),
        'top_k_features': int(np.sum(global_importance > np.mean(global_importance)))
    }
    
    if feature_names:
        top_k = 10
        top_indices = np.argsort(global_importance)[-top_k:][::-1]
        summary['top_features'] = [
            {'feature': feature_names[i], 'importance': float(global_importance[i])}
            for i in top_indices
        ]
    
    if group_importance:
        summary['n_groups'] = len(group_importance)
        summary['groups'] = list(group_importance.keys())
    
    import json
    summary_path = output_path / 'feature_weights_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")

