"""
Test Prediction Verification Script

Extract 10-20 users from test set, apply predictions from trained models,
and compare with ground truth.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence, Union
import argparse
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import setup_logging, load_joblib, load_json
from tensorflow import keras
from .models.random_forest import RandomForestTrainer
from .models.tabnet_model import TabNetTrainer, TabNetModel

logger = logging.getLogger('HeartDisease_RiskDiscovery')


def load_test_data(fe_dir: Path, model_name: str, target_col: str = "cardio") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load test data for a specific model.
    
    Args:
        fe_dir: Feature engineering directory
        model_name: Model name ('rf' or 'tabnet')
        target_col: Target column name
        
    Returns:
        (X_test, y_test, feature_names)
    """
    if model_name == 'rf':
        test_csv = fe_dir / 'random_forest' / 'Test_data_rf.csv'
        features_json = fe_dir / 'random_forest' / 'selected_features_rf.json'
    elif model_name == 'tabnet':
        test_csv = fe_dir / 'tabnet' / 'Test_data_tabnet.csv'
        features_json = fe_dir / 'tabnet' / 'selected_features_tabnet.json'
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Load selected features
    features_data = load_json(str(features_json))
    if 'kept_features' in features_data:
        selected_features = features_data['kept_features']
    else:
        selected_features = features_data.get('selected_features', [])
    
    # Load test data
    test_df = pd.read_csv(test_csv)
    if target_col not in test_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {test_csv}")
    
    # Extract features and target
    X_test = test_df[selected_features].values
    y_test = test_df[target_col].values
    
    logger.info(f"Loaded test data for {model_name}: {X_test.shape}, features: {len(selected_features)}")
    return X_test, y_test, selected_features


def load_trained_model(
    model_dir: Path,
    model_name: str,
    fold: int = 0,
    expected_input_dim: Optional[int] = None
) -> Tuple[any, Dict]:
    """
    Load trained model from artifacts.
    
    Args:
        model_dir: Model directory (e.g., artifacts/Model/rf)
        model_name: Model name ('rf' or 'tabnet')
        fold: Fold number (default: 0)
        
    Returns:
        (model, params)
    """
    fold_dir = model_dir / f'fold_{fold}'
    
    if model_name == 'rf':
        model_path = fold_dir / 'model.joblib'
        params_path = fold_dir / 'params.json'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = load_joblib(str(model_path))
        params = load_json(str(params_path)) if params_path.exists() else {}
        
        # Create trainer wrapper
        trainer = RandomForestTrainer(params, seed=42)
        trainer.model = model
        trainer.is_fitted = True
        
        return trainer, params
    
    elif model_name == 'tabnet':
        # Try new format first (weights only with .weights.h5 extension)
        model_path = fold_dir / 'model.weights.h5'
        # Fallback to old format for backward compatibility
        model_path_old = fold_dir / 'model.h5'
        params_path = fold_dir / 'params.json'
        
        # Check which format exists
        if not model_path.exists() and not model_path_old.exists():
            raise FileNotFoundError(f"Model not found: {model_path} or {model_path_old}")
        
        # Use new format if exists, otherwise old format
        if model_path.exists():
            use_weights_format = True
        else:
            model_path = model_path_old
            use_weights_format = False
        
        # Load params first
        params = load_json(str(params_path)) if params_path.exists() else {}
        
        # Get input_dim from params; fall back to expected_input_dim or default
        if 'input_dim' in params:
            input_dim = params['input_dim']
        elif expected_input_dim is not None:
            logger.warning(
                "TabNet params.json missing 'input_dim'; using provided expected_input_dim=%d.",
                expected_input_dim
            )
            input_dim = expected_input_dim
        else:
            logger.warning(
                "TabNet params.json missing 'input_dim' and no expected_input_dim provided; "
                "defaulting to 16. This may cause shape mismatches."
            )
            input_dim = 16
        
        # Import all custom layers for custom_objects
        from .models.tabnet_model import TabNetBlock, FeatureTransformer, AttentiveTransformer, GLU
        custom_objects = {
            'TabNetModel': TabNetModel,
            'TabNetBlock': TabNetBlock,
            'FeatureTransformer': FeatureTransformer,
            'AttentiveTransformer': AttentiveTransformer,
            'GLU': GLU
        }
        
        # Solution 3: Load weights only (Best long-term approach)
        # Create trainer and build model with dummy input, then load weights
        trainer = TabNetTrainer(input_dim=input_dim, params=params, seed=42)
        
        # Build model with dummy input to initialize layers
        import numpy as np
        dummy_input = np.zeros((1, input_dim), dtype=np.float32)
        _ = trainer.model(dummy_input, training=False)  # Build model
        
        # Load weights (model was saved with save_weights, not save)
        if use_weights_format:
            # New format: weights only with .weights.h5 extension
            try:
                trainer.model.load_weights(str(model_path))
                trainer.is_fitted = True
                logger.info("TabNet model loaded successfully (weights only - new format)")
            except Exception as e:
                logger.error(f"Failed to load TabNet weights: {e}")
                raise
        else:
            # Old format: full model save (backward compatibility)
            try:
                saved_model = keras.models.load_model(str(model_path), custom_objects=custom_objects, compile=False)
                trainer.model.set_weights(saved_model.get_weights())
                trainer.is_fitted = True
                logger.info("TabNet model loaded successfully (old format - full model)")
            except Exception as e:
                logger.error(f"Failed to load TabNet model: {e}")
                raise
        
        return trainer, params
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def predict_test_samples(
    model_trainer: any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_samples: Union[int, Sequence[int]] = 20,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Predict on test samples and create comparison DataFrame.
    
    Args:
        model_trainer: Trained model trainer
        X_test: Test features
        y_test: Test labels
        n_samples: Number of samples to extract
        random_seed: Random seed
        
    Returns:
        DataFrame with predictions and ground truth
    """
    # Predict on all test data
    predictions = model_trainer.predict_proba(X_test)
    
    # Select random samples
    rng = np.random.RandomState(random_seed)
    n_total = len(X_test)
    if isinstance(n_samples, int):
        desired = max(1, min(n_samples, n_total))
        indices = np.asarray(rng.choice(n_total, size=desired, replace=False), dtype=int)
    else:
        # Preserve order provided, drop duplicates
        raw_indices = []
        for val in n_samples:
            try:
                idx = int(val)
            except (TypeError, ValueError):
                continue
            if idx not in raw_indices:
                raw_indices.append(idx)
        valid_indices = [idx for idx in raw_indices if 0 <= idx < n_total]
        if not valid_indices:
            logger.warning(
                "No valid sample IDs supplied; falling back to random selection of 20 samples."
            )
            desired = min(20, n_total)
            indices = np.asarray(rng.choice(n_total, size=desired, replace=False), dtype=int)
        else:
            if len(valid_indices) < len(raw_indices):
                logger.warning(
                    "Some sample IDs were invalid or out of range and have been omitted."
                )
            indices = np.asarray(valid_indices, dtype=int)
    
    # Create comparison DataFrame
    results = pd.DataFrame({
        'sample_id': indices,
        'y_true': y_test[indices],
        'y_pred_prob': predictions[indices],
        'y_pred_binary': (predictions[indices] >= 0.5).astype(int)
    })
    
    # Add correctness
    results['correct'] = (results['y_pred_binary'] == results['y_true']).astype(int)
    
    # Sort by prediction probability (descending)
    results = results.sort_values('y_pred_prob', ascending=False).reset_index(drop=True)
    
    logger.info(f"Selected {len(indices)} test samples for verification")
    return results


def plot_probability_spectrum(
    results_rf: pd.DataFrame,
    results_tabnet: pd.DataFrame,
    output_dir: Path,
    tabnet_available: bool = True
) -> Path:
    """
    Create probability spectrum plot showing TP, TN, FP, FN with threshold and ideal line.
    
    Args:
        results_rf: RF prediction results
        results_tabnet: TabNet prediction results
        output_dir: Output directory for plots
        tabnet_available: Whether TabNet is available
        
    Returns:
        Path to saved plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    models = [('RandomForest', results_rf, axes[0])]
    if tabnet_available and len(results_tabnet) > 0:
        models.append(('TabNet', results_tabnet, axes[1]))
    else:
        axes[1].text(0.5, 0.5, 'TabNet model không thể load', 
                     ha='center', va='center', fontsize=14)
        axes[1].set_title('TabNet - Probability Spectrum')
    
    for model_name, results, ax in models:
        # Sort by probability for better visualization
        results_sorted = results.sort_values('y_pred_prob', ascending=False).reset_index(drop=True)
        
        # Classify samples
        tp_mask = (results_sorted['y_pred_binary'] == 1) & (results_sorted['y_true'] == 1)
        tn_mask = (results_sorted['y_pred_binary'] == 0) & (results_sorted['y_true'] == 0)
        fp_mask = (results_sorted['y_pred_binary'] == 1) & (results_sorted['y_true'] == 0)
        fn_mask = (results_sorted['y_pred_binary'] == 0) & (results_sorted['y_true'] == 1)
        
        # Plot TP (green)
        if tp_mask.sum() > 0:
            tp_indices = results_sorted[tp_mask].index.values
            tp_probs = results_sorted.loc[tp_mask, 'y_pred_prob'].values
            ax.scatter(tp_indices, tp_probs, c='#2ecc71', s=100, alpha=0.7, 
                      label=f'TP ({tp_mask.sum()})', marker='o', edgecolors='darkgreen', linewidths=1.5)
        
        # Plot TN (blue)
        if tn_mask.sum() > 0:
            tn_indices = results_sorted[tn_mask].index.values
            tn_probs = results_sorted.loc[tn_mask, 'y_pred_prob'].values
            ax.scatter(tn_indices, tn_probs, c='#3498db', s=100, alpha=0.7, 
                      label=f'TN ({tn_mask.sum()})', marker='s', edgecolors='darkblue', linewidths=1.5)
        
        # Plot FP (orange)
        if fp_mask.sum() > 0:
            fp_indices = results_sorted[fp_mask].index.values
            fp_probs = results_sorted.loc[fp_mask, 'y_pred_prob'].values
            ax.scatter(fp_indices, fp_probs, c='#f39c12', s=120, alpha=0.8, 
                      label=f'FP ({fp_mask.sum()})', marker='^', edgecolors='darkorange', linewidths=2)
        
        # Plot FN (red - serious error)
        if fn_mask.sum() > 0:
            fn_indices = results_sorted[fn_mask].index.values
            fn_probs = results_sorted.loc[fn_mask, 'y_pred_prob'].values
            ax.scatter(fn_indices, fn_probs, c='#e74c3c', s=150, alpha=0.9, 
                      label=f'FN ({fn_mask.sum()})', marker='X', edgecolors='darkred', linewidths=2.5)
        
        # Threshold line at 0.5
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Threshold = 0.5', alpha=0.7)
        
        # Ideal diagonal line (perfect calibration: prob = y_true)
        # For TP: should be at 1.0, for TN: should be at 0.0
        # We'll draw a reference line from (0, 0) to (n_samples, 1) for ideal case
        n_samples = len(results_sorted)
        ax.plot([0, n_samples], [0, 1], 'k:', linewidth=1.5, alpha=0.5, label='Ideal (perfect calibration)')
        
        ax.set_xlabel('Sample Index (sorted by probability)', fontsize=12)
        ax.set_ylabel('Predicted Probability', fontsize=12)
        ax.set_title(f'{model_name} - Correct/Incorrect Predictions – Probability Spectrum', fontsize=14, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.5, n_samples - 0.5)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plot_path = output_dir / 'probability_spectrum.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Probability spectrum plot saved to {plot_path}")
    return plot_path


def create_detailed_report(
    results_rf: pd.DataFrame,
    results_tabnet: pd.DataFrame,
    feature_names_rf: List[str],
    feature_names_tabnet: List[str],
    X_test: np.ndarray,
    X_test_tabnet: np.ndarray,
    indices_rf: np.ndarray,
    indices_tabnet: np.ndarray,
    output_path: Path,
    tabnet_available: bool = True,
    test_df_rf: Optional[pd.DataFrame] = None,
    test_df_tabnet: Optional[pd.DataFrame] = None
) -> None:
    """
    Create detailed markdown report with sample predictions.
    
    Args:
        results_rf: RF prediction results
        results_tabnet: TabNet prediction results
        feature_names_rf: RF feature names
        feature_names_tabnet: TabNet feature names
        X_test: RF test features
        X_test_tabnet: TabNet test features
        indices_rf: RF sample indices
        indices_tabnet: TabNet sample indices
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Test Prediction Verification Report\n\n")
        f.write("**Ngày tạo:** 2025-11-07\n\n")
        f.write("Báo cáo này so sánh dự báo của models với ground truth trên tập test.\n\n")
        f.write("---\n\n")
        
        # Summary statistics
        f.write("## 1. Summary Statistics\n\n")
        f.write("### RandomForest\n\n")
        f.write(f"- **Total samples:** {len(results_rf)}\n")
        f.write(f"- **Correct predictions:** {results_rf['correct'].sum()} ({results_rf['correct'].mean()*100:.1f}%)\n")
        f.write(f"- **Accuracy:** {accuracy_score(results_rf['y_true'], results_rf['y_pred_binary']):.4f}\n")
        f.write(f"- **Precision:** {precision_score(results_rf['y_true'], results_rf['y_pred_binary'], zero_division=0):.4f}\n")
        f.write(f"- **Recall:** {recall_score(results_rf['y_true'], results_rf['y_pred_binary'], zero_division=0):.4f}\n")
        f.write(f"- **F1:** {f1_score(results_rf['y_true'], results_rf['y_pred_binary'], zero_division=0):.4f}\n\n")
        
        if tabnet_available and len(results_tabnet) > 0:
            f.write("### TabNet\n\n")
            f.write(f"- **Total samples:** {len(results_tabnet)}\n")
            f.write(f"- **Correct predictions:** {results_tabnet['correct'].sum()} ({results_tabnet['correct'].mean()*100:.1f}%)\n")
            f.write(f"- **Accuracy:** {accuracy_score(results_tabnet['y_true'], results_tabnet['y_pred_binary']):.4f}\n")
            f.write(f"- **Precision:** {precision_score(results_tabnet['y_true'], results_tabnet['y_pred_binary'], zero_division=0):.4f}\n")
            f.write(f"- **Recall:** {recall_score(results_tabnet['y_true'], results_tabnet['y_pred_binary'], zero_division=0):.4f}\n")
            f.write(f"- **F1:** {f1_score(results_tabnet['y_true'], results_tabnet['y_pred_binary'], zero_division=0):.4f}\n\n")
        else:
            f.write("### TabNet\n\n")
            f.write("- [WARN] **TabNet model không thể load** (serialization issue)\n\n")
        
        # Detailed predictions table
        f.write("## 2. Detailed Predictions - RandomForest\n\n")
        f.write("| Sample ID | y_true | y_pred_prob | y_pred_binary | Correct |\n")
        f.write("|-----------|--------|-------------|---------------|----------|\n")
        for _, row in results_rf.iterrows():
            correct_mark = "[OK]" if row['correct'] else "[X]"
            f.write(f"| {int(row['sample_id'])} | {int(row['y_true'])} | {row['y_pred_prob']:.4f} | {int(row['y_pred_binary'])} | {correct_mark} |\n")
        f.write("\n")
        
        if tabnet_available and len(results_tabnet) > 0:
            f.write("## 3. Detailed Predictions - TabNet\n\n")
            f.write("| Sample ID | y_true | y_pred_prob | y_pred_binary | Correct |\n")
            f.write("|-----------|--------|-------------|---------------|----------|\n")
            for _, row in results_tabnet.iterrows():
                correct_mark = "[OK]" if row['correct'] else "[X]"
                f.write(f"| {int(row['sample_id'])} | {int(row['y_true'])} | {row['y_pred_prob']:.4f} | {int(row['y_pred_binary'])} | {correct_mark} |\n")
            f.write("\n")
        else:
            f.write("## 3. Detailed Predictions - TabNet\n\n")
            f.write("[WARN] TabNet model không thể load.\n\n")
        
        # Feature values for top samples
        f.write("## 4. Feature Values (Top 10 Samples by Prediction Probability)\n\n")
        f.write("### RandomForest\n\n")
        top_rf = results_rf.head(10)
        feat_cols_rf = feature_names_rf[: min(10, len(feature_names_rf))]
        header_rf = ["Sample ID"] + feat_cols_rf + ["y_true", "y_pred_prob"]
        f.write("| " + " | ".join(header_rf) + " |\n")
        f.write("|" + "|".join(["---"] * len(header_rf)) + "|\n")
        for _, row in top_rf.iterrows():
            idx = int(row["sample_id"])
            feat_vals = [f"{X_test[idx, i]:.2f}" for i in range(len(feat_cols_rf))]
            row_values = [str(idx)] + feat_vals + [str(int(row["y_true"])), f"{row['y_pred_prob']:.4f}"]
            f.write("| " + " | ".join(row_values) + " |\n")
        f.write("\n")
        
        if tabnet_available and len(results_tabnet) > 0:
            f.write("### TabNet\n\n")
            top_tabnet = results_tabnet.head(10)
            feat_cols_tab = feature_names_tabnet[: min(10, len(feature_names_tabnet))]
            header_tab = ["Sample ID"] + feat_cols_tab + ["y_true", "y_pred_prob"]
            f.write("| " + " | ".join(header_tab) + " |\n")
            f.write("|" + "|".join(["---"] * len(header_tab)) + "|\n")
            for _, row in top_tabnet.iterrows():
                idx = int(row["sample_id"])
                feat_vals = [f"{X_test_tabnet[idx, i]:.2f}" for i in range(len(feat_cols_tab))]
                row_values = [str(idx)] + feat_vals + [str(int(row["y_true"])), f"{row['y_pred_prob']:.4f}"]
                f.write("| " + " | ".join(row_values) + " |\n")
            f.write("\n")
        
        # Analysis
        f.write("## 5. Analysis\n\n")
        f.write("### Correct Predictions\n\n")
        f.write(f"- **RF:** {results_rf['correct'].sum()}/{len(results_rf)} ({results_rf['correct'].mean()*100:.1f}%)\n")
        if tabnet_available and len(results_tabnet) > 0:
            f.write(f"- **TabNet:** {results_tabnet['correct'].sum()}/{len(results_tabnet)} ({results_tabnet['correct'].mean()*100:.1f}%)\n\n")
        else:
            f.write("- **TabNet:** N/A (model không thể load)\n\n")
        
        f.write("### Incorrect Predictions (False Positives & False Negatives)\n\n")
        rf_fp = ((results_rf['y_pred_binary'] == 1) & (results_rf['y_true'] == 0)).sum()
        rf_fn = ((results_rf['y_pred_binary'] == 0) & (results_rf['y_true'] == 1)).sum()
        
        f.write(f"- **RF False Positives:** {rf_fp}\n")
        f.write(f"- **RF False Negatives:** {rf_fn}\n")
        
        if tabnet_available and len(results_tabnet) > 0:
            tabnet_fp = ((results_tabnet['y_pred_binary'] == 1) & (results_tabnet['y_true'] == 0)).sum()
            tabnet_fn = ((results_tabnet['y_pred_binary'] == 0) & (results_tabnet['y_true'] == 1)).sum()
            f.write(f"- **TabNet False Positives:** {tabnet_fp}\n")
            f.write(f"- **TabNet False Negatives:** {tabnet_fn}\n\n")
        else:
            f.write("- **TabNet:** N/A (model không thể load)\n\n")
        
        # Create probability spectrum plot
        plot_dir = output_path.parent
        plot_path = plot_probability_spectrum(results_rf, results_tabnet, plot_dir, tabnet_available)
        f.write("## 6. Probability Spectrum Visualization\n\n")
        f.write(f"![Probability Spectrum]({plot_path.name})\n\n")
        f.write("*Hình 6.1: Biểu đồ phân phối probability cho các nhóm dự đoán. "
                "TP (True Positive), TN (True Negative), FP (False Positive), "
                "FN (False Negative - sai nghiêm trọng). Đường đứt nét đen: Threshold = 0.5. "
                "Đường chấm chấm: Ideal calibration line.*\n\n")
        
        # Separate tables for correct predictions
        f.write("## 7. Correct Predictions - Detailed Tables\n\n")
        
        # True Positives
        tp_mask_rf = (results_rf['y_pred_binary'] == 1) & (results_rf['y_true'] == 1)
        tp_rf = results_rf[tp_mask_rf].sort_values('y_pred_prob', ascending=False)
        
        f.write("### 7.1. True Positives (Correctly Predicted Disease) - RandomForest\n\n")
        f.write("| Sample ID | y_true | RF Prob | TabNet Prob | Key Features | Lifestyle |\n")
        f.write("|-----------|--------|---------|-------------|--------------|----------|\n")
        for _, row in tp_rf.iterrows():
            idx = int(row['sample_id'])
            tabnet_row = results_tabnet[results_tabnet['sample_id'] == idx] if tabnet_available else None
            tabnet_prob = f"{tabnet_row.iloc[0]['y_pred_prob']:.4f}" if tabnet_available and len(tabnet_row) > 0 else "N/A"
            
            # Get key features and lifestyle
            key_feats = "N/A"
            lifestyle = "N/A"
            if test_df_rf is not None:
                # Find row by sample_id
                sample_row = test_df_rf[test_df_rf['sample_id'] == idx]
                if len(sample_row) > 0:
                    sample_row = sample_row.iloc[0]
                    key_feats_list = []
                    if 'ap_hi' in test_df_rf.columns and pd.notna(sample_row.get('ap_hi')):
                        key_feats_list.append(f"ap_hi={int(sample_row['ap_hi'])}")
                    if 'ap_lo' in test_df_rf.columns and pd.notna(sample_row.get('ap_lo')):
                        key_feats_list.append(f"ap_lo={int(sample_row['ap_lo'])}")
                    if 'age_years' in test_df_rf.columns and pd.notna(sample_row.get('age_years')):
                        key_feats_list.append(f"age={int(sample_row['age_years'])}")
                    if 'bmi' in test_df_rf.columns and pd.notna(sample_row.get('bmi')):
                        key_feats_list.append(f"BMI={sample_row['bmi']:.1f}")
                    if 'cholesterol' in test_df_rf.columns and pd.notna(sample_row.get('cholesterol')):
                        key_feats_list.append(f"chol={int(sample_row['cholesterol'])}")
                    key_feats = ", ".join(key_feats_list) if key_feats_list else "N/A"
                    
                    # Lifestyle
                    lifestyle_list = []
                    if 'smoke' in test_df_rf.columns and pd.notna(sample_row.get('smoke')):
                        lifestyle_list.append(f"smoke={int(sample_row['smoke'])}")
                    if 'alco' in test_df_rf.columns and pd.notna(sample_row.get('alco')):
                        lifestyle_list.append(f"alco={int(sample_row['alco'])}")
                    if 'active' in test_df_rf.columns and pd.notna(sample_row.get('active')):
                        lifestyle_list.append(f"active={int(sample_row['active'])}")
                    if 'lifestyle_risk' in test_df_rf.columns and pd.notna(sample_row.get('lifestyle_risk')):
                        lifestyle_list.append(f"risk={sample_row['lifestyle_risk']:.1f}")
                    lifestyle = ", ".join(lifestyle_list) if lifestyle_list else "N/A"
            
            f.write(f"| {idx} | {int(row['y_true'])} | {row['y_pred_prob']:.4f} | {tabnet_prob} | {key_feats} | {lifestyle} |\n")
        f.write("\n")
        
        # True Negatives
        tn_mask_rf = (results_rf['y_pred_binary'] == 0) & (results_rf['y_true'] == 0)
        tn_rf = results_rf[tn_mask_rf].sort_values('y_pred_prob', ascending=False)
        
        f.write("### 7.2. True Negatives (Correctly Predicted No Disease) - RandomForest\n\n")
        f.write("| Sample ID | y_true | RF Prob | TabNet Prob | Key Features | Lifestyle |\n")
        f.write("|-----------|--------|---------|-------------|--------------|----------|\n")
        for _, row in tn_rf.iterrows():
            idx = int(row['sample_id'])
            tabnet_row = results_tabnet[results_tabnet['sample_id'] == idx] if tabnet_available else None
            tabnet_prob = f"{tabnet_row.iloc[0]['y_pred_prob']:.4f}" if tabnet_available and len(tabnet_row) > 0 else "N/A"
            
            # Get key features and lifestyle
            key_feats = "N/A"
            lifestyle = "N/A"
            if test_df_rf is not None:
                # Find row by sample_id
                sample_row = test_df_rf[test_df_rf['sample_id'] == idx]
                if len(sample_row) > 0:
                    sample_row = sample_row.iloc[0]
                    key_feats_list = []
                    if 'ap_hi' in test_df_rf.columns and pd.notna(sample_row.get('ap_hi')):
                        key_feats_list.append(f"ap_hi={int(sample_row['ap_hi'])}")
                    if 'ap_lo' in test_df_rf.columns and pd.notna(sample_row.get('ap_lo')):
                        key_feats_list.append(f"ap_lo={int(sample_row['ap_lo'])}")
                    if 'age_years' in test_df_rf.columns and pd.notna(sample_row.get('age_years')):
                        key_feats_list.append(f"age={int(sample_row['age_years'])}")
                    if 'bmi' in test_df_rf.columns and pd.notna(sample_row.get('bmi')):
                        key_feats_list.append(f"BMI={sample_row['bmi']:.1f}")
                    if 'cholesterol' in test_df_rf.columns and pd.notna(sample_row.get('cholesterol')):
                        key_feats_list.append(f"chol={int(sample_row['cholesterol'])}")
                    if 'bp_stage' in test_df_rf.columns and pd.notna(sample_row.get('bp_stage')):
                        key_feats_list.append(f"bp_stage={int(sample_row['bp_stage'])}")
                    key_feats = ", ".join(key_feats_list) if key_feats_list else "N/A"
                    
                    # Lifestyle
                    lifestyle_list = []
                    if 'smoke' in test_df_rf.columns and pd.notna(sample_row.get('smoke')):
                        lifestyle_list.append(f"smoke={int(sample_row['smoke'])}")
                    if 'alco' in test_df_rf.columns and pd.notna(sample_row.get('alco')):
                        lifestyle_list.append(f"alco={int(sample_row['alco'])}")
                    if 'active' in test_df_rf.columns and pd.notna(sample_row.get('active')):
                        lifestyle_list.append(f"active={int(sample_row['active'])}")
                    if 'lifestyle_risk' in test_df_rf.columns and pd.notna(sample_row.get('lifestyle_risk')):
                        lifestyle_list.append(f"risk={sample_row['lifestyle_risk']:.1f}")
                    lifestyle = ", ".join(lifestyle_list) if lifestyle_list else "N/A"
            
            f.write(f"| {idx} | {int(row['y_true'])} | {row['y_pred_prob']:.4f} | {tabnet_prob} | {key_feats} | {lifestyle} |\n")
        f.write("\n")
        
        # Separate tables for incorrect predictions
        f.write("## 8. Incorrect Predictions - Detailed Tables\n\n")
        
        # False Positives
        fp_mask = (results_rf['y_pred_binary'] == 1) & (results_rf['y_true'] == 0)
        fp_samples = results_rf[fp_mask].sort_values('y_pred_prob', ascending=False)
        
        f.write("### 8.1. False Positives (Predicted: 1, Actual: 0) - RandomForest\n\n")
        f.write("| Sample ID | y_true | RF Prob | TabNet Prob | Key Features | Lifestyle |\n")
        f.write("|-----------|--------|---------|-------------|--------------|----------|\n")
        for _, row in fp_samples.iterrows():
            idx = int(row['sample_id'])
            tabnet_row = results_tabnet[results_tabnet['sample_id'] == idx] if tabnet_available else None
            tabnet_prob = f"{tabnet_row.iloc[0]['y_pred_prob']:.4f}" if tabnet_available and len(tabnet_row) > 0 else "N/A"
            
            key_feats = "N/A"
            lifestyle = "N/A"
            if test_df_rf is not None:
                # Find row by sample_id
                sample_row = test_df_rf[test_df_rf['sample_id'] == idx]
                if len(sample_row) > 0:
                    sample_row = sample_row.iloc[0]
                    key_feats_list = []
                    if 'ap_hi' in test_df_rf.columns and pd.notna(sample_row.get('ap_hi')):
                        key_feats_list.append(f"ap_hi={int(sample_row['ap_hi'])}")
                    if 'ap_lo' in test_df_rf.columns and pd.notna(sample_row.get('ap_lo')):
                        key_feats_list.append(f"ap_lo={int(sample_row['ap_lo'])}")
                    if 'age_years' in test_df_rf.columns and pd.notna(sample_row.get('age_years')):
                        key_feats_list.append(f"age={int(sample_row['age_years'])}")
                    if 'bmi' in test_df_rf.columns and pd.notna(sample_row.get('bmi')):
                        key_feats_list.append(f"BMI={sample_row['bmi']:.1f}")
                    if 'cholesterol' in test_df_rf.columns and pd.notna(sample_row.get('cholesterol')):
                        key_feats_list.append(f"chol={int(sample_row['cholesterol'])}")
                    key_feats = ", ".join(key_feats_list) if key_feats_list else "N/A"
                    
                    lifestyle_list = []
                    if 'smoke' in test_df_rf.columns and pd.notna(sample_row.get('smoke')):
                        lifestyle_list.append(f"smoke={int(sample_row['smoke'])}")
                    if 'alco' in test_df_rf.columns and pd.notna(sample_row.get('alco')):
                        lifestyle_list.append(f"alco={int(sample_row['alco'])}")
                    if 'active' in test_df_rf.columns and pd.notna(sample_row.get('active')):
                        lifestyle_list.append(f"active={int(sample_row['active'])}")
                    if 'lifestyle_risk' in test_df_rf.columns and pd.notna(sample_row.get('lifestyle_risk')):
                        lifestyle_list.append(f"risk={sample_row['lifestyle_risk']:.1f}")
                    lifestyle = ", ".join(lifestyle_list) if lifestyle_list else "N/A"
            
            f.write(f"| {idx} | {int(row['y_true'])} | {row['y_pred_prob']:.4f} | {tabnet_prob} | {key_feats} | {lifestyle} |\n")
        f.write("\n")
        
        # False Negatives
        fn_mask = (results_rf['y_pred_binary'] == 0) & (results_rf['y_true'] == 1)
        fn_samples = results_rf[fn_mask].sort_values('y_pred_prob', ascending=False)
        
        f.write("### 8.2. False Negatives (Predicted: 0, Actual: 1) - RandomForest\n\n")
        f.write("| Sample ID | y_true | RF Prob | TabNet Prob | Key Features | Lifestyle |\n")
        f.write("|-----------|--------|---------|-------------|--------------|----------|\n")
        for _, row in fn_samples.iterrows():
            idx = int(row['sample_id'])
            tabnet_row = results_tabnet[results_tabnet['sample_id'] == idx] if tabnet_available else None
            tabnet_prob = f"{tabnet_row.iloc[0]['y_pred_prob']:.4f}" if tabnet_available and len(tabnet_row) > 0 else "N/A"
            
            key_feats = "N/A"
            lifestyle = "N/A"
            if test_df_rf is not None:
                # Find row by sample_id
                sample_row = test_df_rf[test_df_rf['sample_id'] == idx]
                if len(sample_row) > 0:
                    sample_row = sample_row.iloc[0]
                    key_feats_list = []
                    if 'ap_hi' in test_df_rf.columns and pd.notna(sample_row.get('ap_hi')):
                        key_feats_list.append(f"ap_hi={int(sample_row['ap_hi'])}")
                    if 'ap_lo' in test_df_rf.columns and pd.notna(sample_row.get('ap_lo')):
                        key_feats_list.append(f"ap_lo={int(sample_row['ap_lo'])}")
                    if 'age_years' in test_df_rf.columns and pd.notna(sample_row.get('age_years')):
                        key_feats_list.append(f"age={int(sample_row['age_years'])}")
                    if 'bmi' in test_df_rf.columns and pd.notna(sample_row.get('bmi')):
                        key_feats_list.append(f"BMI={sample_row['bmi']:.1f}")
                    if 'cholesterol' in test_df_rf.columns and pd.notna(sample_row.get('cholesterol')):
                        key_feats_list.append(f"chol={int(sample_row['cholesterol'])}")
                    key_feats = ", ".join(key_feats_list) if key_feats_list else "N/A"
                    
                    lifestyle_list = []
                    if 'smoke' in test_df_rf.columns and pd.notna(sample_row.get('smoke')):
                        lifestyle_list.append(f"smoke={int(sample_row['smoke'])}")
                    if 'alco' in test_df_rf.columns and pd.notna(sample_row.get('alco')):
                        lifestyle_list.append(f"alco={int(sample_row['alco'])}")
                    if 'active' in test_df_rf.columns and pd.notna(sample_row.get('active')):
                        lifestyle_list.append(f"active={int(sample_row['active'])}")
                    if 'lifestyle_risk' in test_df_rf.columns and pd.notna(sample_row.get('lifestyle_risk')):
                        lifestyle_list.append(f"risk={sample_row['lifestyle_risk']:.1f}")
                    lifestyle = ", ".join(lifestyle_list) if lifestyle_list else "N/A"
            
            f.write(f"| {idx} | {int(row['y_true'])} | {row['y_pred_prob']:.4f} | {tabnet_prob} | {key_feats} | {lifestyle} |\n")
        f.write("\n")
        
        f.write("---\n\n")
        f.write("**Báo cáo được tạo tự động từ test predictions.**\n")
    
    logger.info(f"Detailed report saved to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Test Prediction Verification Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model_root',
        type=str,
        default='artifacts/Model',
        help='Root directory containing model subfolders'
    )
    parser.add_argument(
        '--fe_dir',
        type=str,
        default='artifacts/fe',
        help='Feature engineering directory'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='reports',
        help='Output directory for report'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=20,
        help='Number of test samples to extract randomly (set 0 to use --test_prediction file)'
    )
    parser.add_argument(
        '--test_prediction',
        type=str,
        help='Path to text file containing sample_ids (one per line) when n_samples=0'
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=0,
        help='Fold number to use for model loading'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sample selection'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level='INFO')
    logger.info("=" * 70)
    logger.info("TEST PREDICTION VERIFICATION")
    logger.info("=" * 70)
    
    model_root = Path(args.model_root)
    fe_dir = Path(args.fe_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    logger.info("Loading test data...")
    X_test_rf, y_test_rf, feature_names_rf = load_test_data(fe_dir, 'rf')
    X_test_tabnet, y_test_tabnet, feature_names_tabnet = load_test_data(fe_dir, 'tabnet')
    
    # Load full test DataFrames for detailed analysis
    test_csv_rf = fe_dir / 'random_forest' / 'Test_data_rf.csv'
    test_csv_tabnet = fe_dir / 'tabnet' / 'Test_data_tabnet.csv'
    test_df_rf = pd.read_csv(test_csv_rf) if test_csv_rf.exists() else None
    test_df_tabnet = pd.read_csv(test_csv_tabnet) if test_csv_tabnet.exists() else None
    
    # Load trained models
    logger.info("Loading trained models...")
    rf_trainer, rf_params = load_trained_model(model_root / 'rf', 'rf', fold=args.fold)
    
    # Try to load TabNet (may fail due to serialization issues)
    try:
        tabnet_trainer, tabnet_params = load_trained_model(
            model_root / 'tabnet',
            'tabnet',
            fold=args.fold,
            expected_input_dim=X_test_tabnet.shape[1]
        )
        tabnet_available = True
    except Exception as e:
        logger.warning(f"Failed to load TabNet model: {e}")
        logger.warning("Continuing with RF only...")
        tabnet_trainer = None
        tabnet_params = {}
        tabnet_available = False
    
    if args.n_samples == 0:
        if args.test_prediction:
            custom_path = Path(args.test_prediction)
            if custom_path.exists():
                with custom_path.open("r", encoding="utf-8") as f:
                    sample_ids = [
                        int(line.strip())
                        for line in f.readlines()
                        if line.strip().isdigit()
                    ]
                if not sample_ids:
                    logger.warning("No valid sample IDs in %s; falling back to random selection.", custom_path)
                    sample_ids = None
                else:
                    logger.info("Using %d sample IDs from %s.", len(sample_ids), custom_path)
            else:
                logger.warning("test_prediction file not found at %s; falling back to random selection.", custom_path)
                sample_ids = None
        else:
            logger.warning(
                "n_samples set to 0 but --test_prediction not provided; falling back to random selection of 20 samples."
            )
            sample_ids = None
        if sample_ids is None:
            sample_ids = 20  # fallback to random 20 samples
        results_rf = predict_test_samples(
            rf_trainer,
            X_test_rf,
            y_test_rf,
            n_samples=sample_ids,
            random_seed=args.seed,
        )
        if tabnet_available:
            results_tabnet = predict_test_samples(
                tabnet_trainer,
                X_test_tabnet,
                y_test_tabnet,
                n_samples=sample_ids,
                random_seed=args.seed,
            )
        else:
            results_tabnet = pd.DataFrame(
                columns=['sample_id', 'y_true', 'y_pred_prob', 'y_pred_binary', 'correct']
            )
    else:
        # Predict on test samples
        logger.info("Making predictions on test samples...")
        results_rf = predict_test_samples(
            rf_trainer,
            X_test_rf,
            y_test_rf,
            n_samples=args.n_samples,
            random_seed=args.seed,
        )
        if tabnet_available:
            results_tabnet = predict_test_samples(
                tabnet_trainer,
                X_test_tabnet,
                y_test_tabnet,
                n_samples=args.n_samples,
                random_seed=args.seed,
            )
        else:
            # Create empty results for TabNet
            results_tabnet = pd.DataFrame(
                columns=['sample_id', 'y_true', 'y_pred_prob', 'y_pred_binary', 'correct']
            )
    
    interpretation_out = Path("artifacts/Interpretation/test_prediction_result")
    interpretation_out.mkdir(parents=True, exist_ok=True)

    rf_csv_path = interpretation_out / 'test_predictions_rf.csv'
    tabnet_csv_path = interpretation_out / 'test_predictions_tabnet.csv'
    results_rf.to_csv(rf_csv_path, index=False)
    results_tabnet.to_csv(tabnet_csv_path, index=False)
    logger.info("Saved prediction CSVs to %s", interpretation_out)

    if out_dir.resolve() != interpretation_out.resolve():
        (out_dir / 'test_predictions_rf.csv').parent.mkdir(parents=True, exist_ok=True)
        results_rf.to_csv(out_dir / 'test_predictions_rf.csv', index=False)
        results_tabnet.to_csv(out_dir / 'test_predictions_tabnet.csv', index=False)

    report_path = interpretation_out / 'Test_Prediction_Verification.md'
    create_detailed_report(
        results_rf,
        results_tabnet,
        feature_names_rf,
        feature_names_tabnet,
        X_test_rf,
        X_test_tabnet,
        results_rf['sample_id'].values.astype(int),
        results_tabnet['sample_id'].values.astype(int) if tabnet_available and len(results_tabnet) > 0 else np.array([]),
        report_path,
        tabnet_available=tabnet_available,
        test_df_rf=test_df_rf,
        test_df_tabnet=test_df_tabnet
    )
    
    logger.info("=" * 70)
    logger.info("VERIFICATION COMPLETE")
    logger.info("=" * 70)
    print(f"Report → {report_path}")


if __name__ == '__main__':
    main()

