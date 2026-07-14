"""
Compute Confusion Matrix on Test Set for Random Forest and TabNet
"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_joblib, load_json
from src.models.random_forest import RandomForestTrainer
from src.models.tabnet_model import TabNetTrainer

def find_best_threshold(y_true, y_proba):
    """Tìm threshold tối ưu cho F1 score"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1 = 0
    best_thresh = 0.5
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1

def plot_confusion_matrix(y_true, y_pred, model_name, threshold, save_path, normalize=False):
    """Vẽ confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title_suffix = ' (Normalized)'
    else:
        fmt = 'd'
        title_suffix = ''
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                xticklabels=['Negative (0)', 'Positive (1)'],
                yticklabels=['Negative (0)', 'Positive (1)'],
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix (TEST SET): {model_name.upper()}\nThreshold = {threshold:.3f}{title_suffix}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return cm

def compute_metrics_from_cm(cm):
    """Tính các metrics từ confusion matrix"""
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall (Sensitivity)': recall,
        'Specificity': specificity,
        'F1-Score': f1
    }

def load_test_data(fe_dir, model_name, target_col="cardio"):
    """Load test data"""
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
    
    # Remove sample_id if exists
    if 'sample_id' in test_df.columns:
        test_df = test_df.drop(columns=['sample_id'])
    
    # Use only selected features
    X_test = test_df[selected_features].values
    y_test = test_df[target_col].values.astype(int)
    
    return X_test, y_test

def main():
    print("=" * 80)
    print("TÍNH MA TRẬN NHẦM LẪN TRÊN TẬP TEST")
    print("=" * 80)
    
    # Đường dẫn
    fe_dir = Path("artifacts/fe")
    rf_model_dir = Path("artifacts/Model/rf")
    tabnet_model_dir = Path("artifacts/Model/tabnet")
    output_dir = Path("reports/confusion_matrices_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    n_folds = 5
    
    # Xử lý Random Forest
    if (fe_dir / 'random_forest' / 'Test_data_rf.csv').exists() and rf_model_dir.exists():
        print("\n" + "=" * 80)
        print("RANDOM FOREST - Test Set")
        print("=" * 80)
        
        X_test, y_test = load_test_data(fe_dir, 'rf')
        print(f"Test set size: {len(y_test)}")
        print(f"Test set distribution: {np.bincount(y_test)}")
        
        test_predictions = []
        for fold in range(n_folds):
            model_path = rf_model_dir / f"fold_{fold}" / "model.joblib"
            calibrator_path = rf_model_dir / f"fold_{fold}" / "calibrator.pkl"
            
            if model_path.exists():
                model = load_joblib(str(model_path))
                p_test = model.predict_proba(X_test)[:, 1]
                
                if calibrator_path.exists():
                    calibrator = load_joblib(str(calibrator_path))
                    p_test = calibrator.transform(p_test)
                
                test_predictions.append(p_test)
                print(f"  Fold {fold}: Loaded and predicted")
        
        if test_predictions:
            y_proba_test = np.mean(test_predictions, axis=0)
            print(f"\nAveraged predictions from {len(test_predictions)} folds")
            
            # Threshold 0.5
            y_pred_05 = (y_proba_test >= 0.5).astype(int)
            cm_05 = confusion_matrix(y_test, y_pred_05)
            metrics_05 = compute_metrics_from_cm(cm_05)
            
            # Best threshold
            best_thresh, best_f1 = find_best_threshold(y_test, y_proba_test)
            y_pred_best = (y_proba_test >= best_thresh).astype(int)
            cm_best = confusion_matrix(y_test, y_pred_best)
            metrics_best = compute_metrics_from_cm(cm_best)
            
            # Vẽ và lưu
            plot_confusion_matrix(y_test, y_pred_05, 'Random Forest', 0.5,
                                 output_dir / 'confusion_matrix_rf_test_threshold_0.5.png')
            plot_confusion_matrix(y_test, y_pred_05, 'Random Forest', 0.5,
                                 output_dir / 'confusion_matrix_rf_test_threshold_0.5_normalized.png', normalize=True)
            plot_confusion_matrix(y_test, y_pred_best, 'Random Forest', best_thresh,
                                 output_dir / 'confusion_matrix_rf_test_best_threshold.png')
            plot_confusion_matrix(y_test, y_pred_best, 'Random Forest', best_thresh,
                                 output_dir / 'confusion_matrix_rf_test_best_threshold_normalized.png', normalize=True)
            
            results['rf'] = {
                'threshold_0.5': {
                    'threshold': 0.5,
                    'confusion_matrix': cm_05.tolist(),
                    'metrics': metrics_05
                },
                'best_threshold': {
                    'threshold': float(best_thresh),
                    'confusion_matrix': cm_best.tolist(),
                    'metrics': metrics_best
                },
                'n_test_samples': int(len(y_test)),
                'n_folds_used': len(test_predictions)
            }
            
            print(f"\nThreshold = 0.5:")
            print(f"  Confusion Matrix:\n{cm_05}")
            print(f"  Accuracy: {metrics_05['Accuracy']:.4f}")
            print(f"  Precision: {metrics_05['Precision']:.4f}")
            print(f"  Recall: {metrics_05['Recall (Sensitivity)']:.4f}")
            print(f"  F1-Score: {metrics_05['F1-Score']:.4f}")
            
            print(f"\nBest Threshold = {best_thresh:.3f} (F1 = {best_f1:.4f}):")
            print(f"  Confusion Matrix:\n{cm_best}")
            print(f"  Accuracy: {metrics_best['Accuracy']:.4f}")
            print(f"  Precision: {metrics_best['Precision']:.4f}")
            print(f"  Recall: {metrics_best['Recall (Sensitivity)']:.4f}")
            print(f"  F1-Score: {metrics_best['F1-Score']:.4f}")
    
    # Xử lý TabNet
    if (fe_dir / 'tabnet' / 'Test_data_tabnet.csv').exists() and tabnet_model_dir.exists():
        print("\n" + "=" * 80)
        print("TABNET - Test Set")
        print("=" * 80)
        
        X_test, y_test = load_test_data(fe_dir, 'tabnet')
        X_test = X_test.astype(np.float32)
        print(f"Test set size: {len(y_test)}")
        print(f"Test set distribution: {np.bincount(y_test)}")
        
        test_predictions = []
        for fold in range(n_folds):
            model_path = tabnet_model_dir / f"fold_{fold}" / "model.weights.h5"
            params_path = tabnet_model_dir / f"fold_{fold}" / "params.json"
            calibrator_path = tabnet_model_dir / f"fold_{fold}" / "calibrator.pkl"
            
            if model_path.exists() and params_path.exists():
                params = load_json(str(params_path))
                input_dim = params.get('input_dim', X_test.shape[1])
                
                # Create trainer and build model
                trainer = TabNetTrainer(input_dim=input_dim, params=params, seed=42)
                
                # Build model with dummy input to initialize layers
                dummy_input = np.zeros((1, input_dim), dtype=np.float32)
                _ = trainer.model(dummy_input, training=False)  # Build model
                
                # Load weights
                trainer.model.load_weights(str(model_path))
                trainer.is_fitted = True
                
                p_test = trainer.predict_proba(X_test)
                
                if calibrator_path.exists():
                    calibrator = load_joblib(str(calibrator_path))
                    p_test = calibrator.transform(p_test)
                
                test_predictions.append(p_test)
                print(f"  Fold {fold}: Loaded and predicted")
        
        if test_predictions:
            y_proba_test = np.mean(test_predictions, axis=0)
            print(f"\nAveraged predictions from {len(test_predictions)} folds")
            
            # Threshold 0.5
            y_pred_05 = (y_proba_test >= 0.5).astype(int)
            cm_05 = confusion_matrix(y_test, y_pred_05)
            metrics_05 = compute_metrics_from_cm(cm_05)
            
            # Best threshold
            best_thresh, best_f1 = find_best_threshold(y_test, y_proba_test)
            y_pred_best = (y_proba_test >= best_thresh).astype(int)
            cm_best = confusion_matrix(y_test, y_pred_best)
            metrics_best = compute_metrics_from_cm(cm_best)
            
            # Vẽ và lưu
            plot_confusion_matrix(y_test, y_pred_05, 'TabNet', 0.5,
                                 output_dir / 'confusion_matrix_tabnet_test_threshold_0.5.png')
            plot_confusion_matrix(y_test, y_pred_05, 'TabNet', 0.5,
                                 output_dir / 'confusion_matrix_tabnet_test_threshold_0.5_normalized.png', normalize=True)
            plot_confusion_matrix(y_test, y_pred_best, 'TabNet', best_thresh,
                                 output_dir / 'confusion_matrix_tabnet_test_best_threshold.png')
            plot_confusion_matrix(y_test, y_pred_best, 'TabNet', best_thresh,
                                 output_dir / 'confusion_matrix_tabnet_test_best_threshold_normalized.png', normalize=True)
            
            results['tabnet'] = {
                'threshold_0.5': {
                    'threshold': 0.5,
                    'confusion_matrix': cm_05.tolist(),
                    'metrics': metrics_05
                },
                'best_threshold': {
                    'threshold': float(best_thresh),
                    'confusion_matrix': cm_best.tolist(),
                    'metrics': metrics_best
                },
                'n_test_samples': int(len(y_test)),
                'n_folds_used': len(test_predictions)
            }
            
            print(f"\nThreshold = 0.5:")
            print(f"  Confusion Matrix:\n{cm_05}")
            print(f"  Accuracy: {metrics_05['Accuracy']:.4f}")
            print(f"  Precision: {metrics_05['Precision']:.4f}")
            print(f"  Recall: {metrics_05['Recall (Sensitivity)']:.4f}")
            print(f"  F1-Score: {metrics_05['F1-Score']:.4f}")
            
            print(f"\nBest Threshold = {best_thresh:.3f} (F1 = {best_f1:.4f}):")
            print(f"  Confusion Matrix:\n{cm_best}")
            print(f"  Accuracy: {metrics_best['Accuracy']:.4f}")
            print(f"  Precision: {metrics_best['Precision']:.4f}")
            print(f"  Recall: {metrics_best['Recall (Sensitivity)']:.4f}")
            print(f"  F1-Score: {metrics_best['F1-Score']:.4f}")
    
    # Lưu kết quả JSON
    with open(output_dir / 'confusion_matrices_test_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Đã lưu confusion matrices (TEST SET) vào: {output_dir}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

