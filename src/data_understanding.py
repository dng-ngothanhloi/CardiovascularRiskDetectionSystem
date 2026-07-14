"""
Data Understanding Module
Implements comprehensive EDA, correlation analysis, and causal discovery.
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from .utils import load_config, get_logger, ensure_dirs, set_seeds

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def zscore_outliers(df: pd.DataFrame, cols: List[str], z: float = 4.0) -> pd.DataFrame:
    """
    Detect outliers using Z-score method.
    
    Args:
        df: Input dataframe
        cols: List of column names to check for outliers
        z: Z-score threshold (default: 4.0)
        
    Returns:
        Boolean dataframe indicating outliers (True = outlier)
    """
    outlier_mask = pd.DataFrame(False, index=df.index, columns=cols)
    
    for col in cols:
        if col not in df.columns:
            continue
        # Select only numeric values (exclude NaN, inf)
        col_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(col_data) < 2:
            continue  # Need at least 2 values to compute std
        
        mean_val = col_data.mean()
        std_val = col_data.std(ddof=0)
        
        if std_val == 0:
            continue  # Constant column, no outliers
        
        # Calculate z-scores
        zvals = np.abs((df[col] - mean_val) / std_val)
        # Replace inf/NaN with False
        zvals = zvals.replace([np.inf, -np.inf, np.nan], 0)
        outlier_mask[col] = zvals > z
    
    return outlier_mask


def iqr_outliers(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Outliers are defined as values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    
    Args:
        df: Input dataframe
        cols: List of column names to check for outliers
        
    Returns:
        Boolean dataframe indicating outliers (True = outlier)
    """
    outlier_mask = pd.DataFrame(False, index=df.index, columns=cols)
    
    for col in cols:
        if col not in df.columns:
            continue
        
        # Select only numeric values (exclude NaN, inf)
        col_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(col_data) < 4:
            continue  # Need at least 4 values for quartiles
        
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            continue  # No variation, skip
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        # Mark inf/NaN as non-outliers (False)
        outlier_mask[col] = outlier_mask[col].fillna(False)
    
    return outlier_mask


def plot_distributions(df: pd.DataFrame, num_cols: List[str], 
                      target_col: str, output_dir: str) -> None:
    """
    Generate distribution plots (histograms and box plots) for numeric features.
    
    Args:
        df: Input dataframe
        num_cols: List of numeric column names
        target_col: Target variable column name
        output_dir: Directory to save plots
    """
    logger = get_logger()
    
    # Check if target column exists
    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found in dataframe. Skipping distribution plots.")
        logger.warning(f"Available columns: {list(df.columns)}")
        return
    
    # Check if there are any numeric columns
    if not num_cols or len(num_cols) == 0:
        logger.warning("No numeric columns found. Skipping distribution plots.")
        return
    
    # Histograms for each numeric feature by target
    n_cols = min(4, len(num_cols))
    n_cols = max(1, n_cols)  # Ensure at least 1 column to avoid division by zero
    n_rows = (len(num_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(num_cols[:len(axes)]):
        ax = axes[idx]
        # Plot histogram for each target class (handle NaN/inf)
        for target_val in sorted(df[target_col].unique()):
            subset = df[df[target_col] == target_val][col]
            # Remove NaN and inf values
            subset_clean = subset.replace([np.inf, -np.inf], np.nan).dropna()
            if len(subset_clean) > 0:
                ax.hist(subset_clean, alpha=0.6, bins=30, label=f'{target_col}={target_val}', 
                       color='blue' if target_val == 0 else 'red')
        ax.set_title(f'Distribution of {col} by {target_col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(num_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/distributions_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved distribution histograms to {output_dir}/distributions_histogram.png")
    
    # Box plots
    major, minor = map(int, matplotlib.__version__.split(".")[:2])
    n_cols = min(4, len(num_cols))
    n_cols = max(1, n_cols)  # Ensure at least 1 column to avoid division by zero
    n_rows = (len(num_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(num_cols[:len(axes)]):
        ax = axes[idx]
        # Create boxplot manually to avoid KeyError (handle NaN/inf)
        box_data = []
        box_labels = []
        for target_val in sorted(df[target_col].unique()):
            subset = df[df[target_col] == target_val][col]
            # Remove NaN and inf values
            subset_clean = subset.replace([np.inf, -np.inf], np.nan).dropna()
            if len(subset_clean) > 0:
                box_data.append(subset_clean)
                box_labels.append(f'{target_col}={target_val}')
        
        if box_data:
            if (major, minor) >= (3, 9):
                ax.boxplot(box_data, tick_labels=box_labels)
            else:
                ax.boxplot(box_data, labels=box_labels)
                	
        ax.set_title(f'Box plot: {col} by {target_col}')
        ax.set_xlabel(target_col)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
    
    for idx in range(len(num_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/distributions_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved box plots to {output_dir}/distributions_boxplot.png")


def causal_discovery_pc(df: pd.DataFrame, causal_vars: List[str], 
                       output_dir: str) -> Dict[str, Any]:
    """
    Perform causal discovery using PC Algorithm (Peter-Clark).
    
    Note: Requires 'causal-learn' or 'pgmpy' library for PC algorithm.
    Falls back to correlation-based graph if library not available.
    
    Args:
        df: Input dataframe
        causal_vars: List of variables for causal discovery (e.g., ['age', 'cholesterol', 'ap_hi', 'ap_lo'])
        output_dir: Directory to save results
        
    Returns:
        Dictionary with causal graph information
    """
    logger = get_logger()
    
    try:
        # Try using causal-learn (preferred)
        try:
            from causallearn.search.ConstraintBased.PC import pc
            from causallearn.utils.GraphUtils import GraphUtils
            
            data = df[causal_vars].values
            cg = pc(data, alpha=0.05)
            
            # Save graph visualization
            pdy = GraphUtils.to_pydot(cg)
            pdy.write_png(f'{output_dir}/causal_dag_pc.png')
            
            logger.info("PC Algorithm causal discovery completed using causal-learn")
            
            return {
                'method': 'PC_Algorithm',
                'library': 'causal-learn',
                'graph_file': f'{output_dir}/causal_dag_pc.png'
            }
            
        except ImportError:
            # Try using pgmpy as alternative
            try:
                from pgmpy.estimators import PC
                from pgmpy.models import BayesianNetwork
                
                # Run PC algorithm
                pc_estimator = PC(df[causal_vars])
                dag = pc_estimator.estimate()
                
                # Save to file (simple representation)
                with open(f'{output_dir}/causal_dag_pc.txt', 'w') as f:
                    f.write(str(dag.edges()))
                
                logger.info("PC Algorithm causal discovery completed using pgmpy")
                
                return {
                    'method': 'PC_Algorithm',
                    'library': 'pgmpy',
                    'edges': list(dag.edges()),
                    'graph_file': f'{output_dir}/causal_dag_pc.txt'
                }
                
            except ImportError:
                logger.warning("Causal discovery libraries not available. Using correlation-based graph.")
                return causal_discovery_correlation(df, causal_vars, output_dir)
                
    except Exception as e:
        logger.warning(f"Causal discovery with PC algorithm failed: {e}. Using correlation-based graph.")
        return causal_discovery_correlation(df, causal_vars, output_dir)


def causal_discovery_notears(df: pd.DataFrame, causal_vars: List[str], 
                            output_dir: str) -> Dict[str, Any]:
    """
    Perform causal discovery using NOTEARS algorithm.
    
    NOTEARS (Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian for Structure learning)
    
    Args:
        df: Input dataframe
        causal_vars: List of variables for causal discovery
        output_dir: Directory to save results
        
    Returns:
        Dictionary with causal graph information
    """
    logger = get_logger()
    
    try:
        from notears import notears_linear
        
        # Prepare data
        data = df[causal_vars].values
        data = StandardScaler().fit_transform(data)
        
        # Run NOTEARS
        W_est = notears_linear(data, lambda1=0.1, loss_type='l2')
        
        # Convert to graph structure
        threshold = 0.3  # Threshold for edge detection
        edges = []
        for i in range(len(causal_vars)):
            for j in range(len(causal_vars)):
                if abs(W_est[i, j]) > threshold and i != j:
                    edges.append((causal_vars[i], causal_vars[j]))
        
        # Save graph visualization
        plot_causal_dag(causal_vars, edges, output_dir, 'notears')
        
        logger.info(f"NOTEARS causal discovery completed. Found {len(edges)} edges.")
        
        return {
            'method': 'NOTEARS',
            'edges': edges,
            'weight_matrix': W_est.tolist(),
            'graph_file': f'{output_dir}/causal_dag_notears.png'
        }
        
    except ImportError:
        logger.warning("NOTEARS library not available. Install with: pip install notears")
        return {}
    except Exception as e:
        logger.warning(f"NOTEARS causal discovery failed: {e}")
        return {}


def causal_discovery_correlation(df: pd.DataFrame, causal_vars: List[str], 
                                 output_dir: str) -> Dict[str, Any]:
    """
    Fallback: Create correlation-based causal graph visualization.
    
    Args:
        df: Input dataframe
        causal_vars: List of variables
        output_dir: Directory to save results
        
    Returns:
        Dictionary with graph information
    """
    logger = get_logger()
    
    # Calculate correlation matrix
    corr_matrix = df[causal_vars].corr()
    
    # Create edges based on correlation threshold
    threshold = 0.3
    edges = []
    for i, var1 in enumerate(causal_vars):
        for j, var2 in enumerate(causal_vars):
            if i < j and abs(corr_matrix.loc[var1, var2]) > threshold:
                edges.append((var1, var2))
    
    # Plot correlation-based graph
    plot_causal_dag(causal_vars, edges, output_dir, 'correlation')
    
    logger.info("Correlation-based causal graph created")
    
    return {
        'method': 'Correlation_based',
        'edges': edges,
        'correlation_matrix': corr_matrix.to_dict(),
        'graph_file': f'{output_dir}/causal_dag_correlation.png'
    }


def adversarial_validation(df: pd.DataFrame, feature_cols: List[str], 
                           test_size: float = 0.2, output_dir: str = "artifacts/eda") -> Dict[str, Any]:
    """
    Perform adversarial validation to detect train/test distribution shift.
    
    The idea is to train a classifier to distinguish between train and test sets.
    If the classifier can distinguish them well (AUC > 0.5), there's a distribution shift.
    
    Args:
        df: Input dataframe
        feature_cols: List of feature column names to use
        test_size: Proportion of data to use as test set (default: 0.2)
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing adversarial validation results
    """
    logger = get_logger()
    logger.info("Performing adversarial validation for bias and drift detection...")
    
    try:
        # Prepare features (handle missing values and inf)
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        # Fill any remaining NaN with 0 (if median is NaN)
        X = X.fillna(0)
        
        # Split data into train/test (80/20)
        X_train, X_test, _, _ = train_test_split(
            X, df['cardio'] if 'cardio' in df.columns else np.zeros(len(X)),
            test_size=test_size, random_state=42, stratify=df['cardio'] if 'cardio' in df.columns else None
        )
        
        # Create labels: 0 for train, 1 for test
        y_train = np.zeros(len(X_train))
        y_test = np.ones(len(X_test))
        
        # Combine train and test data
        X_combined = pd.concat([X_train, X_test], axis=0)
        y_combined = np.concatenate([y_train, y_test])
        
        # Train a classifier to distinguish train vs test
        # Use RandomForest for robustness
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_combined, y_combined)
        
        # Predict probabilities
        y_pred_proba = clf.predict_proba(X_combined)[:, 1]
        
        # Calculate AUC
        auc_score = roc_auc_score(y_combined, y_pred_proba)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_combined, y_pred_proba)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'Adversarial Classifier (AUC = {auc_score:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)', linewidth=1.5)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Adversarial Validation: Train vs Test Distribution Shift Detection', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/adversarial_validation_roc.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance from adversarial classifier
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv(f'{output_dir}/adversarial_feature_importance.csv', index=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        sns.barplot(data=feature_importance.head(15), x='importance', y='feature', palette='Reds_r')
        plt.title('Top Features Contributing to Train/Test Distribution Shift', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/adversarial_feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Interpretation
        if auc_score > 0.6:
            shift_detected = True
            severity = "High" if auc_score > 0.7 else "Medium"
            interpretation = f"Distribution shift detected! AUC = {auc_score:.4f}. " \
                           f"Train and test sets are significantly different. " \
                           f"Severity: {severity}"
        elif auc_score > 0.55:
            shift_detected = True
            severity = "Low"
            interpretation = f"Minor distribution shift detected. AUC = {auc_score:.4f}. " \
                           f"Train and test sets show slight differences. " \
                           f"Severity: {severity}"
        else:
            shift_detected = False
            severity = "None"
            interpretation = f"No significant distribution shift detected. AUC = {auc_score:.4f}. " \
                           f"Train and test sets are similar."
        
        logger.info(f"Adversarial Validation Results:")
        logger.info(f"  AUC Score: {auc_score:.4f}")
        logger.info(f"  Shift Detected: {shift_detected}")
        logger.info(f"  Severity: {severity}")
        logger.info(f"  Interpretation: {interpretation}")
        
        # Save results
        results = {
            'auc_score': float(auc_score),
            'shift_detected': shift_detected,
            'severity': severity,
            'interpretation': interpretation,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'top_shift_features': feature_importance.head(10).to_dict('records')
        }
        
        import json
        with open(f'{output_dir}/adversarial_validation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Adversarial validation results saved to {output_dir}/adversarial_validation_results.json")
        
        return results
        
    except Exception as e:
        logger.warning(f"Adversarial validation failed: {e}")
        return {
            'auc_score': None,
            'shift_detected': None,
            'error': str(e)
        }


# NOTE: create_derived_variables() function has been removed
# Derived variables should be created in feature_engineering.py instead
# This function is kept commented for reference only:
#
# def create_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     DEPRECATED: This function has been moved to feature_engineering.py
#     Create derived variables: age_years, BMI, pulse_pressure, age_group, bmi_group.
#     """
#     pass


def flag_data_quality_issues(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Flag data quality issues: ap_lo > ap_hi, ap_hi>240, ap_lo<40, duplicates.
    
    Args:
        df: Input dataframe
        output_dir: Directory to save results
        
    Returns:
        Dataframe with quality flags
    """
    logger = get_logger()
    df = df.copy()
    
    # Initialize quality flags
    df['flag_ap_lo_gt_ap_hi'] = False
    df['flag_ap_hi_high'] = False
    df['flag_ap_lo_low'] = False
    df['flag_ap_hi_negative'] = False
    df['flag_ap_lo_negative'] = False
    df['flag_bp_negative'] = False
    df['flag_duplicate'] = False
    
    # Flag: ap_lo > ap_hi (logically impossible)
    if 'ap_lo' in df.columns and 'ap_hi' in df.columns:
        df['flag_ap_lo_gt_ap_hi'] = df['ap_lo'] > df['ap_hi']
    
    # Flag: ap_hi > 240 (very high systolic) or negative
    if 'ap_hi' in df.columns:
        df['flag_ap_hi_high'] = df['ap_hi'] > 240
        df['flag_ap_hi_negative'] = df['ap_hi'] < 0
    
    # Flag: ap_lo < 40 (very low diastolic) or negative
    if 'ap_lo' in df.columns:
        df['flag_ap_lo_low'] = df['ap_lo'] < 40
        df['flag_ap_lo_negative'] = df['ap_lo'] < 0
    
    # Flag: Negative blood pressure values (impossible)
    if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
        df['flag_bp_negative'] = (df['ap_hi'] < 0) | (df['ap_lo'] < 0)
    
    # Flag: duplicates
    df['flag_duplicate'] = df.duplicated(keep=False)
    
    # Count issues
    quality_counts = {
        'ap_lo > ap_hi': df['flag_ap_lo_gt_ap_hi'].sum(),
        'ap_hi > 240': df['flag_ap_hi_high'].sum(),
        'ap_lo < 40': df['flag_ap_lo_low'].sum(),
        'ap_hi negative': df['flag_ap_hi_negative'].sum(),
        'ap_lo negative': df['flag_ap_lo_negative'].sum(),
        'duplicates': df['flag_duplicate'].sum()
    }
    
    # Save flagged records (any quality issue)
    flagged_records = df[df['flag_ap_lo_gt_ap_hi'] | df['flag_ap_hi_high'] | 
                         df['flag_ap_lo_low'] | df['flag_ap_hi_negative'] | 
                         df['flag_ap_lo_negative'] | df['flag_duplicate']].copy()
    
    if len(flagged_records) > 0:
        flagged_records.to_csv(f'{output_dir}/flagged_records.csv', index=False)
        logger.info(f"Saved {len(flagged_records)} flagged records to {output_dir}/flagged_records.csv")
    
    # Plot quality issues bar chart
    plt.figure(figsize=(10, 6))
    quality_df = pd.DataFrame(list(quality_counts.items()), columns=['Issue', 'Count'])
    sns.barplot(data=quality_df, x='Issue', y='Count', palette='Reds_r')
    plt.title('Data Quality Issues: Count of Anomalous Records', fontsize=14, fontweight='bold')
    plt.xlabel('Issue Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/data_quality_issues.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save quality counts
    quality_df.to_csv(f'{output_dir}/data_quality_counts.csv', index=False)
    logger.info(f"Data quality issues:\n{quality_counts}")
    
    return df


def plot_age_cardio_relationship(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot Age & Cardio Relationship: Histogram + density of age_years by cardio.
    
    Args:
        df: Input dataframe with age_years column
        output_dir: Directory to save plots
    """
    logger = get_logger()
    
    if 'age_years' not in df.columns or 'cardio' not in df.columns:
        logger.warning("Missing age_years or cardio columns. Skipping age-cardio relationship plot.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram (handle NaN/inf)
    ax1 = axes[0]
    age_no_disease = df[df['cardio'] == 0]['age_years'].replace([np.inf, -np.inf], np.nan).dropna()
    age_disease = df[df['cardio'] == 1]['age_years'].replace([np.inf, -np.inf], np.nan).dropna()
    if len(age_no_disease) > 0:
        ax1.hist(age_no_disease, bins=30, alpha=0.6, label='No Disease', color='blue')
    if len(age_disease) > 0:
        ax1.hist(age_disease, bins=30, alpha=0.6, label='Disease', color='red')
    ax1.set_xlabel('Age (years)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Age Distribution by Cardiovascular Disease Status', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Density plot (handle NaN/inf)
    ax2 = axes[1]
    if len(age_no_disease) > 0:
        age_no_disease.plot.density(ax=ax2, label='No Disease', color='blue', linewidth=2)
    if len(age_disease) > 0:
        age_disease.plot.density(ax=ax2, label='Disease', color='red', linewidth=2)
    ax2.set_xlabel('Age (years)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Age Density by Cardiovascular Disease Status', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/age_cardio_relationship.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved age-cardio relationship plot to {output_dir}/age_cardio_relationship.png")


def plot_bmi_cardio_scatter(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot BMI vs Cardio Status: Scatterplot (BMI ~ Age) colored by cardio.
    Highlight BMI > 30 (obese).
    
    Args:
        df: Input dataframe with bmi and age_years columns
        output_dir: Directory to save plots
    """
    logger = get_logger()
    
    if 'bmi' not in df.columns or 'age_years' not in df.columns or 'cardio' not in df.columns:
        logger.warning("Missing bmi, age_years, or cardio columns. Skipping BMI-cardio scatter plot.")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Scatter plot (filter out NaN/inf values)
    valid_mask = (
        df['age_years'].notna() & 
        df['bmi'].notna() & 
        np.isfinite(df['age_years']) & 
        np.isfinite(df['bmi']) &
        df['cardio'].notna()
    )
    df_plot = df[valid_mask]
    
    if len(df_plot) > 0:
        scatter = plt.scatter(df_plot['age_years'], df_plot['bmi'], c=df_plot['cardio'], 
                             cmap='RdYlGn_r', alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    else:
        logger.warning("No valid data points for BMI-cardio scatter plot (all NaN/inf)")
        plt.close()
        return
    
    # Highlight obese region (BMI > 30)
    plt.axhline(y=30, color='red', linestyle='--', linewidth=2, label='Obese threshold (BMI=30)')
    plt.fill_between(plt.xlim(), 30, 40, alpha=0.2, color='red', label='Obese region')
    
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('BMI', fontsize=12)
    plt.title('BMI vs Age by Cardiovascular Disease Status\n(Red=High Risk, Green=Low Risk)', 
             fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Cardio (0=No, 1=Yes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bmi_cardio_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved BMI-cardio scatter plot to {output_dir}/bmi_cardio_scatter.png")


def plot_cholesterol_glucose_levels(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot Cholesterol & Glucose Levels: Bar plot distribution of levels 1-2-3 
    and cardio rate by level.
    
    Args:
        df: Input dataframe
        output_dir: Directory to save plots
    """
    logger = get_logger()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Cholesterol plot
    if 'cholesterol' in df.columns and 'cardio' in df.columns:
        ax1 = axes[0]
        chol_counts = df['cholesterol'].value_counts().sort_index()
        chol_cardio_rate = df.groupby('cholesterol')['cardio'].mean()
        
        x_pos = np.arange(len(chol_counts))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, chol_counts.values, width, label='Count', alpha=0.7, color='skyblue')
        ax1_twin = ax1.twinx()
        bars2 = ax1_twin.bar(x_pos + width/2, chol_cardio_rate.values * 100, width, 
                            label='Cardio Rate (%)', alpha=0.7, color='coral')
        
        ax1.set_xlabel('Cholesterol Level (1=normal, 2=above, 3=well above)', fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1_twin.set_ylabel('Cardio Rate (%)', fontsize=11)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(chol_counts.index)
        ax1.set_title('Cholesterol Levels: Distribution & Cardio Rate', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
    
    # Glucose plot
    if 'gluc' in df.columns and 'cardio' in df.columns:
        ax2 = axes[1]
        gluc_counts = df['gluc'].value_counts().sort_index()
        gluc_cardio_rate = df.groupby('gluc')['cardio'].mean()
        
        x_pos = np.arange(len(gluc_counts))
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, gluc_counts.values, width, label='Count', alpha=0.7, color='lightgreen')
        ax2_twin = ax2.twinx()
        bars2 = ax2_twin.bar(x_pos + width/2, gluc_cardio_rate.values * 100, width, 
                            label='Cardio Rate (%)', alpha=0.7, color='coral')
        
        ax2.set_xlabel('Glucose Level (1=normal, 2=above, 3=well above)', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2_twin.set_ylabel('Cardio Rate (%)', fontsize=11)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(gluc_counts.index)
        ax2.set_title('Glucose Levels: Distribution & Cardio Rate', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cholesterol_glucose_levels.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved cholesterol & glucose levels plot to {output_dir}/cholesterol_glucose_levels.png")


def plot_lifestyle_habits(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot Lifestyle Habits: Clustered bar chart for smoke, alco, active 
    showing cardio rate by habit.
    
    Args:
        df: Input dataframe
        output_dir: Directory to save plots
    """
    logger = get_logger()
    
    if not all(col in df.columns for col in ['smoke', 'alco', 'active', 'cardio']):
        logger.warning("Missing lifestyle columns. Skipping lifestyle habits plot.")
        return
    
    # Calculate cardio rates
    lifestyle_cols = ['smoke', 'alco', 'active']
    lifestyle_labels = ['Smoking', 'Alcohol', 'Physical Activity']
    
    rates = {}
    counts = {}
    for col, label in zip(lifestyle_cols, lifestyle_labels):
        rates[label] = df.groupby(col)['cardio'].mean() * 100
        counts[label] = df.groupby(col).size()
    
    # Create clustered bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Cardio rate plot
    x = np.arange(len(lifestyle_labels))
    width = 0.35
    
    for i, (col, label) in enumerate(zip(lifestyle_cols, lifestyle_labels)):
        pos = x[i]
        no_yes = rates[label].index.tolist()
        values = rates[label].values
        
        ax1.bar(pos - width/2, values[0] if len(values) > 0 else 0, width, 
               label=f'{label} (No)' if i == 0 else '', alpha=0.7, color='lightblue')
        if len(values) > 1:
            ax1.bar(pos + width/2, values[1], width, 
                   label=f'{label} (Yes)' if i == 0 else '', alpha=0.7, color='coral')
    
    ax1.set_xlabel('Lifestyle Habit', fontsize=12)
    ax1.set_ylabel('Cardio Rate (%)', fontsize=12)
    ax1.set_title('Cardiovascular Disease Rate by Lifestyle Habits', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(lifestyle_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Count plot
    for i, (col, label) in enumerate(zip(lifestyle_cols, lifestyle_labels)):
        pos = x[i]
        values = counts[label].values
        labels_list = counts[label].index.tolist()
        
        ax2.bar(pos - width/2, values[0] if len(values) > 0 else 0, width, 
               label=f'{label} (No)' if i == 0 else '', alpha=0.7, color='lightgreen')
        if len(values) > 1:
            ax2.bar(pos + width/2, values[1], width, 
                   label=f'{label} (Yes)' if i == 0 else '', alpha=0.7, color='orange')
    
    ax2.set_xlabel('Lifestyle Habit', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Sample Count by Lifestyle Habits', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(lifestyle_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/lifestyle_habits.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved lifestyle habits plot to {output_dir}/lifestyle_habits.png")


def generate_data_recommendations(df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
    """
    Generate comprehensive data quality recommendations based on EDA results.
    
    Categories:
    1. Errors fixed automatically in code
    2. Errors requiring manual data correction
    3. Outlier handling recommendations
    4. Other recommendations
    
    Args:
        df: Input dataframe
        output_dir: Directory to save recommendations
        
    Returns:
        Dictionary containing recommendations
    """
    logger = get_logger()
    recommendations = {
        "code_fixed_errors": [],
        "manual_data_fixes": [],
        "outlier_handling": [],
        "other_recommendations": []
    }
    
    # ==================== 1. Errors Fixed Automatically in Code ====================
    recommendations["code_fixed_errors"].append({
        "issue": "ZeroDivisionError in plot_distributions when num_cols is empty",
        "status": "✅ Fixed",
        "solution": "Added check for empty num_cols before plotting and ensured n_cols >= 1",
        "location": "src/data_understanding.py: plot_distributions()",
        "impact": "Prevents crashes when no numeric columns are found"
    })
    
    recommendations["code_fixed_errors"].append({
        "issue": "Improved numeric column detection",
        "status": "✅ Fixed",
        "solution": "Changed from dtype checking to pd.api.types.is_numeric_dtype() for better accuracy",
        "location": "src/data_understanding.py: main()",
        "impact": "Better detection of numeric columns including derived variables"
    })
    
    recommendations["code_fixed_errors"].append({
        "issue": "Missing error handling for empty num_cols in multiple functions",
        "status": "✅ Fixed",
        "solution": "Added checks in outlier detection, correlation analysis, MI calculation, and adversarial validation",
        "location": "src/data_understanding.py: multiple functions",
        "impact": "Prevents errors when processing datasets with no numeric columns"
    })
    
    # ==================== 2. Errors Requiring Manual Data Correction ====================
    # Check for data quality issues
    if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
        ap_lo_gt_hi = (df['ap_lo'] > df['ap_hi']).sum()
        if ap_lo_gt_hi > 0:
            recommendations["manual_data_fixes"].append({
                "issue": f"ap_lo > ap_hi: {ap_lo_gt_hi} records",
                "severity": "High",
                "description": "Diastolic pressure cannot be greater than systolic pressure (physiologically impossible)",
                "action_required": "Manual review and correction required",
                "suggested_fix": "Review flagged records in artifacts/eda/flagged_records.csv and correct or remove invalid records",
                "impact": f"{ap_lo_gt_hi} records ({ap_lo_gt_hi/len(df)*100:.2f}%) have invalid blood pressure values"
            })
    
    if 'ap_hi' in df.columns:
        ap_hi_high = (df['ap_hi'] > 240).sum()
        if ap_hi_high > 0:
            recommendations["manual_data_fixes"].append({
                "issue": f"ap_hi > 240 mmHg: {ap_hi_high} records",
                "severity": "Medium",
                "description": "Extremely high systolic blood pressure (may indicate measurement error or critical condition)",
                "action_required": "Review and verify with medical team",
                "suggested_fix": "Flag for medical review or remove if confirmed measurement error",
                "impact": f"{ap_hi_high} records ({ap_hi_high/len(df)*100:.2f}%) have extreme values"
            })
    
    if 'ap_lo' in df.columns:
        ap_lo_low = (df['ap_lo'] < 40).sum()
        if ap_lo_low > 0:
            recommendations["manual_data_fixes"].append({
                "issue": f"ap_lo < 40 mmHg: {ap_lo_low} records",
                "severity": "Medium",
                "description": "Extremely low diastolic blood pressure (may indicate measurement error or critical condition)",
                "action_required": "Review and verify with medical team",
                "suggested_fix": "Flag for medical review or remove if confirmed measurement error",
                "impact": f"{ap_lo_low} records ({ap_lo_low/len(df)*100:.2f}%) have extreme values"
            })
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        recommendations["manual_data_fixes"].append({
            "issue": f"Duplicate records: {duplicates} records",
            "severity": "Low",
            "description": "Exact duplicate rows found in dataset",
            "action_required": "Decide whether to keep or remove duplicates",
            "suggested_fix": "Review duplicates in artifacts/eda/flagged_records.csv and remove if confirmed duplicates",
            "impact": f"{duplicates} records ({duplicates/len(df)*100:.2f}%) are duplicates"
        })
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    if len(missing_cols) > 0:
        recommendations["manual_data_fixes"].append({
            "issue": f"Missing values in {len(missing_cols)} columns",
            "severity": "Medium",
            "description": f"Columns with missing values: {', '.join(missing_cols.index.tolist())}",
            "action_required": "Review missing data patterns and decide on imputation strategy",
            "suggested_fix": "Use MICE or KNN imputation in preprocessing.py, or remove records if missingness is systematic",
            "impact": f"Total missing values: {missing_counts.sum()} ({missing_counts.sum()/(len(df)*len(df.columns))*100:.2f}%)"
        })
    
    # ==================== 3. Outlier Handling Recommendations ====================
    if 'bmi' in df.columns:
        bmi_outliers_high = (df['bmi'] > 50).sum()
        bmi_outliers_low = (df['bmi'] < 10).sum()
        if bmi_outliers_high > 0 or bmi_outliers_low > 0:
            recommendations["outlier_handling"].append({
                "feature": "BMI",
                "method": "Winsorization or Truncation",
                "description": f"Extreme BMI values: {bmi_outliers_high} > 50, {bmi_outliers_low} < 10",
                "recommendation": "Winsorize at 1st and 99th percentiles, or cap at [10, 50]",
                "rationale": "BMI values outside this range are likely measurement errors or data entry mistakes"
            })
    
    if 'height' in df.columns:
        height_outliers_high = ((df['height'] > 220) | (df['height'] < 100)).sum()
        if height_outliers_high > 0:
            recommendations["outlier_handling"].append({
                "feature": "Height",
                "method": "Truncation",
                "description": f"Height values outside reasonable range [100, 220] cm: {height_outliers_high} records",
                "recommendation": "Cap height values at [100, 220] cm or remove extreme outliers",
                "rationale": "Height outside this range is physiologically unlikely for adults"
            })
    
    if 'weight' in df.columns:
        weight_outliers = ((df['weight'] > 200) | (df['weight'] < 30)).sum()
        if weight_outliers > 0:
            recommendations["outlier_handling"].append({
                "feature": "Weight",
                "method": "Winsorization",
                "description": f"Weight values outside reasonable range [30, 200] kg: {weight_outliers} records",
                "recommendation": "Winsorize at 1st and 99th percentiles or cap at [30, 200] kg",
                "rationale": "Extreme weights may indicate measurement errors"
            })
    
    if 'age_years' in df.columns:
        age_outliers = ((df['age_years'] > 100) | (df['age_years'] < 18)).sum()
        if age_outliers > 0:
            recommendations["outlier_handling"].append({
                "feature": "Age",
                "method": "Review and Filter",
                "description": f"Age values outside clinical study range [18, 100] years: {age_outliers} records",
                "recommendation": "Review age outliers - may be valid for very old patients or indicate data entry errors",
                "rationale": "Age outside typical clinical study range requires domain expertise review"
            })
    
    recommendations["outlier_handling"].append({
        "feature": "General Outlier Strategy",
        "method": "Multiple Approaches",
        "description": "Use IQR method for detection, then apply appropriate treatment",
        "recommendation": "1. Z-score method: Flag outliers with |z| > 4\n2. IQR method: Flag values outside [Q1-1.5*IQR, Q3+1.5*IQR]\n3. Treatment: Winsorization for extreme values, removal for impossible values",
        "rationale": "Different features require different outlier handling strategies based on domain knowledge"
    })
    
    # ==================== 4. Other Recommendations ====================
    # Class imbalance
    if 'cardio' in df.columns:
        cardio_dist = df['cardio'].value_counts(normalize=True)
        imbalance = abs(cardio_dist.get(0, 0) - cardio_dist.get(1, 0))
        if imbalance > 0.1:
            recommendations["other_recommendations"].append({
                "category": "Class Imbalance",
                "issue": f"Class imbalance detected: {cardio_dist.get(0, 0)*100:.1f}% vs {cardio_dist.get(1, 0)*100:.1f}%",
                "recommendation": "Use StratifiedKFold for cross-validation and consider class_weight='balanced' in models",
                "priority": "High"
            })
    
    # Feature engineering
    recommendations["other_recommendations"].append({
        "category": "Feature Engineering",
        "issue": "Derived variables created",
        "recommendation": "⚠️  Derived variables should be created in feature_engineering.py. Consider: BMI × age interaction, pulse_pressure × cholesterol interaction",
        "priority": "Medium"
    })
    
    # Feature selection
    recommendations["other_recommendations"].append({
        "category": "Feature Selection",
        "issue": "High correlation between features",
        "recommendation": "Review correlation matrix. Consider removing highly correlated features (corr > 0.9) to reduce multicollinearity",
        "priority": "Medium"
    })
    
    # Data validation
    recommendations["other_recommendations"].append({
        "category": "Data Validation",
        "issue": "Adversarial validation for train/test shift",
        "recommendation": "Monitor adversarial validation AUC. If AUC > 0.6, consider stratified sampling or time-based splitting",
        "priority": "Medium"
    })
    
    # Save recommendations
    import json
    with open(f'{output_dir}/data_recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=2, default=str)
    
    # Generate markdown report
    md_content = generate_recommendations_markdown(recommendations)
    with open(f'{output_dir}/data_recommendations.md', 'w') as f:
        f.write(md_content)
    
    logger.info(f"Data recommendations saved to {output_dir}/data_recommendations.json and .md")
    
    return recommendations


def generate_recommendations_markdown(recommendations: Dict[str, Any]) -> str:
    """
    Generate markdown report from recommendations dictionary.
    
    Args:
        recommendations: Dictionary containing recommendations
        
    Returns:
        Markdown formatted string
    """
    md = []
    md.append("# Data Quality Recommendations & Action Items\n")
    md.append("Generated from EDA analysis results.\n")
    
    # 1. Code Fixed Errors
    md.append("## 1. ✅ Errors Fixed Automatically in Code\n")
    if recommendations["code_fixed_errors"]:
        for i, item in enumerate(recommendations["code_fixed_errors"], 1):
            md.append(f"### {i}. {item['issue']}\n")
            md.append(f"- **Status:** {item['status']}\n")
            md.append(f"- **Solution:** {item['solution']}\n")
            md.append(f"- **Location:** {item['location']}\n")
            md.append(f"- **Impact:** {item['impact']}\n\n")
    else:
        md.append("No code fixes needed.\n\n")
    
    # 2. Manual Data Fixes
    md.append("## 2. ⚠️ Errors Requiring Manual Data Correction\n")
    if recommendations["manual_data_fixes"]:
        for i, item in enumerate(recommendations["manual_data_fixes"], 1):
            md.append(f"### {i}. {item['issue']}\n")
            md.append(f"- **Severity:** {item['severity']}\n")
            md.append(f"- **Description:** {item['description']}\n")
            md.append(f"- **Action Required:** {item['action_required']}\n")
            md.append(f"- **Suggested Fix:** {item['suggested_fix']}\n")
            md.append(f"- **Impact:** {item['impact']}\n\n")
    else:
        md.append("No manual data corrections needed.\n\n")
    
    # 3. Outlier Handling
    md.append("## 3. 📊 Outlier Handling Recommendations\n")
    if recommendations["outlier_handling"]:
        for i, item in enumerate(recommendations["outlier_handling"], 1):
            md.append(f"### {i}. {item['feature']}\n")
            md.append(f"- **Method:** {item['method']}\n")
            md.append(f"- **Description:** {item['description']}\n")
            md.append(f"- **Recommendation:** {item['recommendation']}\n")
            md.append(f"- **Rationale:** {item['rationale']}\n\n")
    else:
        md.append("No specific outlier handling recommendations.\n\n")
    
    # 4. Other Recommendations
    md.append("## 4. 💡 Other Recommendations\n")
    if recommendations["other_recommendations"]:
        for i, item in enumerate(recommendations["other_recommendations"], 1):
            md.append(f"### {i}. {item['category']}: {item['issue']}\n")
            md.append(f"- **Priority:** {item['priority']}\n")
            md.append(f"- **Recommendation:** {item['recommendation']}\n\n")
    else:
        md.append("No additional recommendations.\n\n")
    
    md.append("---\n")
    md.append("*Report generated automatically from EDA results.*\n")
    
    return "\n".join(md)


def generate_key_insights(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate Key Insights & Recommendations summary report.
    
    Args:
        df: Input dataframe
        output_dir: Directory to save report
    """
    logger = get_logger()
    
    insights = []
    
    # Insight 1: Dataset overview
    insights.append("## 1. Dataset Overview")
    insights.append(f"- Total records: {len(df):,}")
    insights.append(f"- Features: {len(df.columns) - 1} (excluding target)")
    insights.append(f"- Target variable: cardio (binary)")
    
    # Insight 2: Class distribution
    if 'cardio' in df.columns:
        cardio_dist = df['cardio'].value_counts(normalize=True) * 100
        insights.append("\n## 2. Class Distribution")
        insights.append(f"- No Disease: {cardio_dist.get(0, 0):.1f}%")
        insights.append(f"- Disease: {cardio_dist.get(1, 0):.1f}%")
        if abs(cardio_dist.get(0, 0) - cardio_dist.get(1, 0)) > 10:
            insights.append("- ⚠️ Class imbalance detected - consider stratified sampling")
    
    # Insight 3: Age analysis
    if 'age_years' in df.columns and 'cardio' in df.columns:
        insights.append("\n## 3. Age & Risk Relationship")
        age_cardio = df.groupby('cardio')['age_years'].mean()
        insights.append(f"- Average age (No Disease): {age_cardio.get(0, 0):.1f} years")
        insights.append(f"- Average age (Disease): {age_cardio.get(1, 0):.1f} years")
        if age_cardio.get(1, 0) > age_cardio.get(0, 0):
            insights.append("- ✅ Older age is associated with higher cardiovascular risk")
    
    # Insight 4: BMI analysis
    if 'bmi' in df.columns and 'cardio' in df.columns:
        insights.append("\n## 4. BMI & Risk Relationship")
        bmi_cardio = df.groupby('cardio')['bmi'].mean()
        insights.append(f"- Average BMI (No Disease): {bmi_cardio.get(0, 0):.1f}")
        insights.append(f"- Average BMI (Disease): {bmi_cardio.get(1, 0):.1f}")
        obese_rate = (df[df['bmi'] > 30]['cardio'].mean() * 100) if 'cardio' in df.columns else 0
        insights.append(f"- Cardio rate for obese (BMI>30): {obese_rate:.1f}%")
    
    # Insight 5: Lifestyle factors
    if all(col in df.columns for col in ['smoke', 'alco', 'active', 'cardio']):
        insights.append("\n## 5. Lifestyle Factors")
        smoke_rate = df.groupby('smoke')['cardio'].mean() * 100
        alco_rate = df.groupby('alco')['cardio'].mean() * 100
        active_rate = df.groupby('active')['cardio'].mean() * 100
        
        insights.append(f"- Smoking increases risk: {smoke_rate.get(1, 0):.1f}% vs {smoke_rate.get(0, 0):.1f}%")
        insights.append(f"- Alcohol increases risk: {alco_rate.get(1, 0):.1f}% vs {alco_rate.get(0, 0):.1f}%")
        insights.append(f"- Physical activity reduces risk: {active_rate.get(0, 0):.1f}% vs {active_rate.get(1, 0):.1f}%")
    
    # Insight 6: Recommendations
    insights.append("\n## 6. Recommendations for Preprocessing & Feature Engineering")
    insights.append("- ⚠️  Derived variables (age_years, BMI, pulse_pressure) should be created in feature_engineering.py")
    insights.append("- ✅ Flag data quality issues: ap_lo > ap_hi, extreme BP values")
    insights.append("- ✅ Consider feature interactions: smoke × active, BMI × age")
    insights.append("- ✅ Use stratified cross-validation due to potential class imbalance")
    insights.append("- ✅ Focus on top MI features: age, BMI, pulse_pressure, cholesterol, glucose")
    
    # Save insights
    insights_text = "\n".join(insights)
    with open(f'{output_dir}/key_insights.txt', 'w') as f:
        f.write(insights_text)
    
    # Also save as markdown
    with open(f'{output_dir}/key_insights.md', 'w') as f:
        f.write(insights_text)
    
    logger.info(f"Key insights saved to {output_dir}/key_insights.txt and .md")
    logger.info("\n" + insights_text)


def plot_causal_dag(variables: List[str], edges: List[Tuple[str, str]], 
                    output_dir: str, method_name: str) -> None:
    """
    Visualize causal DAG using NetworkX and matplotlib.
    
    Args:
        variables: List of variable names
        edges: List of (source, target) tuples representing edges
        output_dir: Directory to save plot
        method_name: Name of the method used (for filename)
    """
    try:
        import networkx as nx
        
        # Create directed graph
        G = nx.DiGraph()
        G.add_nodes_from(variables)
        G.add_edges_from(edges)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Plot
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=3000, alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, alpha=0.6,
                              connectionstyle='arc3,rad=0.1')
        
        plt.title(f'Causal DAG - {method_name.capitalize()} Method\n'
                 f'Variables: {", ".join(variables)}', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/causal_dag_{method_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger = get_logger()
        logger.info(f"Causal DAG visualization saved: {output_dir}/causal_dag_{method_name}.png")
        
    except ImportError:
        logger = get_logger()
        logger.warning("NetworkX not available. Install with: pip install networkx")
    except Exception as e:
        logger = get_logger()
        logger.warning(f"Failed to plot causal DAG: {e}")


def main(args):
    """
    Main function to execute data understanding pipeline.
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    cfg = load_config(args.config)
    logger = get_logger()
    set_seeds()
    
    # Ensure output directories exist
    ensure_dirs("artifacts/eda")
    output_dir = "artifacts/eda"
    
    # Load data
    data_path = cfg.get("paths", {}).get("data", "data/cardio_data.csv")
    logger.info(f"Loading data from: {data_path}")
    
    # Try to detect delimiter (semicolon or comma)
    with open(data_path, 'r') as f:
        first_line = f.readline()
        if ';' in first_line:
            delimiter = ';'
            logger.info("Detected semicolon (;) delimiter in CSV file")
        else:
            delimiter = ','
            logger.info("Using comma (,) delimiter for CSV file")
    
    df = pd.read_csv(data_path, delimiter=delimiter)
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns in raw data: {list(df.columns)}")
    
    target = "cardio"
    
    # Verify target column exists
    if target not in df.columns:
        logger.error(f"Target column '{target}' not found in data!")
        logger.error(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Target column '{target}' not found in dataset. Available columns: {list(df.columns)}")
    
    logger.info(f"Target column '{target}' confirmed in dataset.")
    
    # ==================== 0. Derived Variables ====================
    # NOTE: Derived variables (age_years, BMI, pulse_pressure, etc.) should be created 
    # in feature_engineering.py before running data_understanding.py
    # Check if derived variables exist (for plotting and analysis)
    derived_cols = ['age_years', 'bmi', 'pulse_pressure', 'age_group', 'bmi_group', 'chol_high', 'gluc_high']
    existing_derived = [c for c in derived_cols if c in df.columns]
    if existing_derived:
        logger.info(f"Found existing derived variables: {existing_derived}")
    else:
        logger.info("No derived variables found. They should be created in feature_engineering.py first.")
    
    # ==================== 0.1. Export Schema ====================
    SCHEMA = {
        "age": "Age (days)",
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "gender": "Gender (1=female,2=male)",
        "ap_hi": "Systolic blood pressure",
        "ap_lo": "Diastolic blood pressure",
        "cholesterol": "Cholesterol (1=normal, 2=above normal, 3=well above normal)",
        "gluc": "Glucose (1=normal, 2=above normal, 3=well above normal)",
        "smoke": "Smoking (0/1)",
        "alco": "Alcohol intake (0/1)",
        "active": "Physical activity (0/1)",
        "cardio": "Cardiovascular disease (target)"
    }
    import json
    with open(f'{output_dir}/schema.json', 'w') as f:
        json.dump(SCHEMA, f, indent=2)
    logger.info(f"Schema exported to {output_dir}/schema.json")
    
    # ==================== 0.2. Data Quality Flags ====================
    logger.info("Flagging data quality issues...")
    df = flag_data_quality_issues(df, output_dir)
    
    # Verify target column still exists after processing
    if target not in df.columns:
        logger.error(f"Target column '{target}' lost after processing!")
        logger.error(f"Current columns: {list(df.columns)}")
        raise ValueError(f"Target column '{target}' must be preserved through all processing steps.")
    
    logger.info(f"Target column '{target}' confirmed after processing. DataFrame shape: {df.shape}")
    
    # Identify numeric columns (including derived numeric columns)
    # Exclude: target column, id column (identifier), and flag columns
    exclude_cols = {target, 'id'}
    exclude_cols.update([c for c in df.columns if c.startswith('flag_')])
    
    num_cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        # Check if column is numeric (including int, float, int64, float64)
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
    
    logger.info(f"Found {len(num_cols)} numeric columns (excluding target, id, flags): {num_cols}")
    
    # Warn if no numeric columns found
    if len(num_cols) == 0:
        logger.warning("No numeric columns found! Check data types and derived variables.")
        logger.warning(f"All columns: {list(df.columns)}")
    
    # ==================== 1. Statistical Summary ====================
    logger.info("Generating statistical summary...")
    desc = df.describe(include="all")
    desc.to_csv(f'{output_dir}/summary.csv', index=True)
    logger.info(f"Statistical summary saved to {output_dir}/summary.csv")
    
    # ==================== 2. Distribution Plots ====================
    logger.info("Generating distribution plots...")
    plot_distributions(df, num_cols, target, output_dir)
    
    # ==================== 3. Outlier Detection ====================
    if len(num_cols) > 0:
        logger.info("Detecting outliers using Z-score method...")
        outlier_mask_zscore = zscore_outliers(df, num_cols, z=4.0)
        outlier_mask_zscore.to_csv(f'{output_dir}/outliers_zscore.csv', index=False)
        outlier_counts_zscore = outlier_mask_zscore.sum()
        outlier_counts_zscore.to_csv(f'{output_dir}/outlier_counts_zscore.csv')
        logger.info(f"Z-score outliers detected. Counts:\n{outlier_counts_zscore}")
        
        logger.info("Detecting outliers using IQR method...")
        outlier_mask_iqr = iqr_outliers(df, num_cols)
        outlier_mask_iqr.to_csv(f'{output_dir}/outliers_iqr.csv', index=False)
        outlier_counts_iqr = outlier_mask_iqr.sum()
        outlier_counts_iqr.to_csv(f'{output_dir}/outlier_counts_iqr.csv')
        logger.info(f"IQR outliers detected. Counts:\n{outlier_counts_iqr}")
    else:
        logger.warning("Skipping outlier detection: No numeric columns found.")
    
    # ==================== 4. Correlation Analysis ====================
    if len(num_cols) > 0:
        logger.info("Computing Pearson correlation matrix...")
        # Replace inf with NaN before correlation
        corr_data = df[num_cols + [target]].replace([np.inf, -np.inf], np.nan)
        corr = corr_data.corr(numeric_only=True)
        corr.to_csv(f'{output_dir}/correlation_pearson.csv')
        
        # Correlation heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5)
        plt.title('Pearson Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_pearson_heatmap.png', 
                dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Pearson correlation matrix saved to {output_dir}/correlation_pearson.csv")
    else:
        logger.warning("Skipping correlation analysis: No numeric columns found.")
    
    # ==================== 5. Mutual Information ====================
    if len(num_cols) > 0:
        logger.info("Computing mutual information matrix...")
        # Replace inf with NaN, then fill missing values for MI calculation
        df_clean = df[num_cols].replace([np.inf, -np.inf], np.nan)
        df_filled = df_clean.fillna(df_clean.median())
        # Remove any remaining NaN (if median is NaN for a column)
        df_filled = df_filled.fillna(0)
        mi_scores = mutual_info_classif(df_filled, df[target], random_state=42)
        
        mi_df = pd.DataFrame({
            'feature': num_cols,
            'mutual_information': mi_scores
        }).sort_values('mutual_information', ascending=False)
        
        mi_df.to_csv(f'{output_dir}/mutual_information.csv', index=False)
        
        # Plot mutual information
        plt.figure(figsize=(12, 6))
        sns.barplot(data=mi_df, x='feature', y='mutual_information', palette='viridis')
        plt.title('Mutual Information with Target Variable', fontsize=14, fontweight='bold')
        plt.xlabel('Features')
        plt.ylabel('Mutual Information')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/mutual_information_plot.png', 
                dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Mutual information saved to {output_dir}/mutual_information.csv")
    else:
        logger.warning("Skipping mutual information calculation: No numeric columns found.")
    
    # ==================== 6. Causal Discovery ====================
    logger.info("Performing causal discovery...")
    
    # Variables for causal DAG: age, cholesterol, ap_hi, ap_lo
    causal_vars = ['age', 'cholesterol', 'ap_hi', 'ap_lo']
    causal_vars = [v for v in causal_vars if v in df.columns]
    
    if len(causal_vars) >= 2:
        # PC Algorithm
        logger.info("Running PC Algorithm for causal discovery...")
        pc_result = causal_discovery_pc(df, causal_vars, output_dir)
        if pc_result:
            logger.info(f"PC Algorithm result: {pc_result.get('method', 'N/A')}")
        
        # NOTEARS Algorithm
        logger.info("Running NOTEARS algorithm for causal discovery...")
        notears_result = causal_discovery_notears(df, causal_vars, output_dir)
        if notears_result:
            logger.info(f"NOTEARS result: Found {len(notears_result.get('edges', []))} edges")
        
        # Save causal discovery summary
        causal_summary = {
            'variables': causal_vars,
            'pc_algorithm': pc_result,
            'notears': notears_result
        }
        import json
        with open(f'{output_dir}/causal_discovery_summary.json', 'w') as f:
            json.dump(causal_summary, f, indent=2, default=str)
        
        logger.info("Causal discovery completed")
    else:
        logger.warning(f"Not enough variables for causal discovery. Required: age, cholesterol, ap_hi, ap_lo")
    
    # ==================== 7. Bias and Drift Detection ====================
    if len(num_cols) > 0:
        logger.info("Performing bias and drift detection...")
        adversarial_result = adversarial_validation(df, num_cols, test_size=0.2, output_dir=output_dir)
        if adversarial_result.get('auc_score') is not None:
            logger.info(f"Adversarial validation AUC: {adversarial_result['auc_score']:.4f}")
            logger.info(f"Distribution shift detected: {adversarial_result.get('shift_detected', 'N/A')}")
    else:
        logger.warning("Skipping bias and drift detection: No numeric columns found.")
    
    # ==================== 8. Age & Cardio Relationship ====================
    logger.info("Plotting age & cardio relationship...")
    plot_age_cardio_relationship(df, output_dir)
    
    # ==================== 9. BMI vs Cardio Status ====================
    logger.info("Plotting BMI vs cardio status...")
    plot_bmi_cardio_scatter(df, output_dir)
    
    # ==================== 10. Cholesterol & Glucose Levels ====================
    logger.info("Plotting cholesterol & glucose levels...")
    plot_cholesterol_glucose_levels(df, output_dir)
    
    # ==================== 11. Lifestyle Habits ====================
    logger.info("Plotting lifestyle habits...")
    plot_lifestyle_habits(df, output_dir)
    
    # ==================== 12. Key Insights & Recommendations ====================
    logger.info("Generating key insights and recommendations...")
    generate_key_insights(df, output_dir)
    
    # ==================== 13. Data Quality Recommendations ====================
    logger.info("Generating comprehensive data quality recommendations...")
    recommendations = generate_data_recommendations(df, output_dir)
    
    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info("DATA QUALITY RECOMMENDATIONS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Code-fixed errors: {len(recommendations['code_fixed_errors'])}")
    logger.info(f"⚠️  Manual data fixes required: {len(recommendations['manual_data_fixes'])}")
    logger.info(f"📊 Outlier handling recommendations: {len(recommendations['outlier_handling'])}")
    logger.info(f"💡 Other recommendations: {len(recommendations['other_recommendations'])}")
    logger.info(f"{'='*60}\n")
    
    logger.info(f"All EDA artifacts saved to {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Understanding: EDA, Correlation Analysis, and Causal Discovery"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    args = parser.parse_args()
    main(args)
