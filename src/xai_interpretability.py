"""
XAI Interpretability Module - Lifestyle Analysis

This module provides functions for lifestyle feature analysis, visualization,
and reporting as part of the L2 Interpretative Layer.

Functions:
    - analyze_lifestyle_importance: Calculate lifestyle feature importance statistics
    - create_lifestyle_visualizations: Generate lifestyle pattern visualizations
    - generate_lifestyle_detail_table: Create detailed lifestyle table for test samples
    - calculate_lifestyle_statistics: Compute lifestyle distribution and category statistics
    - analyze_lifestyle_patterns: Main function to run all lifestyle analyses
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_lifestyle_data(
    test_rf_path: str = "artifacts/Interpretation/test_prediction_result/test_predictions_rf_with_lifestyle.csv",
    test_tab_path: str = "artifacts/Interpretation/test_prediction_result/test_predictions_tabnet_with_lifestyle.csv",
    rf_imp_path: str = "artifacts/Interpretation/importance/rf/feature_importance_summary.csv",
    tabnet_imp_path: str = "artifacts/Interpretation/importance/tabnet/feature_importance_summary.csv",
    comp_path: str = "artifacts/Interpretation/importance/comparison_tabnet_rf.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all necessary data files for lifestyle analysis.
    
    Returns:
        Tuple of (test_rf, test_tab, rf_imp, tabnet_imp, comp) DataFrames
    """
    logger.info("Loading lifestyle analysis data...")
    
    test_rf = pd.read_csv(test_rf_path)
    test_tab = pd.read_csv(test_tab_path)
    rf_imp = pd.read_csv(rf_imp_path)
    tabnet_imp = pd.read_csv(tabnet_imp_path)
    comp = pd.read_csv(comp_path)
    
    logger.info(f"Loaded {len(test_rf)} RF test samples, {len(test_tab)} TabNet test samples")
    return test_rf, test_tab, rf_imp, tabnet_imp, comp


def analyze_lifestyle_importance(
    rf_imp: pd.DataFrame,
    tabnet_imp: pd.DataFrame,
    lifestyle_features: list = None
) -> Dict:
    """
    Analyze lifestyle feature importance for both RF and TabNet.
    
    Args:
        rf_imp: Random Forest importance DataFrame
        tabnet_imp: TabNet attention DataFrame
        lifestyle_features: List of lifestyle feature names
        
    Returns:
        Dictionary with lifestyle importance statistics
    """
    if lifestyle_features is None:
        lifestyle_features = ['smoke', 'alco', 'active', 'lifestyle_risk']
    
    logger.info("Analyzing lifestyle feature importance...")
    
    # Filter lifestyle features
    rf_lifestyle = rf_imp[rf_imp['feature'].isin(lifestyle_features)].sort_values(
        'mean_importance', ascending=False
    )
    tabnet_lifestyle = tabnet_imp[tabnet_imp['feature'].isin(lifestyle_features)].sort_values(
        'mean_attention', ascending=False
    )
    
    # Calculate total importance
    rf_total_imp = rf_imp['mean_importance'].sum()
    tabnet_total_attn = tabnet_imp['mean_attention'].sum()
    rf_lifestyle_total = rf_lifestyle['mean_importance'].sum()
    tabnet_lifestyle_total = tabnet_lifestyle['mean_attention'].sum()
    
    # Calculate percentages
    rf_lifestyle_pct = (rf_lifestyle_total / rf_total_imp) * 100
    tabnet_lifestyle_pct = (tabnet_lifestyle_total / tabnet_total_attn) * 100
    
    # Calculate ratios
    ratios = {}
    for feature in lifestyle_features:
        rf_val = rf_lifestyle[rf_lifestyle['feature'] == feature]['mean_importance'].values
        tabnet_val = tabnet_lifestyle[tabnet_lifestyle['feature'] == feature]['mean_attention'].values
        
        if len(rf_val) > 0 and len(tabnet_val) > 0 and rf_val[0] > 0:
            ratios[feature] = tabnet_val[0] / rf_val[0]
        else:
            ratios[feature] = 0.0
    
    results = {
        'rf_lifestyle': rf_lifestyle,
        'tabnet_lifestyle': tabnet_lifestyle,
        'rf_contribution_pct': rf_lifestyle_pct,
        'tabnet_contribution_pct': tabnet_lifestyle_pct,
        'ratios': ratios,
        'rf_total': rf_lifestyle_total,
        'tabnet_total': tabnet_lifestyle_total
    }
    
    logger.info(f"RF lifestyle contribution: {rf_lifestyle_pct:.2f}%")
    logger.info(f"TabNet lifestyle contribution: {tabnet_lifestyle_pct:.2f}%")
    
    return results


def create_lifestyle_visualizations(
    test_rf: pd.DataFrame,
    rf_imp: pd.DataFrame,
    tabnet_imp: pd.DataFrame,
    output_dir: str = "artifacts/Interpretation/lifestyle/lifestyle_plots"
) -> Dict[str, str]:
    """
    Create all lifestyle pattern visualizations.
    
    Args:
        test_rf: Test predictions with lifestyle data
        rf_imp: Random Forest importance DataFrame
        tabnet_imp: TabNet attention DataFrame
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    logger.info("Creating lifestyle visualizations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    lifestyle_features = ['smoke', 'alco', 'active', 'lifestyle_risk']
    test_rf_complete = test_rf[test_rf[lifestyle_features].notna().all(axis=1)].copy()
    
    # Create lifestyle category
    test_rf_complete['lifestyle_category'] = test_rf_complete['lifestyle_risk'].apply(
        lambda x: 'Very Good' if x < -0.25 else (
            'Good' if x < 0 else ('Medium' if x <= 0.5 else 'Bad')
        )
    )
    
    plot_paths = {}
    
    # 1. Lifestyle Importance Comparison (RF vs TabNet)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    lifestyle_features_ordered = ['smoke', 'alco', 'active', 'lifestyle_risk']
    rf_lifestyle = rf_imp[rf_imp['feature'].isin(lifestyle_features_ordered)]
    tabnet_lifestyle = tabnet_imp[tabnet_imp['feature'].isin(lifestyle_features_ordered)]
    
    # RF
    rf_data = [
        rf_lifestyle[rf_lifestyle['feature'] == f]['mean_importance'].values[0]
        if len(rf_lifestyle[rf_lifestyle['feature'] == f]) > 0 else 0
        for f in lifestyle_features_ordered
    ]
    axes[0].barh(lifestyle_features_ordered, rf_data, color='#2ecc71')
    axes[0].set_xlabel('Mean Importance', fontsize=12)
    axes[0].set_title('Random Forest Lifestyle Importance', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # TabNet
    tabnet_data = [
        tabnet_lifestyle[tabnet_lifestyle['feature'] == f]['mean_attention'].values[0]
        if len(tabnet_lifestyle[tabnet_lifestyle['feature'] == f]) > 0 else 0
        for f in lifestyle_features_ordered
    ]
    axes[1].barh(lifestyle_features_ordered, tabnet_data, color='#3498db')
    axes[1].set_xlabel('Mean Attention', fontsize=12)
    axes[1].set_title('TabNet Lifestyle Attention', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / 'lifestyle_importance_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths['importance_comparison'] = str(plot_path)
    logger.info(f"Created: {plot_path}")
    
    # 2. Lifestyle Risk vs Prediction Probability
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'Very Good': '#27ae60', 'Good': '#f39c12', 'Medium': '#e74c3c', 'Bad': '#8e44ad'}
    
    for category in test_rf_complete['lifestyle_category'].unique():
        data = test_rf_complete[test_rf_complete['lifestyle_category'] == category]
        ax.scatter(
            data['lifestyle_risk'], data['y_pred_prob'],
            label=category, color=colors.get(category, '#95a5a6'),
            s=100, alpha=0.7
        )
    
    ax.set_xlabel('Lifestyle Risk Score', fontsize=12)
    ax.set_ylabel('Prediction Probability', fontsize=12)
    ax.set_title('Lifestyle Risk vs Prediction Probability', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = output_path / 'lifestyle_risk_vs_prediction.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths['risk_vs_prediction'] = str(plot_path)
    logger.info(f"Created: {plot_path}")
    
    # 3. Lifestyle Features Distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(['smoke', 'alco', 'active']):
        counts = test_rf_complete[feature].value_counts().sort_index()
        axes[idx].bar(counts.index.astype(str), counts.values, color=['#3498db', '#e74c3c'])
        axes[idx].set_title(f'{feature.capitalize()} Distribution', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Value', fontsize=10)
        axes[idx].set_ylabel('Count', fontsize=10)
        axes[idx].set_xticks([0, 1])
        axes[idx].set_xticklabels(['No', 'Yes'])
    
    # Lifestyle risk distribution
    axes[3].hist(
        test_rf_complete['lifestyle_risk'], bins=10,
        color='#9b59b6', alpha=0.7, edgecolor='black'
    )
    axes[3].axvline(
        test_rf_complete['lifestyle_risk'].mean(),
        color='red', linestyle='--',
        label=f'Mean: {test_rf_complete["lifestyle_risk"].mean():.2f}'
    )
    axes[3].set_title('Lifestyle Risk Score Distribution', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Lifestyle Risk Score', fontsize=10)
    axes[3].set_ylabel('Frequency', fontsize=10)
    axes[3].legend()
    axes[3].grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / 'lifestyle_features_distribution.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths['features_distribution'] = str(plot_path)
    logger.info(f"Created: {plot_path}")
    
    # 4. Lifestyle Category Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    category_stats = test_rf_complete.groupby('lifestyle_category').agg({
        'y_pred_prob': 'mean',
        'y_true': 'mean',
        'correct': 'mean'
    }).reset_index()
    
    # Mean prediction probability by category
    colors_list = ['#27ae60', '#f39c12', '#e74c3c', '#8e44ad']
    axes[0].bar(
        category_stats['lifestyle_category'],
        category_stats['y_pred_prob'],
        color=colors_list[:len(category_stats)]
    )
    axes[0].set_ylabel('Mean Prediction Probability', fontsize=12)
    axes[0].set_title('Mean Prediction Probability by Lifestyle Category', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Accuracy by category
    axes[1].bar(
        category_stats['lifestyle_category'],
        category_stats['correct'],
        color=colors_list[:len(category_stats)]
    )
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Prediction Accuracy by Lifestyle Category', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / 'lifestyle_category_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths['category_analysis'] = str(plot_path)
    logger.info(f"Created: {plot_path}")
    
    logger.info(f"All visualizations saved to: {output_dir}")
    return plot_paths


def generate_lifestyle_detail_table(
    test_rf: pd.DataFrame,
    output_path: str = "artifacts/Interpretation/test_prediction_result/lifestyle_detail_table.csv"
) -> pd.DataFrame:
    """
    Generate detailed lifestyle table for test samples.
    
    Args:
        test_rf: Test predictions with lifestyle data
        output_path: Path to save the CSV file
        
    Returns:
        DataFrame with lifestyle details
    """
    logger.info("Generating lifestyle detail table...")
    
    lifestyle_features = ['smoke', 'alco', 'active', 'lifestyle_risk']
    test_rf_complete = test_rf[test_rf[lifestyle_features].notna().all(axis=1)].copy()
    
    # Create lifestyle category
    test_rf_complete['lifestyle_category'] = test_rf_complete['lifestyle_risk'].apply(
        lambda x: 'Very Good' if x < -0.25 else (
            'Good' if x < 0 else ('Medium' if x <= 0.5 else 'Bad')
        )
    )
    
    # Select and sort columns
    lifestyle_detail = test_rf_complete[[
        'sample_id', 'y_true', 'y_pred_prob', 'y_pred_binary', 'correct',
        'smoke', 'alco', 'active', 'lifestyle_risk', 'lifestyle_category'
    ]].copy()
    
    lifestyle_detail = lifestyle_detail.sort_values('y_pred_prob', ascending=False)
    
    # Save to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lifestyle_detail.to_csv(output_file, index=False)
    
    logger.info(f"Saved lifestyle detail table to: {output_path}")
    logger.info(f"Total samples with complete lifestyle data: {len(lifestyle_detail)}/{len(test_rf)}")
    
    return lifestyle_detail


def calculate_lifestyle_statistics(
    test_rf: pd.DataFrame,
    lifestyle_features: list = None
) -> Dict:
    """
    Calculate lifestyle distribution and category statistics.
    
    Args:
        test_rf: Test predictions with lifestyle data
        lifestyle_features: List of lifestyle feature names
        
    Returns:
        Dictionary with lifestyle statistics
    """
    if lifestyle_features is None:
        lifestyle_features = ['smoke', 'alco', 'active', 'lifestyle_risk']
    
    logger.info("Calculating lifestyle statistics...")
    
    test_rf_complete = test_rf[test_rf[lifestyle_features].notna().all(axis=1)].copy()
    
    # Create lifestyle category
    test_rf_complete['lifestyle_category'] = test_rf_complete['lifestyle_risk'].apply(
        lambda x: 'Very Good' if x < -0.25 else (
            'Good' if x < 0 else ('Medium' if x <= 0.5 else 'Bad')
        )
    )
    
    # Distribution statistics
    distribution = {}
    for feature in ['smoke', 'alco', 'active']:
        counts = test_rf_complete[feature].value_counts().sort_index()
        distribution[feature] = {
            '0': int(counts.get(0, 0)),
            '1': int(counts.get(1, 0))
        }
    
    # Lifestyle risk statistics
    lifestyle_risk_stats = {
        'min': float(test_rf_complete['lifestyle_risk'].min()),
        'max': float(test_rf_complete['lifestyle_risk'].max()),
        'mean': float(test_rf_complete['lifestyle_risk'].mean()),
        'std': float(test_rf_complete['lifestyle_risk'].std()),
        'good_count': int((test_rf_complete['lifestyle_risk'] < 0).sum()),
        'medium_count': int(((test_rf_complete['lifestyle_risk'] >= 0) & 
                            (test_rf_complete['lifestyle_risk'] <= 0.5)).sum()),
        'bad_count': int((test_rf_complete['lifestyle_risk'] > 0.5).sum())
    }
    
    # Category statistics
    category_stats = test_rf_complete.groupby('lifestyle_category').agg({
        'y_pred_prob': ['mean', 'std', 'count'],
        'y_true': 'mean',
        'correct': 'mean'
    }).reset_index()
    category_stats.columns = [
        'Category', 'Mean_Prob', 'Std_Prob', 'Count',
        'True_Pos_Rate', 'Accuracy'
    ]
    
    results = {
        'distribution': distribution,
        'lifestyle_risk': lifestyle_risk_stats,
        'category_stats': category_stats,
        'total_complete': len(test_rf_complete),
        'total_samples': len(test_rf)
    }
    
    logger.info(f"Lifestyle statistics calculated for {len(test_rf_complete)}/{len(test_rf)} samples")
    
    return results


def analyze_lifestyle_patterns(
    output_dir: str = "artifacts/Interpretation/lifestyle/lifestyle_plots",
    detail_table_path: str = "artifacts/Interpretation/test_prediction_result/lifestyle_detail_table.csv"
) -> Dict:
    """
    Main function to run all lifestyle analyses.
    
    Args:
        output_dir: Directory to save visualizations
        detail_table_path: Path to save lifestyle detail table
        
    Returns:
        Dictionary with all analysis results
    """
    logger.info("=" * 60)
    logger.info("Starting Lifestyle Pattern Analysis")
    logger.info("=" * 60)
    
    # Load data
    test_rf, test_tab, rf_imp, tabnet_imp, comp = load_lifestyle_data()
    
    # Analyze importance
    importance_results = analyze_lifestyle_importance(rf_imp, tabnet_imp)
    
    # Create visualizations
    plot_paths = create_lifestyle_visualizations(
        test_rf, rf_imp, tabnet_imp, output_dir
    )
    
    # Generate detail table
    lifestyle_detail = generate_lifestyle_detail_table(test_rf, detail_table_path)
    
    # Calculate statistics
    statistics = calculate_lifestyle_statistics(test_rf)
    
    # Compile results
    results = {
        'importance': importance_results,
        'visualizations': plot_paths,
        'detail_table': lifestyle_detail,
        'statistics': statistics,
        'detail_table_path': detail_table_path
    }
    
    logger.info("=" * 60)
    logger.info("Lifestyle Pattern Analysis Complete")
    logger.info("=" * 60)
    logger.info(f"Visualizations: {len(plot_paths)} plots created")
    logger.info(f"Detail table: {len(lifestyle_detail)} samples")
    logger.info(f"RF lifestyle contribution: {importance_results['rf_contribution_pct']:.2f}%")
    logger.info(f"TabNet lifestyle contribution: {importance_results['tabnet_contribution_pct']:.2f}%")
    
    return results


# ==================== Additional L2 Charts Functions ====================

def _load_summary(csv_path: Path) -> pd.DataFrame:
    """Load feature importance summary CSV."""
    df = pd.read_csv(csv_path)
    if "feature" not in df.columns or "scope" not in df.columns:
        raise ValueError(f"Unexpected schema in {csv_path}")
    return df


def _value_column(df: pd.DataFrame) -> str:
    """Determine value column name (mean_importance or mean_attention)."""
    if "mean_importance" in df.columns:
        return "mean_importance"
    if "mean_attention" in df.columns:
        return "mean_attention"
    raise ValueError("DataFrame missing expected value column (mean_importance/mean_attention).")


def plot_global_top(
    df: pd.DataFrame,
    out_path: Path,
    top_n: int = 10,
) -> None:
    """Plot top N global feature importances."""
    global_df = df[df["scope"] == "global"].copy()
    value_col = _value_column(global_df)
    global_df = global_df.sort_values(value_col, ascending=False).head(top_n)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=global_df,
        y="feature",
        x=value_col,
        hue="feature",
        dodge=False,
        legend=False,
        palette="viridis",
    )
    label = "Mean Importance" if value_col == "mean_importance" else "Mean Attention"
    plt.xlabel(label)
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Global Feature Importances")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_fold_heatmap(df: pd.DataFrame, out_path: Path, top_n: int = 12) -> None:
    """Plot feature importance heatmap across CV folds."""
    fold_df = df[df["scope"] == "fold"].copy()
    if fold_df.empty:
        return
    value_col = _value_column(fold_df)
    pivot = (
        fold_df.pivot_table(
            index="feature",
            columns="group",
            values=value_col,
            aggfunc="mean",
        )
        .fillna(0.0)
        .sort_values("fold_0", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".3f")
    plt.xlabel("Fold")
    plt.ylabel("Feature")
    plt.title("Feature Importance per Fold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_group(
    df: pd.DataFrame,
    features: list,
    out_path: Path,
    title: str,
) -> None:
    """Plot importance for a specific feature group."""
    global_df = df[df["scope"] == "global"].copy()
    value_col = _value_column(global_df)
    subset = global_df[global_df["feature"].isin(features)].copy()
    if subset.empty:
        return
    subset = subset.sort_values(value_col, ascending=False)

    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=subset,
        x="feature",
        y=value_col,
        hue="feature",
        dodge=False,
        legend=False,
        palette="crest",
    )
    label = "Mean Importance" if value_col == "mean_importance" else "Mean Attention"
    plt.ylabel(label)
    plt.xlabel("")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_comparison_heatmap(comparison_csv: Path, out_path: Path) -> None:
    """Plot comparison heatmap between TabNet and RF."""
    df = pd.read_csv(comparison_csv)
    if not {"feature", "tabnet_mean_attention", "rf_mean_importance"}.issubset(df.columns):
        raise ValueError("Comparison CSV missing required columns for heatmap.")

    pivot = df.set_index("feature")[["tabnet_mean_attention", "rf_mean_importance"]]
    pivot = pivot.rename(
        columns={
            "tabnet_mean_attention": "TabNet Attention",
            "rf_mean_importance": "RF Importance",
        }
    )
    pivot = pivot.sort_values("TabNet Attention", ascending=False)

    plt.figure(figsize=(8, max(6, len(pivot) * 0.3)))
    sns.heatmap(pivot, cmap="rocket_r", annot=True, fmt=".3f")
    plt.title("TabNet vs RandomForest — Global Feature Signals")
    plt.xlabel("Model Metric")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def build_importance_dashboard(
    csv_path: Path,
    out_dir: Path,
    top_n: int = 10,
    model_label: str = "rf",
) -> Dict[str, str]:
    """
    Build feature importance dashboard for a model.
    
    Returns:
        Dictionary mapping plot names to file paths
    """
    df = _load_summary(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plot_paths = {}
    
    # Global top N
    global_path = out_dir / f"{model_label}_global_top{top_n}.png"
    plot_global_top(df, global_path, top_n)
    plot_paths['global_top'] = str(global_path)
    
    # Fold heatmap
    fold_path = out_dir / f"{model_label}_fold_heatmap.png"
    plot_fold_heatmap(df, fold_path)
    if fold_path.exists():
        plot_paths['fold_heatmap'] = str(fold_path)
    
    # Blood pressure features
    bp_features = ["ap_hi", "ap_lo", "pulse_pressure"]
    bp_path = out_dir / f"{model_label}_blood_pressure.png"
    plot_feature_group(df, bp_features, bp_path, "Blood Pressure Signals")
    if bp_path.exists():
        plot_paths['blood_pressure'] = str(bp_path)
    
    # Demographic features
    demographic_features = ["age", "age_years", "weight", "height", "bmi", "cholesterol"]
    demo_path = out_dir / f"{model_label}_demographic.png"
    plot_feature_group(
        df,
        demographic_features,
        demo_path,
        "Demographic / Anthropometric Importance",
    )
    if demo_path.exists():
        plot_paths['demographic'] = str(demo_path)
    
    # Lifestyle features
    lifestyle_features = ["smoke", "alco", "active"]
    lifestyle_path = out_dir / f"{model_label}_lifestyle.png"
    plot_feature_group(
        df,
        lifestyle_features,
        lifestyle_path,
        "Lifestyle Indicators (Low importance)"
    )
    if lifestyle_path.exists():
        plot_paths['lifestyle'] = str(lifestyle_path)
    
    return plot_paths


def _plot_reliability(ax, bins: pd.DataFrame, label: str, color: str) -> None:
    """Helper function to plot reliability curve."""
    ax.plot(bins["bin_confidence"], bins["bin_accuracy"], marker="o", label=label, color=color)
    ax.bar(
        bins["bin_confidence"],
        bins["bin_accuracy"],
        width=1.0 / len(bins),
        color=color,
        alpha=0.15,
        align="center",
    )


def save_reliability_overlay(
    model_to_df: Dict[str, pd.DataFrame],
    out_path: Path,
    n_bins: int = 15,
) -> Optional[Path]:
    """Create overlay reliability plot comparing models."""
    try:
        from src.utils.metrics import reliability_bins
    except ImportError:
        logger.warning("reliability_bins not found, skipping reliability overlay")
        return None
    
    if not model_to_df:
        logger.warning("No models supplied for reliability overlay plot.")
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diagonal = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap("tab10")

    for idx, (model, df) in enumerate(sorted(model_to_df.items())):
        if df.empty:
            logger.warning("Model %s dataframe empty; skipping overlay.", model)
            continue
        bins = reliability_bins(df["y_true"].to_numpy(), df["y_pred"].to_numpy(), n_bins=n_bins)
        color = cmap(idx % 10)
        ax.plot(bins["bin_confidence"], bins["bin_accuracy"], marker="o", label=model, color=color)

    if not ax.lines:
        logger.warning("No valid model data for overlay plot.")
        plt.close(fig)
        return None

    ax.plot(diagonal, diagonal, linestyle="--", color="#444444", label="Ideal")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability Overlay")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Saved reliability overlay plot to %s", out_path)
    return out_path


def additional_l2_charts(
    rf_summary_csv: str = "artifacts/Interpretation/importance/rf/feature_importance_summary.csv",
    tabnet_summary_csv: str = "artifacts/Interpretation/importance/tabnet/feature_importance_summary.csv",
    comparison_csv: str = "artifacts/Interpretation/importance/comparison_tabnet_rf.csv",
    rf_oof_csv: str = "artifacts/Model/rf/oof_predictions.csv",
    tabnet_oof_csv: str = "artifacts/Model/tabnet/oof_predictions.csv",
    rf_calibrated_csv: Optional[str] = None,
    tabnet_calibrated_csv: Optional[str] = None,
    output_dir: str = "artifacts/Interpretation/reports",
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Generate all additional L2 charts (feature importance and reliability plots).
    
    This function creates:
    1. RF feature importance plots (global top, fold heatmap, BP, demographic, lifestyle)
    2. TabNet feature importance plots (global top, BP, demographic, lifestyle)
    3. TabNet vs RF comparison heatmap
    4. Reliability overlay plots (pre and post calibration)
    
    Args:
        rf_summary_csv: Path to RF feature importance summary CSV
        tabnet_summary_csv: Path to TabNet feature importance summary CSV
        comparison_csv: Path to comparison CSV (TabNet vs RF)
        rf_oof_csv: Path to RF OOF predictions CSV
        tabnet_oof_csv: Path to TabNet OOF predictions CSV
        rf_calibrated_csv: Optional path to RF calibrated predictions CSV
        tabnet_calibrated_csv: Optional path to TabNet calibrated predictions CSV
        output_dir: Base output directory for all plots (creates subdirectories: importance_plots/, plots/)
        top_n: Number of top features to show in global charts
        
    Returns:
        Dictionary with plot paths and summary statistics
    """
    logger.info("=" * 60)
    logger.info("Generating Additional L2 Charts")
    logger.info("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    importance_dir = output_path / "importance_plots"
    plots_dir = output_path / "plots"
    importance_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'rf_plots': {},
        'tabnet_plots': {},
        'comparison_plots': {},
        'reliability_plots': {}
    }
    
    # 1. RF Feature Importance Dashboard
    logger.info("Creating RF feature importance plots...")
    rf_dir = importance_dir / "rf"
    rf_dir.mkdir(parents=True, exist_ok=True)
    
    if Path(rf_summary_csv).exists():
        results['rf_plots'] = build_importance_dashboard(
            Path(rf_summary_csv),
            rf_dir,
            top_n=top_n,
            model_label="rf"
        )
        logger.info(f"Created {len(results['rf_plots'])} RF plots")
    else:
        logger.warning(f"RF summary CSV not found: {rf_summary_csv}")
    
    # 2. TabNet Feature Importance Dashboard
    logger.info("Creating TabNet feature importance plots...")
    tabnet_dir = importance_dir / "tabnet"
    tabnet_dir.mkdir(parents=True, exist_ok=True)
    
    if Path(tabnet_summary_csv).exists():
        results['tabnet_plots'] = build_importance_dashboard(
            Path(tabnet_summary_csv),
            tabnet_dir,
            top_n=top_n,
            model_label="tabnet"
        )
        logger.info(f"Created {len(results['tabnet_plots'])} TabNet plots")
    else:
        logger.warning(f"TabNet summary CSV not found: {tabnet_summary_csv}")
    
    # 3. Comparison Heatmap
    logger.info("Creating comparison heatmap...")
    if Path(comparison_csv).exists():
        comparison_path = tabnet_dir / "tabnet_rf_heatmap.png"
        plot_comparison_heatmap(Path(comparison_csv), comparison_path)
        results['comparison_plots']['heatmap'] = str(comparison_path)
        logger.info(f"Created comparison heatmap: {comparison_path}")
    else:
        logger.warning(f"Comparison CSV not found: {comparison_csv}")
    
    # 4. Reliability Overlay Plots
    logger.info("Creating reliability overlay plots...")
    
    model_to_df = {}
    
    # Load RF OOF predictions
    if Path(rf_oof_csv).exists():
        rf_oof = pd.read_csv(rf_oof_csv)
        if 'y_true' in rf_oof.columns and 'y_pred' in rf_oof.columns:
            model_to_df['RF'] = rf_oof[['y_true', 'y_pred']].copy()
            logger.info(f"Loaded RF OOF predictions: {len(rf_oof)} samples")
    else:
        logger.warning(f"RF OOF CSV not found: {rf_oof_csv}")
    
    # Load TabNet OOF predictions
    if Path(tabnet_oof_csv).exists():
        tabnet_oof = pd.read_csv(tabnet_oof_csv)
        if 'y_true' in tabnet_oof.columns and 'y_pred' in tabnet_oof.columns:
            model_to_df['TabNet'] = tabnet_oof[['y_true', 'y_pred']].copy()
            logger.info(f"Loaded TabNet OOF predictions: {len(tabnet_oof)} samples")
    else:
        logger.warning(f"TabNet OOF CSV not found: {tabnet_oof_csv}")
    
    # Create reliability overlay (pre-calibration)
    if model_to_df:
        pre_path = plots_dir / "reliability_overlay_pre.png"
        saved_path = save_reliability_overlay(model_to_df, pre_path)
        if saved_path:
            results['reliability_plots']['pre_calibration'] = str(saved_path)
    
    # Load calibrated predictions (use provided paths or try default locations)
    model_to_df_calibrated = {}
    
    # Try provided paths first
    if rf_calibrated_csv and Path(rf_calibrated_csv).exists():
        rf_cal = pd.read_csv(rf_calibrated_csv)
        if 'y_true' in rf_cal.columns and 'y_pred' in rf_cal.columns:
            model_to_df_calibrated['RF (Calibrated)'] = rf_cal[['y_true', 'y_pred']].copy()
            logger.info(f"Loaded RF calibrated predictions from: {rf_calibrated_csv}")
    else:
        # Try default location
        default_rf_cal = Path("artifacts/Interpretation/calibration/rf/oof_calibrated.csv")
        if default_rf_cal.exists():
            rf_cal = pd.read_csv(default_rf_cal)
            if 'y_true' in rf_cal.columns and 'y_pred' in rf_cal.columns:
                model_to_df_calibrated['RF (Calibrated)'] = rf_cal[['y_true', 'y_pred']].copy()
                logger.info(f"Loaded RF calibrated predictions from default location: {default_rf_cal}")
    
    if tabnet_calibrated_csv and Path(tabnet_calibrated_csv).exists():
        tabnet_cal = pd.read_csv(tabnet_calibrated_csv)
        if 'y_true' in tabnet_cal.columns and 'y_pred' in tabnet_cal.columns:
            model_to_df_calibrated['TabNet (Calibrated)'] = tabnet_cal[['y_true', 'y_pred']].copy()
            logger.info(f"Loaded TabNet calibrated predictions from: {tabnet_calibrated_csv}")
    else:
        # Try default location
        default_tabnet_cal = Path("artifacts/Interpretation/calibration/tabnet/oof_calibrated.csv")
        if default_tabnet_cal.exists():
            tabnet_cal = pd.read_csv(default_tabnet_cal)
            if 'y_true' in tabnet_cal.columns and 'y_pred' in tabnet_cal.columns:
                model_to_df_calibrated['TabNet (Calibrated)'] = tabnet_cal[['y_true', 'y_pred']].copy()
                logger.info(f"Loaded TabNet calibrated predictions from default location: {default_tabnet_cal}")
    
    # Create reliability overlay (post-calibration)
    if model_to_df_calibrated:
        post_path = plots_dir / "reliability_overlay_post.png"
        saved_path = save_reliability_overlay(model_to_df_calibrated, post_path)
        if saved_path:
            results['reliability_plots']['post_calibration'] = str(saved_path)
    
    logger.info("=" * 60)
    logger.info("Additional L2 Charts Generation Complete")
    logger.info("=" * 60)
    logger.info(f"RF plots: {len(results['rf_plots'])}")
    logger.info(f"TabNet plots: {len(results['tabnet_plots'])}")
    logger.info(f"Comparison plots: {len(results['comparison_plots'])}")
    logger.info(f"Reliability plots: {len(results['reliability_plots'])}")
    
    return results


if __name__ == "__main__":
    """
    Command-line interface for lifestyle analysis and additional L2 charts.
    
    Usage:
        python -m src.xai_interpretability [--lifestyle-only] [--charts-only]
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="L2 Interpretation Analysis")
    parser.add_argument(
        "--lifestyle-only",
        action="store_true",
        help="Run only lifestyle analysis"
    )
    parser.add_argument(
        "--charts-only",
        action="store_true",
        help="Run only additional L2 charts"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/Interpretation/reports",
        help="Base output directory for all charts (creates subdirectories: lifestyle_plots/, importance_plots/, plots/)"
    )
    parser.add_argument(
        "--detail-table",
        type=str,
        default="artifacts/Interpretation/test_prediction_result/lifestyle_detail_table.csv",
        help="Path to save lifestyle detail table"
    )
    
    args = parser.parse_args()
    
    # Run lifestyle analysis
    if not args.charts_only:
        print("\n" + "=" * 60)
        print("Running Lifestyle Analysis")
        print("=" * 60)
        # Lifestyle plots go to {output_dir}/lifestyle_plots/
        lifestyle_output_dir = str(Path(args.output_dir) / "lifestyle_plots")
        lifestyle_results = analyze_lifestyle_patterns(
            output_dir=lifestyle_output_dir,
            detail_table_path=args.detail_table
        )
        
        print("\nLifestyle Analysis Summary:")
        print(f"  Visualizations: {len(lifestyle_results['visualizations'])}")
        for name, path in lifestyle_results['visualizations'].items():
            print(f"    - {name}: {path}")
        print(f"  Detail table: {lifestyle_results['detail_table_path']}")
        print(f"  RF lifestyle contribution: {lifestyle_results['importance']['rf_contribution_pct']:.2f}%")
        print(f"  TabNet lifestyle contribution: {lifestyle_results['importance']['tabnet_contribution_pct']:.2f}%")
    
    # Run additional L2 charts
    if not args.lifestyle_only:
        print("\n" + "=" * 60)
        print("Generating Additional L2 Charts")
        print("=" * 60)
        charts_results = additional_l2_charts(output_dir=args.output_dir)
        
        print("\nAdditional L2 Charts Summary:")
        print(f"  RF plots: {len(charts_results['rf_plots'])}")
        for name, path in charts_results['rf_plots'].items():
            print(f"    - {name}: {path}")
        print(f"  TabNet plots: {len(charts_results['tabnet_plots'])}")
        for name, path in charts_results['tabnet_plots'].items():
            print(f"    - {name}: {path}")
        print(f"  Comparison plots: {len(charts_results['comparison_plots'])}")
        for name, path in charts_results['comparison_plots'].items():
            print(f"    - {name}: {path}")
        print(f"  Reliability plots: {len(charts_results['reliability_plots'])}")
        for name, path in charts_results['reliability_plots'].items():
            print(f"    - {name}: {path}")

