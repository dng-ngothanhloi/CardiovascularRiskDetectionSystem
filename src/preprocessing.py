"""
Preprocessing Module
Implements comprehensive data cleaning, outlier handling, imputation, and feature engineering.
Based on EDA recommendations from data_understanding.py
"""
from __future__ import annotations
import argparse
import warnings
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
import sys
from pathlib import Path
# Add parent directory to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from .utils import load_config, get_logger, ensure_dirs, set_seeds
except ImportError:
    from src.utils import load_config, get_logger, ensure_dirs, set_seeds

# Constants
ORDINAL_COLS = ["cholesterol", "gluc"]
BINARY_COLS = ["smoke", "alco", "active", "gender"]

logger = get_logger()


def load_eda_outputs(eda_dir: str = "artifacts/eda") -> Dict[str, Any]:
    """
    Load EDA output files for use in preprocessing.
    
    Based on EDA_Output_Explanation.md, loads:
    - flagged_records.csv: Records with data quality issues
    - outliers_iqr.csv: IQR outlier masks
    - outliers_zscore.csv: Z-score outlier masks
    - correlation_pearson.csv: Correlation matrix for feature selection
    - mutual_information.csv: MI scores for feature selection
    - data_recommendations.json: Preprocessing recommendations
    
    Args:
        eda_dir: Directory containing EDA outputs
        
    Returns:
        Dictionary containing loaded EDA outputs
    """
    eda_path = Path(eda_dir)
    outputs = {}
    
    # Load flagged records
    flagged_path = eda_path / "flagged_records.csv"
    if flagged_path.exists():
        try:
            outputs['flagged_records'] = pd.read_csv(flagged_path)
            logger.info(f"Loaded flagged_records.csv: {len(outputs['flagged_records'])} records")
        except Exception as e:
            logger.warning(f"Could not load flagged_records.csv: {e}")
            outputs['flagged_records'] = None
    else:
        logger.warning(f"flagged_records.csv not found at {flagged_path}")
        outputs['flagged_records'] = None
    
    # Load IQR outliers
    outliers_iqr_path = eda_path / "outliers_iqr.csv"
    if outliers_iqr_path.exists():
        try:
            outputs['outliers_iqr'] = pd.read_csv(outliers_iqr_path)
            logger.info(f"Loaded outliers_iqr.csv: {outputs['outliers_iqr'].shape}")
        except Exception as e:
            logger.warning(f"Could not load outliers_iqr.csv: {e}")
            outputs['outliers_iqr'] = None
    else:
        logger.warning(f"outliers_iqr.csv not found at {outliers_iqr_path}")
        outputs['outliers_iqr'] = None
    
    # Load Z-score outliers
    outliers_zscore_path = eda_path / "outliers_zscore.csv"
    if outliers_zscore_path.exists():
        try:
            outputs['outliers_zscore'] = pd.read_csv(outliers_zscore_path)
            logger.info(f"Loaded outliers_zscore.csv: {outputs['outliers_zscore'].shape}")
        except Exception as e:
            logger.warning(f"Could not load outliers_zscore.csv: {e}")
            outputs['outliers_zscore'] = None
    else:
        logger.warning(f"outliers_zscore.csv not found at {outliers_zscore_path}")
        outputs['outliers_zscore'] = None
    
    # Load correlation matrix
    corr_path = eda_path / "correlation_pearson.csv"
    if corr_path.exists():
        try:
            outputs['correlation'] = pd.read_csv(corr_path, index_col=0)
            logger.info(f"Loaded correlation_pearson.csv: {outputs['correlation'].shape}")
        except Exception as e:
            logger.warning(f"Could not load correlation_pearson.csv: {e}")
            outputs['correlation'] = None
    else:
        logger.warning(f"correlation_pearson.csv not found at {corr_path}")
        outputs['correlation'] = None
    
    # Load mutual information
    mi_path = eda_path / "mutual_information.csv"
    if mi_path.exists():
        try:
            outputs['mutual_information'] = pd.read_csv(mi_path)
            logger.info(f"Loaded mutual_information.csv: {len(outputs['mutual_information'])} features")
        except Exception as e:
            logger.warning(f"Could not load mutual_information.csv: {e}")
            outputs['mutual_information'] = None
    else:
        logger.warning(f"mutual_information.csv not found at {mi_path}")
        outputs['mutual_information'] = None
    
    # Load adversarial feature importance
    adv_path = eda_path / "adversarial_feature_importance.csv"
    if adv_path.exists():
        try:
            outputs['adversarial_importance'] = pd.read_csv(adv_path)
            logger.info(f"Loaded adversarial_feature_importance.csv: {len(outputs['adversarial_importance'])} features")
        except Exception as e:
            logger.warning(f"Could not load adversarial_feature_importance.csv: {e}")
            outputs['adversarial_importance'] = None
    else:
        logger.warning(f"adversarial_feature_importance.csv not found at {adv_path}")
        outputs['adversarial_importance'] = None
    
    # Load data recommendations
    rec_path = eda_path / "data_recommendations.json"
    if rec_path.exists():
        try:
            with open(rec_path, 'r') as f:
                outputs['recommendations'] = json.load(f)
            logger.info("Loaded data_recommendations.json")
        except Exception as e:
            logger.warning(f"Could not load data_recommendations.json: {e}")
            outputs['recommendations'] = None
    else:
        logger.warning(f"data_recommendations.json not found at {rec_path}")
        outputs['recommendations'] = None
    
    return outputs


def load_data_with_delimiter(data_path: str) -> pd.DataFrame:
    """
    Load CSV data with automatic delimiter detection.
    
    Args:
        data_path: Path to CSV file
        
    Returns:
        Loaded DataFrame
    """
    logger.info(f"Loading data from: {data_path}")
    
    # Detect delimiter
    with open(data_path, 'r') as f:
        first_line = f.readline()
        if ';' in first_line:
            delimiter = ';'
            logger.info("Detected semicolon (;) delimiter")
        else:
            delimiter = ','
            logger.info("Using comma (,) delimiter")
    
    df = pd.read_csv(data_path, delimiter=delimiter)
    logger.info(f"Data loaded: {df.shape}")
    
    return df


def clean_blood_pressure(
    df: pd.DataFrame,
    fix_swap: bool = True,
    remove_negatives: bool = True,
    cap_extreme: bool = True,
    flagged_records: Optional[pd.DataFrame] = None,
    repair_equal_with_median_pp: bool = True,
    min_sys: int = 90,
    max_sys: int = 240,
    min_dia: int = 60,
    max_dia: int = 140,
) -> pd.DataFrame:
    """
    Patch: Robust BP cleaning to prevent NaN in pulse_pressure downstream.
    Implements Plan-4 (Swap + me"dian fill for edge cases) in a safe, logged way.

    Steps
    -----
    1) Remove negatives (optional)
    2) Swap ap_hi/ap_lo when ap_lo > ap_hi (using EDA flags if provided, else direct mask)
    3) Cap extreme values into physiological ranges
    4) Edge-case repair (optional):
       - If ap_hi == ap_lo: lift ap_hi by median pulse pressure (computed on valid rows)
       - If ap_hi <= 0 or ap_lo <= 0: coerce to min bounds; if still ap_hi <= ap_lo, lift by median pp

    Notes
    -----
    - median_pp is computed from rows with valid and sane BP (ap_hi > ap_lo, ap_hi>=min_sys, ap_lo>=min_dia).
    - This function *does not* compute pulse_pressure; it only ensures inputs are consistent so that
      feature_engineering.py can compute pulse_pressure without NaNs.
    """
    df = df.copy()
    initial_len = len(df)

    # 0) Quick guards
    if "ap_hi" not in df.columns or "ap_lo" not in df.columns:
        logger.warning("ap_hi/ap_lo not found; skipping BP cleaning")
        return df

    # 1) Remove negatives
    n_removed_negatives = 0
    if remove_negatives:
        mask_valid = (df["ap_hi"] >= 0) & (df["ap_lo"] >= 0)
        removed = int((~mask_valid).sum())
        if removed:
            logger.info("Removed %d rows with negative BP values.", removed)
            df = df.loc[mask_valid].copy()
            n_removed_negatives = removed

    # 2) Swap using EDA flags if available
    swaps_by_flag = 0
    swaps_by_mask = 0
    if fix_swap and flagged_records is not None and len(flagged_records) > 0 and "flag_ap_lo_gt_ap_hi" in flagged_records.columns:
        swap_idx = flagged_records.index[flagged_records["flag_ap_lo_gt_ap_hi"] == 1]
        swap_idx = [i for i in swap_idx if i in df.index]
        if len(swap_idx) > 0:
            logger.info("Fixing %d rows (swap by EDA flags).", len(swap_idx))
            df.loc[swap_idx, ["ap_hi", "ap_lo"]] = df.loc[swap_idx, ["ap_lo", "ap_hi"]].values
            swaps_by_flag = len(swap_idx)

    # 2b) Fallback swap by direct mask
    if fix_swap:
        mask_swap = df["ap_lo"] > df["ap_hi"]
        n_swap = int(mask_swap.sum())
        if n_swap:
            logger.info("Fixing %d rows (swap by direct mask ap_lo > ap_hi).", n_swap)
            df.loc[mask_swap, ["ap_hi", "ap_lo"]] = df.loc[mask_swap, ["ap_lo", "ap_hi"]].values
            swaps_by_mask = n_swap

    # 3) Cap extreme values
    if cap_extreme:
        # systolic
        mask_hi = df["ap_hi"] > max_sys
        systolic_high = int(mask_hi.sum())
        if systolic_high:
            logger.info("Capping %d rows with ap_hi > %d.", systolic_high, max_sys)
            df.loc[mask_hi, "ap_hi"] = max_sys
        mask_lo = df["ap_hi"] < min_sys
        systolic_low = int(mask_lo.sum())
        if systolic_low:
            logger.info("Capping %d rows with ap_hi < %d.", systolic_low, min_sys)
            df.loc[mask_lo, "ap_hi"] = min_sys

        # diastolic
        mask_hi = df["ap_lo"] > max_dia
        diastolic_high = int(mask_hi.sum())
        if diastolic_high:
            logger.info("Capping %d rows with ap_lo > %d.", diastolic_high, max_dia)
            df.loc[mask_hi, "ap_lo"] = max_dia
        mask_lo = df["ap_lo"] < min_dia
        diastolic_low = int(mask_lo.sum())
        if diastolic_low:
            logger.info("Capping %d rows with ap_lo < %d.", diastolic_low, min_dia)
            df.loc[mask_lo, "ap_lo"] = min_dia

    # Compute median pulse pressure on valid rows (after cleaning)
    valid_pp_mask = (df["ap_hi"] > df["ap_lo"]) & (df["ap_hi"] >= min_sys) & (df["ap_lo"] >= min_dia)
    if valid_pp_mask.any():
        median_pp = float((df.loc[valid_pp_mask, "ap_hi"] - df.loc[valid_pp_mask, "ap_lo"]).median())
    else:
        median_pp = 40.0  # safe default
    if not np.isfinite(median_pp) or median_pp <= 0:
        median_pp = 40.0

    # 4) Edge-case repair
    if repair_equal_with_median_pp:
        # a) ap_hi == ap_lo (physiologically unlikely): lift ap_hi by median_pp
        mask_equal = df["ap_hi"] == df["ap_lo"]
        n_equal = mask_equal.sum()
        if n_equal:
            logger.info("Repairing %d rows with ap_hi == ap_lo using median_pp=%.1f.", n_equal, median_pp)
            df.loc[mask_equal, "ap_hi"] = df.loc[mask_equal, "ap_lo"] + median_pp

        # b) Non-positive after previous steps: coerce to bounds then ensure order
        mask_nonpos = (df["ap_hi"] <= 0) | (df["ap_lo"] <= 0)
        n_nonpos = mask_nonpos.sum()
        if n_nonpos:
            logger.info("Repairing %d rows with non-positive BP values.", n_nonpos)
            df.loc[mask_nonpos, "ap_hi"] = np.maximum(df.loc[mask_nonpos, "ap_hi"], min_sys)
            df.loc[mask_nonpos, "ap_lo"] = np.maximum(df.loc[mask_nonpos, "ap_lo"], min_dia)
            # enforce order if still invalid
            still_bad = (df["ap_hi"] <= df["ap_lo"]) & mask_nonpos
            if still_bad.sum():
                df.loc[still_bad, "ap_hi"] = df.loc[still_bad, "ap_lo"] + median_pp

        # c) Final safety pass: if any residual ap_hi <= ap_lo, lift by median_pp
        residual = df["ap_hi"] <= df["ap_lo"]
        residual_bad = int(residual.sum())
        if residual_bad:
            logger.info("Final safety: repairing %d rows with ap_hi <= ap_lo.", residual_bad)
            df.loc[residual, "ap_hi"] = df.loc[residual, "ap_lo"] + median_pp

    # Report
    final_len = len(df)
    logger.info(
        "BP cleaning completed. Rows: %d (from %d). Median_PP=%.1f | removed_neg=%d | swaps_flag=%d | swaps_mask=%d",
        final_len,
        initial_len,
        median_pp,
        n_removed_negatives,
        swaps_by_flag,
        swaps_by_mask,
    )
    # Optional sanity counters
    invalid_after = (df["ap_hi"] <= df["ap_lo"]).sum()
    if invalid_after:
        logger.warning(f"{invalid_after} rows still have ap_hi <= ap_lo after repair.")
    return df

def handle_outliers(df: pd.DataFrame, method: str = 'winsorize',
                   outliers_iqr: Optional[pd.DataFrame] = None,
                   outliers_zscore: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Handle outliers using winsorization or truncation.
    
    Based on EDA recommendations:
    - BMI: Winsorize at [10, 50]
    - Height: Cap at [100, 220] cm
    - Weight: Winsorize at [30, 200] kg
    - Age: Review and filter if needed
    
    Args:
        df: Input dataframe
        method: 'winsorize' or 'truncate'
        outliers_iqr: Optional DataFrame with IQR outlier masks from EDA
        outliers_zscore: Optional DataFrame with Z-score outlier masks from EDA
        
    Returns:
        Dataframe with outliers handled
    """
    logger.info(f"Handling outliers using {method} method...")
    df = df.copy()
    
    # Use EDA outlier masks if available
    use_eda_outliers = (outliers_iqr is not None and len(outliers_iqr) > 0) or \
                       (outliers_zscore is not None and len(outliers_zscore) > 0)
    
    if use_eda_outliers:
        logger.info("Using outlier masks from EDA")
        # Prefer IQR over Z-score
        outlier_mask = outliers_iqr if outliers_iqr is not None else outliers_zscore
        
        # Align indices - reset index if needed
        if len(outlier_mask) == len(df):
            # Reset indices to align
            df_reset = df.reset_index(drop=True)
            outlier_mask_reset = outlier_mask.reset_index(drop=True)
            
            # Apply outlier handling per column based on EDA masks
            for col in outlier_mask_reset.columns:
                if col in df_reset.columns and pd.api.types.is_numeric_dtype(df_reset[col]):
                    outlier_indices = outlier_mask_reset[col].fillna(False).astype(bool)
                    if outlier_indices.sum() > 0:
                        logger.info(f"Handling {outlier_indices.sum()} outliers for {col} (from EDA)")
                        # Apply winsorization/capping based on column
                        if col == 'bmi':
                            df_reset.loc[outlier_indices & (df_reset[col] < 10), col] = 10
                            df_reset.loc[outlier_indices & (df_reset[col] > 50), col] = 50
                        elif col == 'height':
                            df_reset.loc[outlier_indices & (df_reset[col] < 100), col] = 100
                            df_reset.loc[outlier_indices & (df_reset[col] > 220), col] = 220
                        elif col == 'weight':
                            df_reset.loc[outlier_indices & (df_reset[col] < 30), col] = 30
                            df_reset.loc[outlier_indices & (df_reset[col] > 200), col] = 200
            
            # Restore original index
            df_reset.index = df.index[:len(df_reset)]
            df = df_reset
    
    # BMI: Winsorize (fallback if EDA masks not used)
    if 'bmi' in df.columns:
        bmi_low = (df['bmi'] < 10).sum()
        bmi_high = (df['bmi'] > 50).sum()
        if bmi_low > 0:
            logger.info(f"Winsorizing {bmi_low} records with BMI < 10")
            df.loc[df['bmi'] < 10, 'bmi'] = 10
        if bmi_high > 0:
            logger.info(f"Winsorizing {bmi_high} records with BMI > 50")
            df.loc[df['bmi'] > 50, 'bmi'] = 50
    
    # Height: Cap
    if 'height' in df.columns:
        height_low = ((df['height'] < 100) & (df['height'] > 0)).sum()
        height_high = (df['height'] > 220).sum()
        if height_low > 0:
            logger.info(f"Capping {height_low} records with height < 100 cm")
            df.loc[(df['height'] < 100) & (df['height'] > 0), 'height'] = 100
        if height_high > 0:
            logger.info(f"Capping {height_high} records with height > 220 cm")
            df.loc[df['height'] > 220, 'height'] = 220
    
    # Weight: Winsorize
    if 'weight' in df.columns:
        weight_low = ((df['weight'] < 30) & (df['weight'] > 0)).sum()
        weight_high = (df['weight'] > 200).sum()
        if weight_low > 0:
            logger.info(f"Winsorizing {weight_low} records with weight < 30 kg")
            df.loc[(df['weight'] < 30) & (df['weight'] > 0), 'weight'] = 30
        if weight_high > 0:
            logger.info(f"Winsorizing {weight_high} records with weight > 200 kg")
            df.loc[df['weight'] > 200, 'weight'] = 200
    
    # Age: Cap at reasonable range (if age_years exists)
    if 'age_years' in df.columns:
        age_high = (df['age_years'] > 100).sum()
        if age_high > 0:
            logger.warning(f"Found {age_high} records with age > 100 years. Review recommended.")
    
    return df


def improved_imputation(df: pd.DataFrame, method: str = 'mice', 
                       use_mice_for_numeric: bool = True) -> pd.DataFrame:
    """
    Improved imputation strategy.
    
    Steps:
    1. Use MICE for complex missing patterns (if method='mice')
    2. Use KNN for simpler patterns (if method='knn')
    
    Args:
        df: Input dataframe
        method: 'mice' or 'knn'
        use_mice_for_numeric: Whether to use MICE for numeric columns
        
    Returns:
        Dataframe with imputed values
    """
    logger.info(f"Imputing missing values using {method} method...")
    df = df.copy()
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    
    if len(missing_cols) == 0:
        logger.info("No missing values found. Skipping imputation.")
        return df
    
    logger.info(f"Missing values found in {len(missing_cols)} columns:")
    for col, count in missing_cols.items():
        logger.info(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Impute numeric columns
    numeric_missing = [c for c in numeric_cols if c in missing_cols.index]
    if len(numeric_missing) > 0 and use_mice_for_numeric and method == 'mice':
        try:
            logger.info("Using MICE (IterativeImputer) for numeric columns...")
            imputer = IterativeImputer(random_state=42, max_iter=10, verbose=0)
            df[numeric_missing] = imputer.fit_transform(df[numeric_missing])
            logger.info("MICE imputation completed")
        except Exception as e:
            logger.warning(f"MICE imputation failed: {e}. Falling back to median imputation.")
            df[numeric_missing] = df[numeric_missing].fillna(df[numeric_missing].median())
    elif len(numeric_missing) > 0:
        # Use median for numeric (simple but effective)
        logger.info("Using median imputation for numeric columns...")
        df[numeric_missing] = df[numeric_missing].fillna(df[numeric_missing].median())
        df[numeric_missing] = df[numeric_missing].fillna(0)  # Fill any remaining NaN
    
    # Impute categorical columns (use mode)
    categorical_missing = [c for c in categorical_cols if c in missing_cols.index]
    if len(categorical_missing) > 0:
        logger.info("Using mode imputation for categorical columns...")
        for col in categorical_missing:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
            else:
                # If no mode, fill with first non-null value or 'unknown'
                df[col] = df[col].fillna('unknown')
    
    logger.info("Imputation completed")
    return df


def remove_highly_correlated_features(df: pd.DataFrame, target: str, 
                                     threshold: float = 0.9,
                                     correlation_matrix: Optional[pd.DataFrame] = None,
                                     mutual_information: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove highly correlated features to reduce multicollinearity.
    
    Based on EDA recommendations: remove features with correlation > threshold.
    
    Args:
        df: Input dataframe
        target: Target column name (to exclude from correlation)
        threshold: Correlation threshold (default: 0.9)
        correlation_matrix: Optional correlation matrix from EDA
        mutual_information: Optional MI scores from EDA for feature prioritization
        
    Returns:
        Tuple of (cleaned dataframe, removed features list)
    """
    logger.info(f"Removing highly correlated features (threshold: {threshold})...")
    
    # Use EDA correlation matrix if available
    if correlation_matrix is not None and len(correlation_matrix) > 0:
        logger.info("Using correlation matrix from EDA")
        corr_matrix = correlation_matrix.abs()
        # Filter to only include columns that exist in df
        numeric_cols = [c for c in corr_matrix.columns if c in df.columns and c != target]
        corr_matrix = corr_matrix.loc[numeric_cols, numeric_cols]
    else:
        # Calculate correlation matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)
        
        if len(numeric_cols) < 2:
            logger.warning("Not enough numeric columns for correlation analysis")
            return df, []
        
        corr_matrix = df[numeric_cols].corr().abs()
    
    # Find pairs with correlation > threshold
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
    
    if len(high_corr_pairs) == 0:
        logger.info("No highly correlated features found")
        return df, []
    
    logger.info(f"Found {len(high_corr_pairs)} highly correlated pairs:")
    for col1, col2, corr_val in high_corr_pairs:
        logger.info(f"  {col1} <-> {col2}: {corr_val:.3f}")
    
    # Remove one feature from each pair (prefer keeping the one with higher correlation to target)
    features_to_remove = set()
    
    # Priority: Remove derived/flag features if they correlate with original
    priority_remove = ['age_years', 'chol_high', 'gluc_high', 'bmi_group', 'age_group']
    
    for col1, col2, _ in high_corr_pairs:
        # Check priority
        if col1 in priority_remove and col2 not in priority_remove:
            features_to_remove.add(col1)
        elif col2 in priority_remove and col1 not in priority_remove:
            features_to_remove.add(col2)
        else:
            # Remove the one with lower correlation to target or lower MI score
            if mutual_information is not None and len(mutual_information) > 0:
                # Use MI scores if available
                mi1 = mutual_information[mutual_information['feature'] == col1]['mutual_information'].values
                mi2 = mutual_information[mutual_information['feature'] == col2]['mutual_information'].values
                if len(mi1) > 0 and len(mi2) > 0:
                    if mi1[0] < mi2[0]:
                        features_to_remove.add(col1)
                    else:
                        features_to_remove.add(col2)
                elif len(mi1) == 0:
                    features_to_remove.add(col1)
                elif len(mi2) == 0:
                    features_to_remove.add(col2)
            elif target in df.columns:
                # Fallback to correlation with target
                corr1_target = abs(df[[col1, target]].corr().iloc[0, 1]) if col1 != target else 0
                corr2_target = abs(df[[col2, target]].corr().iloc[0, 1]) if col2 != target else 0
                if corr1_target < corr2_target:
                    features_to_remove.add(col1)
                else:
                    features_to_remove.add(col2)
            else:
                # If no target, remove the second one
                features_to_remove.add(col2)
    
    features_to_remove = list(features_to_remove)
    
    if len(features_to_remove) > 0:
        logger.info(f"Removing {len(features_to_remove)} features: {features_to_remove}")
        df_cleaned = df.drop(columns=features_to_remove)
        return df_cleaned, features_to_remove
    
    return df, []


def bp_sanity_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create blood pressure sanity flags.
    
    Flags:
    - ap_swap_suspect: ap_lo > ap_hi
    - ap_hi_unreal: ap_hi < 80 or > 240
    - ap_lo_unreal: ap_lo < 40 or > 180
    - ap_pulse_neg: pulse pressure <= 0
    - any_flag: Any of the above flags
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with flags
    """
    flags = pd.DataFrame(index=df.index)
    
    if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
        flags['ap_swap_suspect'] = (df['ap_lo'] > df['ap_hi']).astype(int)
        flags['ap_hi_unreal'] = ((df['ap_hi'] < 80) | (df['ap_hi'] > 240)).astype(int)
        flags['ap_lo_unreal'] = ((df['ap_lo'] < 40) | (df['ap_lo'] > 180)).astype(int)
        flags['ap_pulse_neg'] = ((df['ap_hi'] - df['ap_lo']) <= 0).astype(int)
        flags['any_flag'] = flags.max(axis=1)
    
    return flags


def filter_outliers(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """
    Filter records with data quality issues.
    
    Args:
        df: Input dataframe
        strict: If True, remove all flagged records. If False, only remove ap_swap_suspect and ap_pulse_neg
        
    Returns:
        Filtered dataframe
    """
    flags = bp_sanity_flags(df)
    
    if strict:
        mask = flags['any_flag'] == 0
        logger.info(f"Strict filtering: Removing {(~mask).sum()} records with any BP flag")
    else:
        mask = (flags['ap_swap_suspect'] == 0) & (flags['ap_pulse_neg'] == 0)
        logger.info(f"Non-strict filtering: Removing {(~mask).sum()} records with ap_swap_suspect or ap_pulse_neg")
    
    return df.loc[mask].copy()


def build_preprocess(df: pd.DataFrame, target: str, 
                    imputation_method: str = 'knn',
                    use_mice: bool = False) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build preprocessing pipeline with ColumnTransformer.
    
    Based on EDA recommendations:
    - Numeric: KNNImputer(5) or MICE → StandardScaler
    - Discrete: KNNImputer(5) → OneHotEncoder
    
    Args:
        df: Input dataframe
        target: Target column name
        imputation_method: 'knn' or 'mice'
        use_mice: Whether to use MICE for numeric columns
        
    Returns:
        Tuple of (ColumnTransformer, numeric_cols, discrete_cols)
    """
    df = df.copy()
    
    # Ensure numeric types
    for c in ORDINAL_COLS + BINARY_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Identify columns
    discrete_cols = [c for c in (ORDINAL_COLS + BINARY_COLS) if c in df.columns]
    num_cols = [c for c in df.columns 
                if c not in discrete_cols + [target] 
                and pd.api.types.is_numeric_dtype(df[c])]
    
    # Build transformers
    transformers = []
    
    # Numeric transformer
    if len(num_cols) > 0:
        if use_mice and imputation_method == 'mice':
            try:
                numeric_steps = [
                    ("imputer", IterativeImputer(random_state=42, max_iter=10)),
                    ("scaler", StandardScaler())
                ]
            except Exception:
                logger.warning("MICE not available, falling back to KNN")
                numeric_steps = [
                    ("imputer", KNNImputer(n_neighbors=5)),
                    ("scaler", StandardScaler())
                ]
        else:
            numeric_steps = [
                ("imputer", KNNImputer(n_neighbors=5)),
                ("scaler", StandardScaler())
            ]
        transformers.append(("num", Pipeline(steps=numeric_steps), num_cols))
    
    # Discrete transformer
    if len(discrete_cols) > 0:
        discrete_steps = [
            ("imputer", KNNImputer(n_neighbors=5)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
        transformers.append(("disc", Pipeline(steps=discrete_steps), discrete_cols))
    
    ct = ColumnTransformer(transformers=transformers, remainder='passthrough')
    return ct, num_cols, discrete_cols


def validate_train_test_split(df_train: pd.DataFrame, df_test: pd.DataFrame, 
                             target: str, threshold: float = 0.6) -> float:
    """
    Validate train/test split using adversarial validation.
    
    Args:
        df_train: Training dataframe
        df_test: Test dataframe
        target: Target column name
        threshold: AUC threshold for warning (default: 0.6)
        
    Returns:
        AUC score
    """
    logger.info("Validating train/test split using adversarial validation...")
    
    try:
        # Combine train and test
        X_train = df_train.drop(columns=[target]).select_dtypes(include=[np.number])
        X_test = df_test.drop(columns=[target]).select_dtypes(include=[np.number])
        
        # Handle missing values
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_test.median())
        
        X_combined = pd.concat([X_train, X_test], ignore_index=True)
        y_adversarial = np.hstack([np.zeros(len(X_train)), np.ones(len(X_test))])
        
        # Train classifier
        clf = LogisticRegression(max_iter=2000, random_state=42)
        clf.fit(X_combined, y_adversarial)
        
        # Calculate AUC
        y_pred_proba = clf.predict_proba(X_combined)[:, 1]
        auc = roc_auc_score(y_adversarial, y_pred_proba)
        
        if auc > threshold:
            warnings.warn(
                f"Train/test distribution shift detected! AUC: {auc:.4f}. "
                f"Consider using stratified sampling or reviewing the split strategy."
            )
        else:
            logger.info(f"Adversarial validation AUC: {auc:.4f} - No significant shift detected")
        
        return auc
        
    except Exception as e:
        logger.warning(f"Adversarial validation failed: {e}")
        return 0.5  # Return neutral value


def prepare_cleaned_data(df: pd.DataFrame, target: str = "cardio",
                         strict_filter: bool = False,
                         clean_bp: bool = True,
                         handle_outliers_flag: bool = True,
                         use_mice: bool = False,
                         eda_outputs: Optional[Dict[str, Any]] = None,
                         use_eda_outputs: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Data cleaning pipeline: clean → handle outliers → impute.
    
    Based on High-Level Design: Preprocessing only handles data cleaning and imputation.
    Derived variables, transformation, scaling, and feature selection are moved to Feature Engineering stage.
    
    Args:
        df: Input dataframe
        target: Target column name
        strict_filter: Whether to use strict filtering
        clean_bp: Whether to clean blood pressure
        handle_outliers_flag: Whether to handle outliers
        use_mice: Whether to use MICE imputation
        eda_outputs: Optional dictionary of EDA outputs (from load_eda_outputs)
        use_eda_outputs: Whether to use EDA outputs if available
        
    Returns:
        Tuple of (cleaned_dataframe, metadata)
    """
    logger.info("=" * 60)
    logger.info("DATA CLEANING PIPELINE")
    logger.info("=" * 60)
    
    initial_count = len(df)
    metadata = {'initial_count': initial_count}
    
    # Load EDA outputs if not provided
    if eda_outputs is None and use_eda_outputs:
        logger.info("Loading EDA outputs...")
        eda_outputs = load_eda_outputs()
        metadata['used_eda_outputs'] = True
    else:
        metadata['used_eda_outputs'] = False
    
    # Step 1: Filter outliers (if strict_filter)
    if strict_filter:
        logger.info("Step 1: Strict filtering of outliers...")
        df = filter_outliers(df, strict=True)
        logger.info(f"Records after strict filtering: {len(df)} ({len(df)/initial_count*100:.2f}%)")
    
    # Step 2: Clean blood pressure
    if clean_bp:
        logger.info("Step 2: Cleaning blood pressure...")
        flagged_records = eda_outputs.get('flagged_records') if eda_outputs else None
        df = clean_blood_pressure(df, fix_swap=True, remove_negatives=True, cap_extreme=True,
                                 flagged_records=flagged_records if use_eda_outputs else None)
        logger.info(f"Records after BP cleaning: {len(df)} ({len(df)/initial_count*100:.2f}%)")
    
    # Step 3: Handle outliers (winsorize/truncate)
    if handle_outliers_flag:
        logger.info("Step 3: Handling outliers (winsorize/truncate)...")
        outliers_iqr = eda_outputs.get('outliers_iqr') if eda_outputs else None
        outliers_zscore = eda_outputs.get('outliers_zscore') if eda_outputs else None
        df = handle_outliers(df, method='winsorize',
                           outliers_iqr=outliers_iqr if use_eda_outputs else None,
                           outliers_zscore=outliers_zscore if use_eda_outputs else None)
    
    # Step 4: Improved imputation
    logger.info("Step 4: Imputing missing values...")
    df = improved_imputation(df, method='mice' if use_mice else 'knn', use_mice_for_numeric=use_mice)
    
    # Step 5: Final data quality check
    logger.info("Step 5: Final data quality check...")
    missing_final = df.isnull().sum().sum()
    if missing_final > 0:
        logger.warning(f"Still have {missing_final} missing values after imputation")
    else:
        logger.info("No missing values remaining")
    
    metadata['final_count'] = len(df)
    metadata['final_columns'] = list(df.columns)
    metadata['missing_after_imputation'] = int(missing_final)
    
    logger.info("=" * 60)
    logger.info("DATA CLEANING COMPLETED")
    logger.info(f"Initial records: {initial_count}")
    logger.info(f"Final records: {len(df)}")
    logger.info(f"Columns: {len(df.columns)}")
    logger.info("=" * 60)
    
    return df, metadata


def adversarial_validation(df_train: pd.DataFrame, df_test: pd.DataFrame, target: str) -> float:
    """
    Adversarial validation: detect distribution shift between train and test sets.
    
    Args:
        df_train: Training dataframe
        df_test: Test dataframe
        target: Target column name
        
    Returns:
        AUC score (higher = more shift)
    """
    X = pd.concat([df_train.drop(columns=[target]), df_test.drop(columns=[target])], ignore_index=True)
    y = np.r_[np.zeros(len(df_train)), np.ones(len(df_test))]
    X = X.select_dtypes(exclude=["object"]).fillna(X.median(numeric_only=True))
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(xte)[:, 1])


def main(args):
    """
    Main function to run preprocessing pipeline.
    
    Args:
        args: Command line arguments
    """
    cfg = load_config(args.config)
    logger = get_logger()
    set_seeds()
    
    # Ensure output directories exist
    ensure_dirs("artifacts/preprocessing")
    output_dir = "artifacts/preprocessing"
    
    # Load data
    data_path = cfg.get("paths", {}).get("data", "data/cardio_data.csv")
    df = load_data_with_delimiter(data_path)
    
    # Reliability: drop duplicate rows before any processing
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        logger.warning("Detected %d duplicated rows. Removing prior to cleaning.", duplicate_count)
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        logger.info("No duplicated rows detected.")

    # Basic input sanity checks
    if (df["gender"].isin([1, 2]).mean() < 1.0):
        bad_gender = df[~df["gender"].isin([1, 2])]
        logger.warning("Found %d rows with invalid gender values. Coercing to nearest valid category.", len(bad_gender))
        df.loc[~df["gender"].isin([1, 2]), "gender"] = df["gender"].mode(dropna=True).iloc[0]
    if (df["cardio"].isin([0, 1]).mean() < 1.0):
        bad_cardio = df[~df["cardio"].isin([0, 1])]
        logger.warning("Found %d rows with invalid cardio labels; dropping.", len(bad_cardio))
        df = df[df["cardio"].isin([0, 1])].copy()
        df.reset_index(drop=True, inplace=True)

    target = cfg.get("preprocessing", {}).get("target", "cardio")
    
    # Check if EDA outputs should be used
    use_eda_outputs = cfg.get("preprocessing", {}).get("use_eda_outputs", True)
    eda_dir = cfg.get("paths", {}).get("eda_dir", "artifacts/eda")
    
    # Verify target exists
    if target not in df.columns:
        logger.error(f"Target column '{target}' not found!")
        raise ValueError(f"Target column '{target}' not found")
    
    # Load EDA outputs if enabled
    eda_outputs = None
    if use_eda_outputs:
        eda_outputs = load_eda_outputs(eda_dir=eda_dir)
    
    # Run data cleaning pipeline
    df_cleaned, metadata = prepare_cleaned_data(
        df=df,
        target=target,
        strict_filter=args.strict_filter,
        clean_bp=cfg.get("preprocessing", {}).get("clean_bp", True),
        handle_outliers_flag=cfg.get("preprocessing", {}).get("handle_outliers", True),
        use_mice=cfg.get("preprocessing", {}).get("use_mice", False),
        eda_outputs=eda_outputs,
        use_eda_outputs=use_eda_outputs
    )
    metadata["duplicates_removed"] = duplicate_count

    # Class imbalance snapshot
    if target in df_cleaned.columns:
        ratio = df_cleaned[target].mean()
        metadata["positive_class_ratio"] = float(ratio)
        if ratio < 0.45 or ratio > 0.55:
            logger.warning(
                "Class imbalance detected after preprocessing. Positive ratio = %.3f. "
                "Consider class weighting or resampling in modeling.", ratio
            )
        else:
            logger.info("Class balance within acceptable range (positive ratio = %.3f).", ratio)
    
    # Save cleaned data to CSV
    cleaned_csv_path = f'{output_dir}/data_cleaned.csv'
    df_cleaned.to_csv(cleaned_csv_path, index=False)
    logger.info(f"Cleaned data saved to {cleaned_csv_path}")
    
    # Save metadata
    metadata_path = f'{output_dir}/preprocessing_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadata saved to {metadata_path}")
    
    logger.info("Data cleaning completed successfully!")
    logger.info("Next step: Feature Engineering (transformation, scaling, feature selection)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocessing: Data Cleaning, Outlier Handling, and Imputation"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--strict-filter", action="store_true",
                       help="Use strict filtering to remove all flagged records")
    args = parser.parse_args()
    main(args)
