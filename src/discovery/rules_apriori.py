from __future__ import annotations
import pandas as pd
from pathlib import Path
from mlxtend.frequent_patterns import apriori, association_rules

def load_feature_mapping(mapping_path: str = "artifacts/fe/feature_mapping.csv") -> dict:
    """
    Load feature mapping from CSV to map transformed names to meaningful names.
    
    Args:
        mapping_path: Path to feature_mapping.csv
        
    Returns:
        Dictionary mapping transformed_name -> selected_name (or meaningful name)
    """
    try:
        mapping_df = pd.read_csv(mapping_path)
        # Create mapping: transformed_name -> selected_name
        mapping = {}
        for _, row in mapping_df.iterrows():
            if row['selected'] == 'Yes' and pd.notna(row['selected_name']):
                mapping[row['transformed_name']] = row['selected_name']
        return mapping
    except Exception as e:
        print(f"Warning: Could not load feature mapping from {mapping_path}: {e}")
        return {}

def clean_feature_name(name: str) -> str:
    """
    Clean feature name for better readability in association rules.
    Examples: "num__age" -> "age", "ordinal__cholesterol" -> "cholesterol"
    """
    # Remove transformer prefix (e.g., "num__", "ordinal__")
    if '__' in name:
        name = name.split('__', 1)[1]
    return name

def discretize(df: pd.DataFrame, feature_mapping: dict = None) -> pd.DataFrame:
    """
    Discretize features for association rule mining.
    
    Args:
        df: Input dataframe with features
        feature_mapping: Optional mapping to rename features with meaningful names
    """
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        # Use meaningful name if available
        display_name = c
        if feature_mapping and c in feature_mapping:
            display_name = clean_feature_name(feature_mapping[c])
        elif '__' in c:
            display_name = clean_feature_name(c)
        
        if df[c].dtype.kind in "if":
            # Numeric: quantile-based discretization
            out[display_name] = pd.qcut(df[c].rank(method="first"), q=4, 
                                       labels=[f"{display_name}_Q1", f"{display_name}_Q2", 
                                               f"{display_name}_Q3", f"{display_name}_Q4"])
        else:
            # Categorical: convert to string
            out[display_name] = df[c].astype(str)
    return out

def mine_rules(df: pd.DataFrame, target_col: str = "cardio", 
               mapping_path: str = "artifacts/fe/feature_mapping.csv") -> pd.DataFrame:
    """
    Mine association rules from data.
    
    Args:
        df: Input dataframe with features and target
        target_col: Target column name
        mapping_path: Path to feature mapping CSV for meaningful feature names
        
    Returns:
        DataFrame with association rules sorted by lift and confidence
    """
    # Load feature mapping if available
    feature_mapping = load_feature_mapping(mapping_path) if Path(mapping_path).exists() else {}
    
    X = df.drop(columns=[target_col])
    y = df[target_col].map({0:"cardio=0", 1:"cardio=1"})
    
    # Discretize with meaningful names
    Xd = discretize(X, feature_mapping=feature_mapping)
    
    # Combine and one-hot encode
    onehot = pd.get_dummies(pd.concat([Xd, y], axis=1))
    
    # Mine frequent itemsets
    freq = apriori(onehot, min_support=0.05, use_colnames=True)
    
    # Generate association rules
    if len(freq) > 0:
        rules = association_rules(freq, metric="lift", min_threshold=1.1)
        return rules.sort_values(["lift","confidence"], ascending=False)
    else:
        print("Warning: No frequent itemsets found. Try lowering min_support.")
        return pd.DataFrame()
