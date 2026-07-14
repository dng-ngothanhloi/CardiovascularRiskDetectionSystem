"""
feature_engineering_v1.py  — clean replacement

Mục tiêu (đúng yêu cầu):
1) Không dùng --derived-csv. 
- Giữ Tạo derived_features.csv làm input cho step tiếp theo

2) Lưu kết quả riêng biệt theo từng model:
   - TabNet: <outdir>/tabnet/
   - RandomForest: <outdir>/random_forest/

3) Với mỗi model:
   - JSON: selected_features_*.json (danh sách feature)
   - CSV: selected_features_*.csv (FULL dataset = các feature đã chọn + target)
   - Train/Test: Train_data_*.csv / Test_data_*.csv (Stratified 80/20)
   - TabNet bổ sung FeatureAudit_tabnet.csv theo tiêu chí audit MI/AUC & lọc:
     * Constant/Near-constant
     * Redundancy |Spearman| > 0.92 (giữ MI cao hơn với cardio)
     * Binary duplicate |Spearman| > 0.95 so với “parent”
     * Loại ID-like {id, ID, index}
4/Xóa Feature Selection logic (auto-select-k, SHAP/L1)
5/Thêm CLI param --model-type để xử lý riêng cho TabNet và Random Forest
"""
from __future__ import annotations
import argparse
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import sys
# Add parent directory to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from .utils import load_config, get_logger, ensure_dirs, set_seeds
except ImportError:
    from src.utils import load_config, get_logger, ensure_dirs, set_seeds

# Constants for BMI groups
WHO_BMI_BINS = [0, 18.5, 25, 30, 1000]
WHO_BMI_LABELS = ["underweight", "normal", "overweight", "obese"]

logger = get_logger()


# ======================= Config & Helpers =======================
@dataclass
class AuditParams:
    threshold_corr_abs: float = 0.92
    binary_dup_corr_abs: float = 0.95
    mi_seed: int = 42


_ID_LIKE = {"id", "ID", "index"}
_COMPOSITE_PATTERNS = ("score", "risk_")
_INTERACTION_PATTERNS = ("*", "x", "_times_", "_mul_", "ratio_age", "bmi*chol", "interaction", "_interaction")
_BINARY_FLAG_SUFFIXES = ("_high", "_flag", "_bin", "_bool", "_cat_high")
_BINARY_FLAG_NAMES = {"overweight", "bp_cat_high", "cholesterol_high", "gluc_high"}


def _is_binary_series(s: pd.Series) -> bool:
    """Check if series contains only binary values (0/1)."""
    vals = set(pd.unique(s.dropna()))
    return vals <= {0, 1}


def _detect_groups(columns: List[str]) -> Dict[str, List[str]]:
    """Detect composite, interaction, and binary flag groups."""
    composite, interaction, binary_flags = [], [], []
    # Known interaction features from add_interaction_features()
    known_interactions = {"bmi_age_interaction", "pp_chol_interaction", "smoke_active"}
    
    for c in columns:
        lc = c.lower()
        if any(p in lc for p in _COMPOSITE_PATTERNS):
            composite.append(c)
        # Detect interaction: known interactions, contains "interaction", or matches patterns
        if c in known_interactions or "interaction" in lc or any(p in lc for p in _INTERACTION_PATTERNS if p not in ("interaction", "_interaction")):
            interaction.append(c)
        if c in _BINARY_FLAG_NAMES or any(lc.endswith(suffix) for suffix in _BINARY_FLAG_SUFFIXES):
            binary_flags.append(c)
    return {"composite": composite, "interaction": interaction, "binary_flags": binary_flags}


def _compute_mi_auc(X: pd.DataFrame, y: np.ndarray, seed: int) -> Tuple[pd.Series, pd.Series]:
    """Compute Mutual Information and univariate AUC for features."""
    num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    if not num_cols:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    Xn = X[num_cols].copy().fillna(X[num_cols].median(numeric_only=True))
    mi = pd.Series(
        mutual_info_classif(Xn.values, y, discrete_features=False, random_state=seed),
        index=num_cols,
    )
    auc_vals = {}
    for c in num_cols:
        try:
            auc_vals[c] = float(roc_auc_score(y, Xn[c].values)) if Xn[c].nunique() > 1 else float("nan")
        except Exception:
            auc_vals[c] = float("nan")
    return mi, pd.Series(auc_vals)


# ======================= Derived Features =======================
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived variables: age_years, BMI, pulse_pressure, age_group, bmi_group.
    
    This function creates 7 derived variables:
    1. age_years: Age in years (converted from days if needed)
    2. age_group: Age groups (<30, 30s, 40s, 50s, 60+)
    3. bmi: Body Mass Index
    4. bmi_group: BMI groups (underweight, normal, overweight, obese)
    5. pulse_pressure: ap_hi - ap_lo
    6. chol_high: Binary flag for cholesterol > 1
    7. gluc_high: Binary flag for glucose > 1
    """
    df = df.copy()
    
    # Age in years (if age is in days)
    if 'age' in df.columns:
        if df['age'].mean() > 120:  # Likely in days
            df['age_years'] = (df['age'] / 365.0).astype(int)
        else:
            df['age_years'] = df['age'].astype(int)
        
        # Age groups (encoded as numeric: 0=<30, 1=30s, 2=40s, 3=50s, 4=60+)
        age_cut = pd.cut(
            df['age_years'], 
            bins=[0, 29, 39, 49, 59, 200], 
            labels=[0, 1, 2, 3, 4],  # Numeric codes instead of string labels
            include_lowest=True
        )
        df['age_group'] = age_cut.astype(float).astype(int)
        # Remove original day-based age to avoid duplicated signal
        df = df.drop(columns=['age'], errors='ignore')
    
    # BMI calculation (only for valid height/weight)
    if 'weight' in df.columns and 'height' in df.columns:
        valid_mask = (df['height'] > 0) & (df['weight'] > 0)
        h_m = df.loc[valid_mask, 'height'] / 100.0
        df['bmi'] = np.nan
        df.loc[valid_mask, 'bmi'] = df.loc[valid_mask, 'weight'] / (h_m ** 2 + 1e-9)
        
        # BMI groups (only for valid BMI values) - encoded as numeric: 0=underweight, 1=normal, 2=overweight, 3=obese
        valid_bmi_mask = df['bmi'].notna() & (df['bmi'] > 0) & (df['bmi'] < 1000)
        if valid_bmi_mask.sum() > 0:
            bmi_cut = pd.cut(
                df.loc[valid_bmi_mask, 'bmi'],
                bins=WHO_BMI_BINS,
                labels=[0, 1, 2, 3],  # Numeric codes instead of string labels
                include_lowest=True
            )
            df['bmi_group'] = np.nan
            df.loc[valid_bmi_mask, 'bmi_group'] = bmi_cut.astype(float).astype(int)
            # Guideline: overweight flag (BMI >= 25)
            df['overweight_flag'] = (df['bmi'] >= 25).astype(int)
        else:
            df['overweight_flag'] = 0
    else:
        df['overweight_flag'] = 0
    
    # Pulse pressure (only for valid BP values)
    if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
        valid_bp_mask = (df['ap_hi'] > 0) & (df['ap_lo'] > 0) & (df['ap_hi'] > df['ap_lo'])
        df['pulse_pressure'] = np.nan
        df.loc[valid_bp_mask, 'pulse_pressure'] = (
            df.loc[valid_bp_mask, 'ap_hi'] - df.loc[valid_bp_mask, 'ap_lo']
        )
        # Blood pressure stage per AHA/ACC guideline
        bp_stage = np.full(len(df), np.nan)
        bp_stage = pd.Series(bp_stage, index=df.index)
        systolic = df['ap_hi']
        diastolic = df['ap_lo']
        bp_stage[(systolic < 120) & (diastolic < 80)] = 0  # Normal
        bp_stage[((120 <= systolic) & (systolic < 130)) & (diastolic < 80)] = 1  # Elevated
        bp_stage[((130 <= systolic) & (systolic < 140)) | ((80 <= diastolic) & (diastolic < 90))] = 2  # Stage 1
        bp_stage[((140 <= systolic) & (systolic < 180)) | ((90 <= diastolic) & (diastolic < 120))] = 3  # Stage 2
        bp_stage[(systolic >= 180) | (diastolic >= 120)] = 4  # Hypertensive crisis
        df['bp_stage'] = bp_stage.fillna(0).astype(int)
    else:
        df['bp_stage'] = 0
    
    # Cholesterol and Glucose high flags
    if 'cholesterol' in df.columns:
        df['chol_high'] = (df['cholesterol'] > 1).astype(int)
        df['cholesterol_category'] = df['cholesterol'].map({1: 0, 2: 1, 3: 2}).fillna(0).astype(int)
    if 'gluc' in df.columns:
        df['gluc_high'] = (df['gluc'] > 1).astype(int)
        df['glucose_category'] = df['gluc'].map({1: 0, 2: 1, 3: 2}).fillna(0).astype(int)

    # Lifestyle score combining smoking, alcohol, physical activity
    lifestyle_cols = [c for c in ["smoke", "alco", "active"] if c in df.columns]
    if lifestyle_cols:
        lifestyle_score = pd.Series(0, index=df.index, dtype=float)
        if "smoke" in df.columns:
            lifestyle_score += df["smoke"].fillna(0) * 1.0
        if "alco" in df.columns:
            lifestyle_score += df["alco"].fillna(0) * 0.5
        if "active" in df.columns:
            lifestyle_score -= df["active"].fillna(0) * 0.5
        df["lifestyle_risk"] = lifestyle_score
    else:
        df["lifestyle_risk"] = 0.0
    
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features."""
    df = df.copy()
    
    # BMI * Age interaction
    if 'bmi' in df.columns and 'age_years' in df.columns:
        df['bmi_age_interaction'] = df['bmi'] * df['age_years']
    
    # Pulse pressure * Cholesterol interaction
    if 'pulse_pressure' in df.columns and 'cholesterol' in df.columns:
        df['pp_chol_interaction'] = df['pulse_pressure'] * df['cholesterol']
    
    # Smoke * Active interaction
    if 'smoke' in df.columns and 'active' in df.columns:
        df['smoke_active'] = df['smoke'] * df['active']
    
    return df


# ======================= Post-process cho TabNet ===================
def _tabnet_postprocess_on_derived(
    derived_csv_path: str,
    outdir: str,
    target: str = "cardio",
    skip_audit: bool = False,
    keep_all_features: bool = False,
) -> Tuple[List[str], Optional[pd.DataFrame]]:
    """
    Post-process derived_features.csv cho TabNet.
    - SKIP: Binary flags, Composite indicators, Interaction terms
    - Nếu skip_audit=False: Loại bỏ Constant/Near-constant, Redundancy cao, Binary duplicates, ID-like
    - Nếu skip_audit=True: Chỉ loại bỏ Binary flags, Composite, Interaction, ID-like (không audit)
    - Xuất: FeatureAudit_tabnet.csv (nếu không skip), selected_features_tabnet.json, selected_features_tabnet.csv
    """
    df_d = pd.read_csv(derived_csv_path)
    if target not in df_d.columns:
        raise ValueError(f"Target column '{target}' not found in {derived_csv_path}")
    y = df_d[target].astype(int).values
    raw = [c for c in df_d.columns if c != target]
    groups = _detect_groups(raw)

    # Bỏ nhóm B/C/D trước khi audit
    drop_initial = set(groups["binary_flags"]) | set(groups["composite"]) | set(groups["interaction"]) | _ID_LIKE
    X = df_d[[c for c in raw if c not in drop_initial]].copy()

    # Numeric cast
    for c in X.columns:
        if X[c].dtype == "bool":
            X[c] = X[c].astype(int)
    for column in X.columns:
        if not pd.api.types.is_numeric_dtype(X[column]):
            try:
                X[column] = pd.to_numeric(X[column])
            except (TypeError, ValueError):
                continue
    num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]

    if skip_audit:
        selected_initial = [c for c in X.columns if c not in _ID_LIKE]
        audit = None
        logger.info("[v1] TabNet post-process (skip audit): Using all features after removing B/C/D groups")
    else:
        # Chạy audit đầy đủ
        params = AuditParams()
        # MI/AUC + thống kê
        mi, auc = _compute_mi_auc(X, y, params.mi_seed)
        variance = X.var(numeric_only=True)
        unique_counts = X.nunique()
        n = len(X)

        constant_feats = [c for c in num_cols if unique_counts[c] <= 1]
        near_constant_feats = [c for c in num_cols if unique_counts[c] <= max(2, int(0.001 * n))]

        corr = X[num_cols].corr(method="spearman").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).fillna(0.0)
        pairs_high = [
            (r, c, float(upper.loc[r, c]))
            for r in upper.index
            for c in upper.columns
            if upper.loc[r, c] > params.threshold_corr_abs
        ]
        pairs_high = sorted(pairs_high, key=lambda x: -x[2])

        to_drop_redundant = set()
        kept = set(num_cols)
        for a, b, r in pairs_high:
            if a in to_drop_redundant or b in to_drop_redundant:
                continue
            mi_a = float(mi.get(a, np.nan)) if a in mi.index else np.nan
            mi_b = float(mi.get(b, np.nan)) if b in mi.index else np.nan
            if not math.isnan(mi_a) and not math.isnan(mi_b):
                drop = b if mi_a >= mi_b else a
            else:
                var_a = float(variance.get(a, 0.0))
                var_b = float(variance.get(b, 0.0))
                drop = b if var_a >= var_b else a
            to_drop_redundant.add(drop)
            kept.discard(drop)

        drop_flags = set()
        for c in num_cols:
            if _is_binary_series(X[c]):
                row = corr.loc[c].drop(index=c) if c in corr.index else pd.Series(dtype=float)
                if not row.empty and float(row.max()) > params.binary_dup_corr_abs:
                    drop_flags.add(c)

        remove_list = sorted(
            set(constant_feats)
            | set(near_constant_feats)
            | to_drop_redundant
            | drop_flags
            | (_ID_LIKE & set(X.columns))
        )
        selected_initial = [c for c in X.columns if c not in remove_list]

    # Thư mục & audit
    outdir_p = Path(outdir) / "tabnet"
    outdir_p.mkdir(parents=True, exist_ok=True)
    
    if not skip_audit:
        # Tạo audit DataFrame chỉ khi không skip audit
        audit = pd.DataFrame(
            {
                "feature": num_cols,
                "mutual_info": [float(mi.get(c, np.nan)) for c in num_cols],
                "univariate_auc": [float(auc.get(c, np.nan)) for c in num_cols],
                "is_binary": [bool(_is_binary_series(X[c])) for c in num_cols],
                "variance": [float(variance.get(c, np.nan)) for c in num_cols],
                "selected": [c in selected_initial for c in num_cols],
            }
        ).sort_values(["selected", "mutual_info"], ascending=[False, False])
        audit.to_csv(outdir_p / "FeatureAudit_tabnet.csv", index=False)
    else:
        audit = None

    # Tạo JSON với notes phù hợp
    if skip_audit:
        notes = {
            "skip_audit": True,
            "note": "Only removed Binary flags, Composite, Interaction, and ID-like features. No MI/AUC audit performed."
        }
    else:
        params = AuditParams()
        notes = {
            "skip_audit": False,
            "threshold_corr_abs": params.threshold_corr_abs,
            "binary_dup_corr_abs": params.binary_dup_corr_abs,
            "mi_seed": params.mi_seed,
        }
    
    selected_final = list(X.columns) if keep_all_features else selected_initial

    Path(outdir_p / "selected_features_tabnet.json").write_text(
        json.dumps(
            {
                "notes": notes,
                "kept_features": selected_final,
            },
            indent=2,
        )
    )

    # FULL dataset theo danh sách đã chọn + target
    df_tab = df_d[selected_final + [target]].copy()
    if "id" in df_d.columns:
        df_tab.insert(0, "sample_id", df_d["id"].values)
    df_tab.to_csv(outdir_p / "selected_features_tabnet.csv", index=False)

    # Train/Test split (Stratified 80/20)
    train_df, test_df = train_test_split(df_tab, test_size=0.2, stratify=df_tab[target], random_state=42)
    train_df.to_csv(outdir_p / "Train_data_tabnet.csv", index=False)
    test_df.to_csv(outdir_p / "Test_data_tabnet.csv", index=False)

    logger.info("[v1] TabNet post-process OK. Outputs in: %s", str(outdir_p))
    return selected_final, audit


# =================== Post-process cho Random Forest ===================
def _rf_postprocess_on_derived(derived_csv_path: str, outdir: str, target: str = "cardio"):
    """
    Post-process derived_features.csv cho Random Forest.
    - Loại bỏ: Interaction, Composite, ID-like
    - Giữ: Binary flags, tất cả tương quan mạnh
    - Xuất: selected_features_rf.json, selected_features_rf.csv
    """
    df_d = pd.read_csv(derived_csv_path)
    if target not in df_d.columns:
        raise ValueError(f"Target column '{target}' not found in {derived_csv_path}")
    raw = [c for c in df_d.columns if c != target]
    groups = _detect_groups(raw)

    # RF: bỏ composite & interaction; giữ binary flags; không cắt theo tương quan
    drop = set(groups["composite"]) | set(groups["interaction"]) | _ID_LIKE
    kept = [c for c in raw if c not in drop]

    outdir_p = Path(outdir) / "random_forest"
    outdir_p.mkdir(parents=True, exist_ok=True)

    Path(outdir_p / "selected_features_rf.json").write_text(json.dumps({"kept_features": kept}, indent=2))

    # FULL dataset theo danh sách đã chọn + target
    df_rf = df_d[kept + [target]].copy()
    if "id" in df_d.columns:
        df_rf.insert(0, "sample_id", df_d["id"].values)
    df_rf.to_csv(outdir_p / "selected_features_rf.csv", index=False)

    # Train/Test split (Stratified 80/20)
    train_df, test_df = train_test_split(df_rf, test_size=0.2, stratify=df_rf[target], random_state=42)
    train_df.to_csv(outdir_p / "Train_data_rf.csv", index=False)
    test_df.to_csv(outdir_p / "Test_data_rf.csv", index=False)

    logger.info("[v1] RF post-process OK. Outputs in: %s", str(outdir_p))
    return kept


# ======================= Main Function =======================
def main(args):
    """Main function to run feature engineering v1 pipeline."""
    logger = get_logger()
    
    # Load config with defaults
    try:
        cfg = load_config(args.config)
    except Exception as e:
        logger.warning(f"Could not load config from {args.config}: {e}. Using defaults.")
        cfg = {}
    set_seeds()
    
    # Get output directory from args or config
    output_dir = getattr(args, 'outdir', None) or cfg.get("paths", {}).get("fe_dir", "artifacts/fe") if isinstance(cfg, dict) else "artifacts/fe"
    ensure_dirs(output_dir)
    
    # Get target from args or config
    target = getattr(args, 'target', None) or cfg.get("preprocessing", {}).get("target", "cardio") if isinstance(cfg, dict) else "cardio"
    
    # Get model-type from args (default: rf)
    model_type = getattr(args, 'model_type', 'rf')
    if model_type not in ['rf', 'tabnet']:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'rf' or 'tabnet'")
    
    # Load cleaned data from preprocessing
    input_path = cfg.get("paths", {}).get("preprocessing_dir", "artifacts/preprocessing") if isinstance(cfg, dict) else "artifacts/preprocessing"
    cleaned_data_path = f"{input_path}/data_cleaned.csv"
    
    # Fallback: check alternative location
    if not Path(cleaned_data_path).exists():
        alt_path = "artifacts/fe/data_cleaned.csv"
        if Path(alt_path).exists():
            cleaned_data_path = alt_path
            logger.info(f"Using alternative path: {cleaned_data_path}")
    
    logger.info(f"Loading cleaned data from: {cleaned_data_path}")
    df = pd.read_csv(cleaned_data_path)
    logger.info(f"Data loaded: {df.shape}")
    
    # Verify target exists
    if target not in df.columns:
        logger.error(f"Target column '{target}' not found!")
        raise ValueError(f"Target column '{target}' not found")
    
    # Get feature engineering config (with defaults)
    fe_config = cfg.get("feature_engineering", {}) if isinstance(cfg, dict) else {}
    
    # Step 1: Create derived features
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING V1 - Creating derived features")
    logger.info("=" * 60)
    logger.info("Step 1: Creating derived variables...")
    original_columns = set(df.columns)
    df_with_derived = add_derived(df)
    derived_columns = set(df_with_derived.columns)
    new_derived_features = sorted(list(derived_columns - original_columns))
    logger.info(f"Derived variables created. Columns: {len(df_with_derived.columns)}")
    logger.info(f"New derived features created ({len(new_derived_features)}): {new_derived_features}")
    
    # Step 2: Add interaction features (always create for derived_features.csv)
    logger.info("Step 2: Adding interaction features...")
    columns_before_interaction = set(df_with_derived.columns)
    df_with_derived = add_interaction_features(df_with_derived)
    columns_after_interaction = set(df_with_derived.columns)
    new_interaction_features = sorted(list(columns_after_interaction - columns_before_interaction))
    logger.info(f"Interaction features added. Columns: {len(df_with_derived.columns)}")
    logger.info(f"New interaction features created ({len(new_interaction_features)}): {new_interaction_features}")
    
    # Step 3: Save derived_features.csv
    derived_features_path = f"{output_dir}/derived_features.csv"
    df_with_derived.to_csv(derived_features_path, index=False)
    logger.info(f"Derived features saved to {derived_features_path}")
    
    # Step 4: Post-process theo model-type
    logger.info("=" * 60)
    logger.info(f"Step 3: Post-processing for model-type: {model_type}")
    logger.info("=" * 60)
    
    # Get skip_audit option (chỉ áp dụng cho TabNet)
    fe_tabnet_cfg = fe_config.get("tabnet", {}) if isinstance(fe_config, dict) else {}
    skip_audit = getattr(args, 'skip_audit', False) or fe_tabnet_cfg.get("skip_audit", False)
    keep_all_features = getattr(args, 'tabnet_keep_all', False) or fe_tabnet_cfg.get("keep_all_features", False)
    
    if model_type == 'tabnet':
        _tabnet_postprocess_on_derived(
            derived_features_path,
            output_dir,
            target,
            skip_audit=skip_audit,
            keep_all_features=keep_all_features,
        )
    else:  # rf
        _rf_postprocess_on_derived(derived_features_path, output_dir, target)
    
    logger.info("=" * 60)
    logger.info("Feature engineering v1 completed successfully!")
    logger.info("Next step: Modeling")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature Engineering v1: Derived Variables + Model-specific Post-processing"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model-type", type=str, choices=["rf", "tabnet"], default="rf",
                       help="Model type: 'rf' (Random Forest) or 'tabnet' (TabNet). Default: 'rf'")
    parser.add_argument("--outdir", type=str, default=None,
                       help="Output directory for feature engineering artifacts (default: from config or 'artifacts/fe')")
    parser.add_argument("--target", type=str, default=None,
                       help="Target column name (default: from config or 'cardio')")
    parser.add_argument("--skip-audit", action="store_true", default=False,
                       help="Skip MI/AUC audit for TabNet (only remove B/C/D groups and ID-like). Default: False (run audit)")
    parser.add_argument("--tabnet-keep-all", action="store_true", default=False,
                       help="Keep all TabNet features after audit (still generate audit report).")
    args = parser.parse_args()
    main(args)

