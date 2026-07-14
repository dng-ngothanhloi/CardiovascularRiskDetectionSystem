from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

logger = logging.getLogger(__name__)


def run_apriori(items_df: pd.DataFrame, cfg: Dict[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the Apriori algorithm followed by association rule mining.

    Parameters
    ----------
    items_df:
        Transaction matrix containing only binary item columns.
    cfg:
        Apriori configuration containing keys:
        - min_support
        - use_colnames
        - max_len
        - metric
        - min_threshold

    Returns
    -------
    frequent_itemsets, rules
    """
    if items_df.empty:
        logger.warning("Received empty itemset DataFrame; skipping Apriori.")
        return pd.DataFrame(), pd.DataFrame()

    apriori_kwargs = {
        "min_support": float(cfg.get("min_support", 0.02)),
        "use_colnames": bool(cfg.get("use_colnames", True)),
        "max_len": int(cfg.get("max_len", 4)),
    }
    metric = str(cfg.get("metric", "confidence"))
    min_threshold = float(cfg.get("min_threshold", 0.6))

    bool_df = items_df.astype(bool)
    frequent_itemsets = apriori(bool_df, **apriori_kwargs)

    if frequent_itemsets.empty:
        logger.warning("No frequent itemsets found with support >= %.4f.", apriori_kwargs["min_support"])
        return frequent_itemsets, pd.DataFrame()

    try:
        rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    except ValueError:
        logger.warning(
            "association_rules returned empty results for metric %s and threshold %.3f.",
            metric,
            min_threshold,
        )
        rules = pd.DataFrame()

    return frequent_itemsets, rules


def filter_consequents_to_positive(rules: pd.DataFrame, target_col: str = "cardio") -> pd.DataFrame:
    """
    Retain only rules whose consequent is exactly the positive target column.
    """
    if rules.empty:
        return rules

    target_set = frozenset({target_col})
    mask = rules["consequents"].apply(lambda cons: isinstance(cons, frozenset) and cons == target_set)
    filtered = rules[mask].copy()
    if filtered.empty:
        logger.warning("No rules matched consequent %s=1; returning empty rules.", target_col)
    return filtered


def mine_rules(
    apriori_cfg: Dict[str, object],
    cohort_cfg: Dict[str, object],
    discovery_dir: Path,
    items_df: Optional[pd.DataFrame] = None,
    target_col: str = "cardio",
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Full Apriori mining pipeline including optional cohort-wise mining.

    Parameters
    ----------
    apriori_cfg:
        Configuration dictionary for Apriori and association rules.
    cohort_cfg:
        Cohort mining configuration dictionary.
    discovery_dir:
        Root discovery directory (`artifacts/Discovery`).
    items_df:
        Optional pre-loaded itemset DataFrame.
    target_col:
        Target column name expected in the itemset.

    Returns
    -------
    frequent_itemsets, rules_global, rules_by_cohort
    """
    itemset_dir = discovery_dir / "itemsets"
    apriori_dir = discovery_dir / "apriori"
    apriori_dir.mkdir(parents=True, exist_ok=True)

    if items_df is None:
        itemset_path = itemset_dir / "binarized_itemset.csv"
        if not itemset_path.exists():
            raise FileNotFoundError(f"Binarized itemset not found: {itemset_path}")
        items_df = pd.read_csv(itemset_path)

    meta_path = itemset_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Itemset metadata missing: {meta_path}")

    meta = json.loads(meta_path.read_text())
    feature_names = meta.get("feature_names", [])

    item_columns = list(feature_names)
    if target_col in items_df.columns:
        item_columns.append(target_col)

    missing_cols = [col for col in item_columns if col not in items_df.columns]
    if missing_cols:
        raise ValueError(f"Itemset missing required columns: {missing_cols}")

    # Ensure binary form (0/1) and drop non-item columns for Apriori.
    transactions = items_df[item_columns].fillna(0).astype(int).clip(lower=0, upper=1)
    frequent_itemsets, rules = run_apriori(transactions, apriori_cfg)

    if target_col in transactions.columns:
        rules = filter_consequents_to_positive(rules, target_col=target_col)
    else:
        logger.warning(
            "Target column %s not present in itemset; mining general rules without target constraint.",
            target_col,
        )

    freq_path = apriori_dir / "frequent_itemsets.csv"
    rules_path = apriori_dir / "rules_raw.csv"
    frequent_itemsets.to_csv(freq_path, index=False)
    rules.to_csv(rules_path, index=False)

    logger.info("Saved frequent itemsets to %s", freq_path)
    logger.info("Saved raw rules to %s", rules_path)

    cohort_rules = _mine_cohort_rules(
        items_df=items_df,
        apriori_cfg=apriori_cfg,
        cohort_cfg=cohort_cfg,
        item_columns=item_columns,
        target_col=target_col,
        apriori_dir=apriori_dir,
    )

    return frequent_itemsets, rules, cohort_rules


def _mine_cohort_rules(
    items_df: pd.DataFrame,
    apriori_cfg: Dict[str, object],
    cohort_cfg: Dict[str, object],
    item_columns: List[str],
    target_col: str,
    apriori_dir: Path,
) -> Optional[pd.DataFrame]:
    if not cohort_cfg.get("enabled", False):
        return None

    cohort_path = cohort_cfg.get("file")
    if not cohort_path:
        logger.warning("Cohort mining enabled but no cohort file provided; skipping.")
        return None

    cohort_file = Path(cohort_path)
    if not cohort_file.exists():
        logger.warning("Cohort file %s not found; skipping cohort mining.", cohort_file)
        return None

    cohort_df = pd.read_csv(cohort_file)
    if "id" not in cohort_df.columns:
        logger.warning("Cohort file missing `id` column; skipping cohort mining.")
        return None

    if "id" not in items_df.columns:
        logger.warning("Itemset lacks `id` column required for cohort join; skipping cohort mining.")
        return None

    merged = items_df.merge(cohort_df[["id", "cohort_label"]], on="id", how="left")
    if "cohort_label" not in merged.columns:
        logger.warning("Cohort labels not found after merge; skipping cohort mining.")
        return None

    min_size = int(cohort_cfg.get("min_cohort_size", 200))
    if not cohort_cfg.get("mine_per_cohort", True):
        return None

    cohort_rules_frames: List[pd.DataFrame] = []
    for label, cohort_subset in merged.groupby("cohort_label"):
        cohort_subset = cohort_subset.dropna(subset=["cohort_label"])
        if cohort_subset.empty:
            continue
        if len(cohort_subset) < min_size:
            logger.info(
                "Skipping cohort %s (%s samples) < min size %s.",
                label,
                len(cohort_subset),
                min_size,
            )
            continue

        transactions = cohort_subset[item_columns].fillna(0).astype(int).clip(lower=0, upper=1)
        _, rules = run_apriori(transactions, apriori_cfg)
        if target_col in transactions.columns:
            rules = filter_consequents_to_positive(rules, target_col=target_col)

        if rules.empty:
            logger.info("No rules mined for cohort %s.", label)
            continue

        rules = rules.copy()
        rules["cohort_label"] = label
        cohort_rules_frames.append(rules)

    if not cohort_rules_frames:
        return None

    cohort_rules = pd.concat(cohort_rules_frames, axis=0, ignore_index=True)
    cohort_path_out = apriori_dir / "rules_by_cohort.csv"
    cohort_rules.to_csv(cohort_path_out, index=False)
    logger.info("Saved cohort rules to %s", cohort_path_out)
    return cohort_rules


