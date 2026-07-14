from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .apriori_runner import filter_consequents_to_positive, run_apriori

logger = logging.getLogger(__name__)


def prune_redundant(rules: pd.DataFrame) -> pd.DataFrame:
    """
    Remove redundant rules where antecedents are supersets of stronger rules.
    """
    if rules.empty:
        return rules

    metrics = ["lift", "confidence", "support"]
    sorted_rules = rules.sort_values(metrics, ascending=[False, False, False]).reset_index(drop=True)
    keep_indices: List[int] = []
    tolerance = 1e-9

    for idx, row in sorted_rules.iterrows():
        antecedent = _ensure_frozenset(row["antecedents"])
        dominated = False
        for kept_idx in keep_indices:
            kept_row = sorted_rules.loc[kept_idx]
            kept_antecedent = _ensure_frozenset(kept_row["antecedents"])
            if not antecedent.issuperset(kept_antecedent):
                continue
            # Check dominance: kept rule metrics >= current.
            dominates = all(
                (kept_row[metric] + tolerance) >= row[metric] for metric in metrics if metric in kept_row and metric in row
            )
            if dominates:
                dominated = True
                break
        if not dominated:
            keep_indices.append(idx)

    pruned = sorted_rules.loc[keep_indices].reset_index(drop=True)
    logger.info("Pruned %s redundant rules (kept %s).", len(sorted_rules) - len(pruned), len(pruned))
    return pruned


def bootstrap_stability(
    items_df: pd.DataFrame,
    apriori_cfg: Dict[str, object],
    bootstrap_cfg: Dict[str, object],
    item_columns: Sequence[str],
    target_col: str = "cardio",
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Estimate rule stability via bootstrap resampling.
    """
    if not bootstrap_cfg.get("enabled", False):
        return pd.DataFrame(columns=["antecedents_key", "stability"])

    n_boot = int(bootstrap_cfg.get("n_boot", 200))
    sample_frac = float(bootstrap_cfg.get("sample_frac", 0.8))
    if not (0 < sample_frac <= 1):
        raise ValueError("Bootstrap sample_frac must be in (0, 1].")

    rng = np.random.default_rng(random_seed or 0)
    presence_counter: Counter[str] = Counter()

    for b in range(n_boot):
        sampled_idx = rng.choice(len(items_df), size=max(1, int(len(items_df) * sample_frac)), replace=True)
        sampled = items_df.iloc[sampled_idx]
        transactions = sampled[item_columns].fillna(0).astype(int).clip(lower=0, upper=1)
        _, rules_b = run_apriori(transactions, apriori_cfg)
        if target_col in transactions.columns:
            rules_b = filter_consequents_to_positive(rules_b, target_col=target_col)
        if rules_b.empty:
            continue
        keys = rules_b["antecedents"].apply(lambda fs: _itemset_key(_ensure_frozenset(fs)))
        presence_counter.update(keys.tolist())

    if n_boot == 0:
        return pd.DataFrame(columns=["antecedents_key", "stability"])

    stability_records = [
        {"antecedents_key": key, "stability": presence_counter[key] / n_boot} for key in presence_counter
    ]
    return pd.DataFrame(stability_records)


def rank_rules(rules: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """
    Compute harmonic-mean based ranking scores for rules.
    """
    if rules.empty:
        rules = rules.copy()
        rules["rank_score"] = pd.Series(dtype=float)
        return rules

    weights = {metric: float(weight) for metric, weight in weights.items() if float(weight) > 0}
    if not weights:
        rules = rules.copy()
        rules["rank_score"] = np.nan
        return rules

    normalised: Dict[str, pd.Series] = {}
    for metric, weight in weights.items():
        if metric not in rules.columns:
            logger.warning("Metric %s missing from rules; skipping in ranking.", metric)
            continue
        series = pd.to_numeric(rules[metric], errors="coerce").fillna(0.0)
        min_val, max_val = series.min(), series.max()
        if np.isclose(max_val, min_val):
            normalised[metric] = pd.Series(np.ones(len(series)), index=rules.index)
        else:
            normalised[metric] = (series - min_val) / (max_val - min_val)

    if not normalised:
        rules = rules.copy()
        rules["rank_score"] = np.nan
        return rules

    epsilon = 1e-9
    scores: List[float] = []
    for idx in rules.index:
        numerator = 0.0
        denominator = 0.0
        for metric, weight in weights.items():
            if metric not in normalised:
                continue
            value = max(normalised[metric].iloc[idx], epsilon)
            numerator += weight
            denominator += weight / value
        scores.append(numerator / denominator if denominator > 0 else 0.0)

    ranked = rules.copy()
    ranked["rank_score"] = scores
    ranked.sort_values(by="rank_score", ascending=False, inplace=True)
    ranked.reset_index(drop=True, inplace=True)
    return ranked


def postprocess_rules(
    apriori_cfg: Dict[str, object],
    post_cfg: Dict[str, object],
    discovery_dir: Path,
    items_df: Optional[pd.DataFrame] = None,
    target_col: str = "cardio",
    random_seed: int = 42,
    ranking_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Apply filtering, redundancy pruning, stability estimation, and ranking to raw rules.
    """
    apriori_dir = discovery_dir / "apriori"
    itemset_dir = discovery_dir / "itemsets"
    rules_path = apriori_dir / "rules_raw.csv"
    if not rules_path.exists():
        logger.warning("Raw rules file not found (%s); skipping post-processing.", rules_path)
        return pd.DataFrame()

    rules = pd.read_csv(rules_path)
    if rules.empty:
        logger.warning("Raw rules file is empty; nothing to post-process.")
        return rules

    rules["antecedents"] = rules["antecedents"].apply(_ensure_frozenset)
    rules["consequents"] = rules["consequents"].apply(_ensure_frozenset)
    rules["antecedents_key"] = rules["antecedents"].apply(_itemset_key)

    min_support = float(post_cfg.get("min_support", apriori_cfg.get("min_support", 0.0)))
    min_confidence = float(post_cfg.get("min_confidence", apriori_cfg.get("min_threshold", 0.0)))
    min_lift = float(post_cfg.get("min_lift", 1.0))

    filtered = rules[
        (rules["support"] >= min_support)
        & (rules["confidence"] >= min_confidence)
        & (rules["lift"] >= min_lift)
    ].copy()

    if filtered.empty:
        logger.warning(
            "All rules filtered out by thresholds (support≥%.3f, confidence≥%.3f, lift≥%.3f).",
            min_support,
            min_confidence,
            min_lift,
        )
        filtered["stability"] = np.nan
        filtered["rank_score"] = np.nan
        filtered.to_csv(apriori_dir / "rules_pruned.csv", index=False)
        return filtered

    if post_cfg.get("redundancy", True):
        pruned = prune_redundant(filtered)
    else:
        pruned = filtered.copy()

    # Attach stability scores if requested.
    itemset_path = itemset_dir / "binarized_itemset.csv"
    if items_df is None and itemset_path.exists():
        items_df = pd.read_csv(itemset_path)

    stability_scores = pd.DataFrame(columns=["antecedents_key", "stability"])
    bootstrap_cfg = post_cfg.get("stability_bootstrap", {})
    if bootstrap_cfg.get("enabled", False):
        if items_df is None:
            logger.warning("Bootstrap requested but itemset data unavailable; skipping stability estimation.")
            pruned["stability"] = np.nan
        else:
            meta_path = itemset_dir / "meta.json"
            if not meta_path.exists():
                logger.warning("Itemset meta.json missing; cannot identify item columns for bootstrap.")
                pruned["stability"] = np.nan
            else:
                meta = json.loads(meta_path.read_text())
                item_columns = list(meta.get("feature_names", []))
                if target_col in items_df.columns:
                    item_columns.append(target_col)
                stability_scores = bootstrap_stability(
                    items_df=items_df,
                    apriori_cfg=apriori_cfg,
                    bootstrap_cfg=bootstrap_cfg,
                    item_columns=item_columns,
                    target_col=target_col,
                    random_seed=random_seed,
                )
                if not stability_scores.empty:
                    min_presence = float(bootstrap_cfg.get("min_presence", 0.0))
                    pruned = pruned.merge(stability_scores, on="antecedents_key", how="left")
                    pruned["stability"] = pruned["stability"].fillna(0.0)
                    pruned = pruned[pruned["stability"] >= min_presence].reset_index(drop=True)
                    logger.info(
                        "Bootstrap stability applied; %s rules retained with presence ≥ %.2f.",
                        len(pruned),
                        min_presence,
                    )
                else:
                    pruned["stability"] = np.nan
    else:
        pruned["stability"] = np.nan

    weights = ranking_weights if ranking_weights is not None else post_cfg.get("ranking", {}).get("weights", {})
    ranked = rank_rules(pruned, weights=weights)

    # Persist final rules.
    rules_pruned_path = apriori_dir / "rules_pruned.csv"
    ranked.to_csv(rules_pruned_path, index=False)
    logger.info("Saved pruned rules to %s", rules_pruned_path)
    return ranked


def _ensure_frozenset(value: object) -> frozenset:
    if isinstance(value, frozenset):
        return value
    if isinstance(value, (set, list, tuple)):
        return frozenset(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("frozenset"):
            start = text.find("(")
            end = text.rfind(")")
            text = text[start + 1 : end].strip()
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1]
        if not text:
            return frozenset()
        items = [token.strip().strip("'\"") for token in text.split(",") if token.strip()]
        return frozenset(items)
    return frozenset()


def _itemset_key(items: frozenset) -> str:
    return "|".join(sorted(items))


