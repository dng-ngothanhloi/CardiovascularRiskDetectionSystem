from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def generate_report(
    discovery_dir: Path,
    config: Dict[str, object],
    rules_df: Optional[pd.DataFrame] = None,
    cohort_rules_df: Optional[pd.DataFrame] = None,
    top_n: int = 20,
) -> Path:
    """
    Generate the Markdown summary report for the discovery layer.
    """
    reports_dir = discovery_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if rules_df is None:
        rules_path = discovery_dir / "apriori" / "rules_pruned.csv"
        if not rules_path.exists():
            logger.warning("No rules available for report; expected at %s", rules_path)
            rules_df = pd.DataFrame()
        else:
            rules_df = pd.read_csv(rules_path)

    if not rules_df.empty:
        rules_df["antecedents"] = rules_df["antecedents"].apply(_ensure_frozenset)
        rules_df["consequents"] = rules_df["consequents"].apply(_ensure_frozenset)

    if cohort_rules_df is None:
        cohort_path = discovery_dir / "apriori" / "rules_by_cohort.csv"
        if cohort_path.exists():
            cohort_rules_df = pd.read_csv(cohort_path)
            if not cohort_rules_df.empty:
                cohort_rules_df["antecedents"] = cohort_rules_df["antecedents"].apply(_ensure_frozenset)
                cohort_rules_df["consequents"] = cohort_rules_df["consequents"].apply(_ensure_frozenset)

    # Load itemsets metadata and binarized itemset for methodology section
    itemsets_meta = None
    itemsets_meta_path = discovery_dir / "itemsets" / "meta.json"
    if itemsets_meta_path.exists():
        with open(itemsets_meta_path, "r") as f:
            itemsets_meta = json.load(f)
    
    binarized_df = None
    binarized_path = discovery_dir / "itemsets" / "binarized_itemset.csv"
    if binarized_path.exists():
        try:
            # Load sample to get distribution without loading full file
            binarized_sample = pd.read_csv(binarized_path, nrows=10000)  # Sample for distribution
            binarized_df = binarized_sample
        except Exception as e:
            logger.warning("Could not load binarized itemset: %s", e)

    content_lines = []
    content_lines.append("# L3 Discovery Summary\n")
    content_lines.append(_render_methodology_section(config, itemsets_meta, binarized_df))
    content_lines.append(_render_settings_section(config))
    content_lines.append(_render_overall_statistics_section(rules_df))
    content_lines.append(_render_top_rules_section(rules_df, top_n))
    content_lines.append(_render_best_and_lowest_confidence_rules(rules_df))
    content_lines.append(_render_feature_frequency_section(rules_df))
    content_lines.append(_render_figures_section())
    content_lines.append(_render_cohort_section(config.get("cohort", {}), cohort_rules_df, top_n))

    report_path = reports_dir / "L3_summary.md"
    report_path.write_text("\n".join(content_lines))
    logger.info("Saved discovery report to %s", report_path)
    return report_path


def _render_methodology_section(
    config: Dict[str, object],
    itemsets_meta: Optional[Dict[str, object]],
    binarized_df: Optional[pd.DataFrame],
) -> str:
    """
    Render Section 1: Methodology & Settings
    Includes: Discovery Pipeline Configuration and Data Quality
    """
    lines = ["## 1. Methodology & Settings\n"]
    
    # Section 1.1: Discovery Pipeline Configuration
    lines.append("### 1.1. Discovery Pipeline Configuration\n")
    
    attention_cfg = config.get("attention_to_item", {})
    apriori_cfg = config.get("apriori", {})
    post_cfg = config.get("postprocess", {})
    bootstrap_cfg = post_cfg.get("stability_bootstrap", {})
    
    # Input Data
    lines.append("**Input Data:**")
    if itemsets_meta:
        n_samples = itemsets_meta.get("n_samples", "-")
        n_features = itemsets_meta.get("n_features", "-")
        sources = itemsets_meta.get("sources", [])
        if sources:
            source_desc = f"TabNet attention masks từ {len(sources)}-fold cross-validation"
        else:
            source_desc = "TabNet attention masks"
        lines.append(f"- **Source:** {source_desc}")
        lines.append(f"- **Total samples:** {n_samples:,}")
        lines.append(f"- **Features:** {n_features} features (demographic, clinical, lifestyle)")
    else:
        lines.append("- **Source:** TabNet attention masks")
        lines.append("- **Total samples:** N/A")
        lines.append("- **Features:** N/A")
    lines.append("- **Target:** `cardio` (binary: 0=no disease, 1=disease)")
    lines.append("")
    
    # Attention-to-Itemset Conversion
    lines.append("**Attention-to-Itemset Conversion:**")
    agg = attention_cfg.get("agg", "-")
    norm = attention_cfg.get("norm", "-")
    tau = itemsets_meta.get("tau", attention_cfg.get("tau", "-")) if itemsets_meta else attention_cfg.get("tau", "-")
    min_active = itemsets_meta.get("min_active_features", attention_cfg.get("min_active_features", "-")) if itemsets_meta else attention_cfg.get("min_active_features", "-")
    
    agg_desc = "Mean across decision steps" if agg == "mean" else agg
    norm_desc = "Per-feature max normalization" if norm == "per_feature_max" else norm
    tau_desc = f"{tau} (percentile {tau[1:] if isinstance(tau, str) and tau.startswith('p') else tau}) - dynamic threshold per feature" if tau else "-"
    
    lines.append(f"- **Aggregation:** {agg_desc}")
    lines.append(f"- **Normalization:** {norm_desc}")
    lines.append(f"- **Threshold (τ):** {tau_desc}")
    lines.append(f"- **Min active features:** {min_active}")
    lines.append("")
    
    # Apriori Algorithm Settings
    lines.append("**Apriori Algorithm Settings:**")
    min_support = apriori_cfg.get("min_support", "-")
    min_confidence = apriori_cfg.get("min_threshold", "-")
    max_len = apriori_cfg.get("max_len", "-")
    metric = apriori_cfg.get("metric", "confidence")
    
    lines.append(f"- **Min support:** {min_support} ({float(min_support)*100:.0f}% of samples)" if isinstance(min_support, (int, float)) else f"- **Min support:** {min_support}")
    lines.append(f"- **Min confidence:** {min_confidence} ({float(min_confidence)*100:.0f}%)" if isinstance(min_confidence, (int, float)) else f"- **Min confidence:** {min_confidence}")
    lines.append(f"- **Max length:** {max_len} (maximum {max_len} items per rule)" if isinstance(max_len, (int, float)) else f"- **Max length:** {max_len}")
    lines.append(f"- **Metric:** {metric.capitalize()}")
    lines.append("")
    
    # Post-processing
    lines.append("**Post-processing:**")
    min_lift = post_cfg.get("min_lift", "-")
    redundancy = post_cfg.get("redundancy", True)
    bootstrap_enabled = bootstrap_cfg.get("enabled", False)
    top_k = post_cfg.get("topk", "-")
    
    lines.append(f"- **Min lift:** {min_lift} (rules must have at least {float(min_lift)*100-100:.0f}% stronger association than random)" if isinstance(min_lift, (int, float)) else f"- **Min lift:** {min_lift}")
    lines.append(f"- **Redundancy pruning:** {'Enabled' if redundancy else 'Disabled'}")
    lines.append(f"- **Bootstrap stability:** {'Enabled' if bootstrap_enabled else 'Disabled'} (for faster execution)")
    lines.append(f"- **Top-K:** {top_k} rules retained" if top_k != "-" else f"- **Top-K:** {top_k}")
    lines.append("")
    
    # Section 1.2: Data Quality
    lines.append("### 1.2. Data Quality\n")
    
    if binarized_df is not None and not binarized_df.empty:
        # Get shape from meta or estimate from sample
        if itemsets_meta:
            n_samples = itemsets_meta.get("n_samples", len(binarized_df))
            n_cols = len(binarized_df.columns)
        else:
            n_samples = len(binarized_df)
            n_cols = len(binarized_df.columns)
        
        lines.append("**Binarized Itemset:**")
        lines.append(f"- **Shape:** {n_samples:,} samples × {n_cols} columns ({itemsets_meta.get('n_features', n_cols-4) if itemsets_meta else n_cols-4} features + metadata)" if itemsets_meta else f"- **Shape:** {n_samples:,} samples × {n_cols} columns")
        
        # Cardio distribution
        if "cardio" in binarized_df.columns:
            cardio_counts = binarized_df["cardio"].value_counts().sort_index()
            cardio_0 = cardio_counts.get(0, 0)
            cardio_1 = cardio_counts.get(1, 0)
            total = cardio_0 + cardio_1
            if total > 0:
                pct_0 = cardio_0 / total * 100
                pct_1 = cardio_1 / total * 100
                lines.append("- **Cardio distribution:**")
                lines.append(f"  - `cardio=0`: {cardio_0:,} ({pct_0:.1f}%)")
                lines.append(f"  - `cardio=1`: {cardio_1:,} ({pct_1:.1f}%)")
                
                # Class balance
                if abs(pct_0 - pct_1) < 1.0:
                    lines.append("- **Class balance:** [OK] Perfectly balanced (50/50)")
                else:
                    lines.append(f"- **Class balance:** [WARN] Imbalanced ({pct_0:.1f}% / {pct_1:.1f}%)")
        
        # Feature Coverage
        lines.append("")
        lines.append("**Feature Coverage:**")
        if itemsets_meta:
            feature_names = itemsets_meta.get("feature_names", [])
            thresholds = itemsets_meta.get("thresholds", [])
            lines.append(f"- All {len(feature_names)} features successfully binarized")
            if thresholds:
                min_thresh = min(thresholds)
                max_thresh = max(thresholds)
                lines.append(f"- Threshold values range from {min_thresh:.2f} to {max_thresh:.2f} (p80 percentiles)")
        
        # Missing values
        missing_count = binarized_df.isna().sum().sum()
        if missing_count == 0:
            lines.append("- No missing values in binarized itemset")
        else:
            lines.append(f"- [WARN] {missing_count} missing values found in binarized itemset")
    else:
        lines.append("**Binarized Itemset:**")
        lines.append("- Data not available for quality assessment")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    return "\n".join(lines)


def _render_settings_section(config: Dict[str, object]) -> str:
    attention_cfg = config.get("attention_to_item", {})
    apriori_cfg = config.get("apriori", {})
    post_cfg = config.get("postprocess", {})
    bootstrap_cfg = post_cfg.get("stability_bootstrap", {})
    cohort_cfg = config.get("cohort", {})

    lines = ["## Settings\n"]
    settings_table = [
        ("Tau", str(attention_cfg.get("tau", "-"))),
        ("Aggregation", str(attention_cfg.get("agg", "-"))),
        ("Normalisation", str(attention_cfg.get("norm", "-"))),
        ("Min Active Features", str(attention_cfg.get("min_active_features", "-"))),
        ("Apriori Min Support", str(apriori_cfg.get("min_support", "-"))),
        ("Apriori Min Confidence", str(apriori_cfg.get("min_threshold", "-"))),
        ("Apriori Max Len", str(apriori_cfg.get("max_len", "-"))),
        ("Postprocess Min Lift", str(post_cfg.get("min_lift", "-"))),
        ("Postprocess Min Confidence", str(post_cfg.get("min_confidence", "-"))),
        ("Postprocess Min Support", str(post_cfg.get("min_support", "-"))),
        ("Redundancy Pruning", "Enabled" if post_cfg.get("redundancy", True) else "Disabled"),
        (
            "Bootstrap",
            f"Enabled (n={bootstrap_cfg.get('n_boot', '-')}, frac={bootstrap_cfg.get('sample_frac', '-')})"
            if bootstrap_cfg.get("enabled", False)
            else "Disabled",
        ),
        (
            "Cohort Mining",
            "Enabled" if cohort_cfg.get("enabled", False) else "Disabled",
        ),
    ]

    lines.append("| Setting | Value |")
    lines.append("| --- | --- |")
    for name, value in settings_table:
        lines.append(f"| {name} | {value} |")
    lines.append("")
    return "\n".join(lines)


def _render_overall_statistics_section(rules_df: pd.DataFrame) -> str:
    """
    Render Section 2.1: Overall Statistics
    """
    lines = ["## 2. Discovered Rules Analysis\n"]
    lines.append("### 2.1. Overall Statistics\n")
    
    if rules_df.empty:
        lines.append("_No rules available for statistics._\n")
        return "\n".join(lines)
    
    # Calculate statistics
    total_rules = len(rules_df)
    mean_support = rules_df["support"].mean()
    mean_confidence = rules_df["confidence"].mean()
    mean_lift = rules_df["lift"].mean()
    min_lift = rules_df["lift"].min()
    max_lift = rules_df["lift"].max()
    
    # Create table
    lines.append("| Metric | Value | Interpretation |")
    lines.append("|--------|-------|----------------|")
    lines.append(f"| **Total Rules** | {total_rules} | Rules passing all filters |")
    lines.append(f"| **Mean Support** | {mean_support:.3f} | {mean_support*100:.1f}% of samples on average |")
    lines.append(f"| **Mean Confidence** | {mean_confidence:.3f} | {mean_confidence*100:.1f}% confidence on average |")
    lines.append(f"| **Mean Lift** | {mean_lift:.3f} | {(mean_lift-1)*100:.1f}% stronger than random |")
    lines.append(f"| **Min Lift** | {min_lift:.3f} | All rules have lift > 1.2 |")
    lines.append(f"| **Max Lift** | {max_lift:.3f} | Strongest rule has {(max_lift-1)*100:.1f}% lift |")
    lines.append("")
    
    return "\n".join(lines)


def _render_top_rules_section(rules_df: pd.DataFrame, top_n: int) -> str:
    lines = ["## Top Rules\n"]
    if rules_df.empty:
        lines.append("_No rules satisfied the pruning criteria._\n")
        return "\n".join(lines)

    top_rules = _select_top_rules(rules_df, top_n)

    columns = [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction",
        "rank_score",
        "stability",
    ]
    lines.append(
        "| Antecedents | Consequents | Support | Confidence | Lift | Leverage | Conviction | Rank Score | Stability |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    for _, row in top_rules.iterrows():
        antecedents = _format_itemset(row.get("antecedents"))
        consequents = _format_itemset(row.get("consequents"))
        support = _format_metric(row.get("support"))
        confidence = _format_metric(row.get("confidence"))
        lift = _format_metric(row.get("lift"))
        leverage = _format_metric(row.get("leverage"))
        conviction = _format_metric(row.get("conviction"))
        rank_score = _format_metric(row.get("rank_score"))
        stability = "-" if pd.isna(row.get("stability")) else _format_metric(row.get("stability"))
        lines.append(
            f"| {antecedents} | {consequents} | {support} | {confidence} | {lift} | {leverage} | "
            f"{conviction} | {rank_score} | {stability} |"
        )

    lines.append("")
    return "\n".join(lines)


def _render_best_and_lowest_confidence_rules(rules_df: pd.DataFrame) -> str:
    """
    Render sections for Top 5 Best Rules and Top 5 Rules with Lowest Confidence.
    """
    lines = []
    
    if rules_df.empty:
        return "\n".join(lines)
    
    # Top 5 Best Rules (by rank_score or lift)
    lines.append("## Top 5 Rules Tốt Nhất\n")
    if "rank_score" in rules_df.columns and not rules_df["rank_score"].isna().all():
        best_rules = rules_df.sort_values("rank_score", ascending=False).head(5)
        sort_metric = "rank_score"
    else:
        best_rules = rules_df.sort_values("lift", ascending=False).head(5)
        sort_metric = "lift"
    
    lines.append(f"*Được sắp xếp theo {sort_metric} (cao nhất).*\n")
    lines.append("| Rank | Antecedents | Consequents | Support | Confidence | Lift | Conviction |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    
    for idx, (_, row) in enumerate(best_rules.iterrows(), 1):
        antecedents = _format_itemset(row.get("antecedents"))
        consequents = _format_itemset(row.get("consequents"))
        support = _format_metric(row.get("support"))
        confidence = _format_metric(row.get("confidence"))
        lift = _format_metric(row.get("lift"))
        conviction = _format_metric(row.get("conviction"))
        lines.append(
            f"| {idx} | {antecedents} | {consequents} | {support} | {confidence} | {lift} | {conviction} |"
        )
    lines.append("")
    
    # Top 5 Rules with Lowest Confidence (cần kiểm tra lâm sàng)
    lines.append("## Top 5 Rules Có Confidence Thấp Nhất (Cần Kiểm Tra Lâm Sàng)\n")
    lowest_conf_rules = rules_df.sort_values("confidence", ascending=True).head(5)
    
    lines.append("*Các rules này có confidence thấp, cần được xem xét và kiểm tra lâm sàng kỹ lưỡng trước khi áp dụng.*\n")
    lines.append("| Rank | Antecedents | Consequents | Support | Confidence | Lift | Conviction | Ghi Chú |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    
    for idx, (_, row) in enumerate(lowest_conf_rules.iterrows(), 1):
        antecedents = _format_itemset(row.get("antecedents"))
        consequents = _format_itemset(row.get("consequents"))
        support = _format_metric(row.get("support"))
        confidence = row.get("confidence")
        confidence_str = _format_metric(confidence)
        lift = _format_metric(row.get("lift"))
        conviction = _format_metric(row.get("conviction"))
        
        # Add warning note based on confidence level
        if confidence < 0.65:
            note = "[WARN] Confidence < 65% - Cần kiểm tra lâm sàng"
        elif confidence < 0.70:
            note = "[WARN] Confidence < 70% - Nên xem xét cẩn thận"
        else:
            note = "Confidence tương đối tốt"
        
        lines.append(
            f"| {idx} | {antecedents} | {consequents} | {support} | {confidence_str} | {lift} | {conviction} | {note} |"
        )
    lines.append("")
    
    # Summary statistics
    lines.append("### Thống Kê Tổng Quan\n")
    lines.append(f"- **Tổng số rules:** {len(rules_df)}")
    lines.append(f"- **Mean confidence:** {rules_df['confidence'].mean():.3f}")
    lines.append(f"- **Min confidence:** {rules_df['confidence'].min():.3f}")
    lines.append(f"- **Max confidence:** {rules_df['confidence'].max():.3f}")
    lines.append(f"- **Mean lift:** {rules_df['lift'].mean():.3f}")
    lines.append(f"- **Rules với confidence < 0.65:** {len(rules_df[rules_df['confidence'] < 0.65])}")
    lines.append(f"- **Rules với confidence < 0.70:** {len(rules_df[rules_df['confidence'] < 0.70])}")
    lines.append(f"- **Rules với lift >= 1.20:** {len(rules_df[rules_df['lift'] >= 1.20])} (đã filter)\n")
    
    return "\n".join(lines)


def _render_feature_frequency_section(rules_df: pd.DataFrame) -> str:
    """
    Render Section 2.3: Feature Frequency in Rules
    Parse antecedents to extract features and count frequency
    """
    lines = ["### 2.3. Feature Frequency in Rules\n"]
    
    if rules_df.empty:
        lines.append("_No rules available for feature frequency analysis._\n")
        return "\n".join(lines)
    
    # Extract features from antecedents
    feature_counts = {}
    for _, row in rules_df.iterrows():
        antecedents = _ensure_frozenset(row.get("antecedents"))
        for feature in antecedents:
            # Clean feature name (remove any prefixes or suffixes)
            clean_feature = feature.strip().strip("'\"")
            feature_counts[clean_feature] = feature_counts.get(clean_feature, 0) + 1
    
    # Sort by frequency
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    
    lines.append("**Most Common Features in Antecedents:**")
    lines.append("")
    lines.append("| Feature | Frequency | Clinical Relevance |")
    lines.append("|---------|-----------|-------------------|")
    
    # Clinical relevance is left empty for manual filling
    for feature, count in sorted_features:
        lines.append(f"| `{feature}` | {count} rules | *[To be filled manually]* |")
    
    lines.append("")
    
    # Basic insights (can be auto-generated)
    total_rules = len(rules_df)
    lines.append("**Insights:**")
    if sorted_features:
        top_feature, top_count = sorted_features[0]
        top_pct = top_count / total_rules * 100
        lines.append(f"- **{top_feature}** xuất hiện trong {top_count}/{total_rules} rules ({top_pct:.1f}%)")
        
        if len(sorted_features) > 1:
            second_feature, second_count = sorted_features[1]
            second_pct = second_count / total_rules * 100
            lines.append(f"- **{second_feature}** xuất hiện trong {second_count}/{total_rules} rules ({second_pct:.1f}%)")
        
        # Count features appearing in multiple rules
        multi_rule_features = [f for f, c in sorted_features if c >= 3]
        if multi_rule_features:
            lines.append(f"- **{len(multi_rule_features)} features** xuất hiện trong 3+ rules: {', '.join(multi_rule_features[:5])}")
    
    lines.append("")
    lines.append("*Note: Clinical Relevance column requires domain knowledge and should be filled manually.*")
    lines.append("")
    
    return "\n".join(lines)


def _render_figures_section() -> str:
    lines = ["## Figures\n"]
    lines.append("- [Rule Graph](../figures/rule_graph_topK.png)")
    lines.append("- [Feature Support](../figures/feature_support_bar.png)\n")
    return "\n".join(lines)


def _render_cohort_section(
    cohort_cfg: Dict[str, object],
    cohort_rules_df: Optional[pd.DataFrame],
    top_n: int,
) -> str:
    lines = ["## Cohort Insights\n"]
    if not cohort_cfg.get("enabled", False):
        lines.append("_Cohort mining disabled._\n")
        return "\n".join(lines)

    if cohort_rules_df is None or cohort_rules_df.empty:
        lines.append("_No cohort-specific rules generated._\n")
        return "\n".join(lines)

    for cohort_label, group in cohort_rules_df.groupby("cohort_label"):
        lines.append(f"### Cohort: {cohort_label}\n")
        top_rules = _select_top_rules(group, top_n)
        lines.append(
            "| Antecedents | Consequents | Support | Confidence | Lift | Leverage | Conviction | Rank/Score |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, row in top_rules.iterrows():
            antecedents = _format_itemset(row.get("antecedents"))
            consequents = _format_itemset(row.get("consequents"))
            support = _format_metric(row.get("support"))
            confidence = _format_metric(row.get("confidence"))
            lift = _format_metric(row.get("lift"))
            leverage = _format_metric(row.get("leverage"))
            conviction = _format_metric(row.get("conviction"))
            rank = _format_metric(row.get("rank_score") or row.get("lift"))
            lines.append(
                f"| {antecedents} | {consequents} | {support} | {confidence} | {lift} | "
                f"{leverage} | {conviction} | {rank} |"
            )
        lines.append("")

    return "\n".join(lines)


def _select_top_rules(rules_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if "rank_score" in rules_df.columns and not rules_df["rank_score"].isna().all():
        return rules_df.sort_values("rank_score", ascending=False).head(top_n).reset_index(drop=True)
    return rules_df.sort_values("lift", ascending=False).head(top_n).reset_index(drop=True)


def _format_metric(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "-"


def _format_itemset(items) -> str:
    fs = _ensure_frozenset(items)
    if not fs:
        return "-"
    return ", ".join(sorted(fs))


def _ensure_frozenset(value) -> frozenset:
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


