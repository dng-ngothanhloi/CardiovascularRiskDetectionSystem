from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_figures(
    discovery_dir: Path,
    rules_df: Optional[pd.DataFrame] = None,
    top_k: int = 20,
    target_col: str = "cardio",
    random_seed: int = 42,
) -> None:
    """
    Generate rule graph and feature support visualisations.
    """
    figures_dir = discovery_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if rules_df is None:
        rules_path = discovery_dir / "apriori" / "rules_pruned.csv"
        if not rules_path.exists():
            logger.warning("Cannot generate figures; rules file missing at %s", rules_path)
            return
        rules_df = pd.read_csv(rules_path)
        if not rules_df.empty:
            rules_df["antecedents"] = rules_df["antecedents"].apply(_ensure_frozenset)
            rules_df["consequents"] = rules_df["consequents"].apply(_ensure_frozenset)

    if rules_df is None or rules_df.empty:
        logger.warning("No rules available for visualisation.")
        return

    top_rules = _select_top_rules(rules_df, top_k)
    if top_rules.empty:
        logger.warning("Top rules selection returned empty; skipping graph.")
    else:
        _plot_rule_graph(
            top_rules=top_rules,
            output_path=figures_dir / "rule_graph_topK.png",
            target_col=target_col,
            random_seed=random_seed,
        )

    _plot_feature_support(
        rules_df=rules_df,
        output_path=figures_dir / "feature_support_bar.png",
        target_col=target_col,
        top_n=20,
    )


def _select_top_rules(rules_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if "rank_score" in rules_df.columns and not rules_df["rank_score"].isna().all():
        return rules_df.sort_values(by="rank_score", ascending=False).head(top_k).reset_index(drop=True)
    return rules_df.sort_values(by="lift", ascending=False).head(top_k).reset_index(drop=True)


def _plot_rule_graph(
    top_rules: pd.DataFrame,
    output_path: Path,
    target_col: str,
    random_seed: int,
) -> None:
    edge_metrics: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)
    for _, row in top_rules.iterrows():
        antecedents = _ensure_frozenset(row["antecedents"])
        lift = float(row.get("lift", 1.0))
        confidence = float(row.get("confidence", 0.0))
        for feature in antecedents:
            if feature == target_col:
                continue
            edge_metrics[(feature, target_col)].append((lift, confidence))

    if not edge_metrics:
        logger.warning("No edges to plot for rule graph.")
        return

    graph = nx.DiGraph()
    for (source, target), metrics in edge_metrics.items():
        lifts, confidences = zip(*metrics)
        graph.add_edge(
            source,
            target,
            lift=float(np.mean(lifts)),
            confidence=float(np.mean(confidences)),
            count=len(metrics),
        )

    pos = nx.spring_layout(graph, seed=random_seed)
    fig, ax = plt.subplots(figsize=(10, 8))

    node_colors = ["#1f77b4" if node != target_col else "#d62728" for node in graph.nodes]
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=900, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=10, ax=ax)

    lifts = [graph.edges[edge]["lift"] for edge in graph.edges]
    confidences = [graph.edges[edge]["confidence"] for edge in graph.edges]

    widths = [2.0 + 2.0 * (lift - min(lifts)) / (max(lifts) - min(lifts) + 1e-9) for lift in lifts]
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(confidences), max(confidences) if confidences else 1.0)
    colors = [cmap(norm(conf)) for conf in confidences]

    nx.draw_networkx_edges(
        graph,
        pos,
        arrowstyle="->",
        arrowsize=15,
        width=widths,
        edge_color=colors,
        ax=ax,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, label="Confidence")
    ax.set_title(f"Top-{len(top_rules)} Rules Directed Graph", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved rule graph to %s", output_path)


def _plot_feature_support(
    rules_df: pd.DataFrame,
    output_path: Path,
    target_col: str,
    top_n: int,
) -> None:
    counter: Counter[str] = Counter()
    for _, row in rules_df.iterrows():
        antecedents = _ensure_frozenset(row["antecedents"])
        for feature in antecedents:
            if feature == target_col:
                continue
            counter[feature] += 1

    if not counter:
        logger.warning("No antecedent features found; skipping support bar chart.")
        return

    top_features = counter.most_common(min(top_n, len(counter)))
    labels, counts = zip(*top_features)

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.4)))
    y_positions = np.arange(len(labels))
    ax.barh(y_positions, counts, color="#1f77b4")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Rule Count")
    ax.set_title("Antecedent Feature Frequency")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved feature support bar chart to %s", output_path)


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


