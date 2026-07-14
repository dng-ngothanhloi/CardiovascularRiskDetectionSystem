"""
Plot dashboards for feature importance summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "feature" not in df.columns or "scope" not in df.columns:
        raise ValueError(f"Unexpected schema in {csv_path}")
    return df


def _value_column(df: pd.DataFrame) -> str:
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
    features: list[str],
    out_path: Path,
    title: str,
) -> None:
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


def build_dashboard(
    csv_path: Path,
    out_dir: Path,
    top_n: int = 10,
    model_label: str = "rf",
) -> None:
    df = _load_summary(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_global_top(df, out_dir / f"{model_label}_global_top{top_n}.png", top_n)
    plot_fold_heatmap(df, out_dir / f"{model_label}_fold_heatmap.png")

    bp_features = ["ap_hi", "ap_lo", "pulse_pressure"]
    plot_feature_group(df, bp_features, out_dir / f"{model_label}_blood_pressure.png", "Blood Pressure Signals")

    demographic_features = ["age", "age_years", "weight", "height", "bmi", "cholesterol"]
    plot_feature_group(
        df,
        demographic_features,
        out_dir / f"{model_label}_demographic.png",
        "Demographic / Anthropometric Importance",
    )

    lifestyle_features = ["smoke", "alco", "active"]
    plot_feature_group(df, lifestyle_features, out_dir / f"{model_label}_lifestyle.png", "Lifestyle Indicators (Low importance)")


def main(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate feature importance dashboard plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        required=True,
        help="Path to feature_importance_summary.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("artifacts/Interpretation/reports/importance_plots"),
        help="Output directory for generated plots",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of top features to show in global chart",
    )
    parser.add_argument(
        "--model_label",
        type=str,
        default="rf",
        help="Short label used in plot filenames",
    )
    parser.add_argument(
        "--comparison_csv",
        type=Path,
        help="Optional comparison CSV (TabNet vs RF) to plot heatmap",
    )

    parsed = parser.parse_args(args)
    build_dashboard(parsed.summary_csv, parsed.out_dir, parsed.top_n, parsed.model_label)
    if parsed.comparison_csv:
        parsed.out_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison_heatmap(parsed.comparison_csv, parsed.out_dir / "tabnet_rf_heatmap.png")


if __name__ == "__main__":
    main()

