"""
Generate Markdown report aggregating importance outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)


def _fmt(path: Optional[Path]) -> str:
    if not path:
        return "-"
    if isinstance(path, Path):
        return str(path)
    return str(path)


def write_importance_report(
    out_dir: Path,
    tabnet_global_csv: Optional[Path],
    tabnet_summary_csv: Optional[Path],
    cohort_csv: Optional[Path],
    rf_summary_csv: Optional[Path] = None,
    comparison_csv: Optional[Path] = None,
) -> Path:
    """Write interpretation summary linking key CSV artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "L2_importance_summary.md"

    lines = [
        "# L2 – Feature Importance Summaries",
        "",
        "## TabNet (Attention)",
        f"- attention/global_attention.csv: {_fmt(tabnet_global_csv)}",
        f"- importance/tabnet/feature_importance_summary.csv: {_fmt(tabnet_summary_csv)}",
        f"- cohorts.csv: {_fmt(cohort_csv)}",
    ]

    if rf_summary_csv:
        lines.extend(
            [
                "",
                "## Random Forest",
                f"- importance/rf/feature_importance_summary.csv: {_fmt(rf_summary_csv)}",
            ]
        )

    if comparison_csv:
        lines.extend(
            [
                "",
                "## Model Comparison",
                f"- importance/comparison_tabnet_rf.csv: {_fmt(comparison_csv)}",
            ]
        )

    with path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    LOGGER.info("Wrote importance layer summary to %s", path)
    return path



