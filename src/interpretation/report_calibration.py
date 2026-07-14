"""
Markdown reporting helpers for calibration outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List

LOGGER = logging.getLogger(__name__)


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, int)):
        return f"{value:.6f}" if isinstance(value, float) else str(value)
    return str(value)


def write_model_calib_report_md(out_dir: Path, model_name: str, rows: List[dict]) -> Path:
    """Write per-model calibration summary markdown."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.md"

    lines = [
        f"# Calibration Report — {model_name}",
        "",
        "| Fold | Method | Brier (Pre) | ECE (Pre) | Brier (Post) | ECE (Post) | rel_pre | rel_post |",
        "|-----:|:-------|------------:|----------:|-------------:|-----------:|:--------|:---------|",
    ]

    if not rows:
        rows = []

    for row in rows:
        lines.append(
            "| {fold} | {method} | {brier_pre} | {ece_pre} | {brier_post} | {ece_post} | {rel_pre} | {rel_post} |".format(
                fold=_fmt(row.get("fold", "-")),
                method=_fmt(row.get("method", "-")).capitalize(),
                brier_pre=_fmt(row.get("brier_pre")),
                ece_pre=_fmt(row.get("ece_pre")),
                brier_post=_fmt(row.get("brier_post")),
                ece_post=_fmt(row.get("ece_post")),
                rel_pre=_fmt(row.get("rel_pre", "-")),
                rel_post=_fmt(row.get("rel_post", "-")),
            )
        )

    with path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    LOGGER.info("Wrote calibration report to %s", path)
    return path


def write_l2_summary_md(
    out_report: Path,
    aggregates: Iterable[dict],
    per_fold_rows: Iterable[dict],
    plot_paths: Dict[str, str],
) -> Path:
    """Write top-level L2 calibration summary markdown."""
    out_report = Path(out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    aggregate_lines = [
        "# L2 – Calibration & Reliability Summary",
        "",
        "## Aggregate OOF (Pre vs Post)",
        "| Model | Brier (Pre) | ECE (Pre) | Brier (Post) | ECE (Post) | ΔBrier | ΔECE |",
        "|:------|------------:|----------:|-------------:|-----------:|------:|-----:|",
    ]

    any_aggregate = False
    for row in aggregates:
        any_aggregate = True
        aggregate_lines.append(
            "| {model} | {brier_pre} | {ece_pre} | {brier_post} | {ece_post} | {delta_brier} | {delta_ece} |".format(
                model=_fmt(row.get("model", "-")),
                brier_pre=_fmt(row.get("brier_pre")),
                ece_pre=_fmt(row.get("ece_pre")),
                brier_post=_fmt(row.get("brier_post")),
                ece_post=_fmt(row.get("ece_post")),
                delta_brier=_fmt(row.get("delta_brier")),
                delta_ece=_fmt(row.get("delta_ece")),
            )
        )
    if not any_aggregate:
        aggregate_lines.append("| - | - | - | - | - | - | - |")

    fold_lines = [
        "",
        "## Per-fold",
        "| Model | Fold | Method | Brier (Pre) | ECE (Pre) | Brier (Post) | ECE (Post) | rel_pre | rel_post |",
        "|:------|-----:|:-------|------------:|----------:|-------------:|-----------:|:--------|:---------|",
    ]

    any_fold = False
    for row in per_fold_rows:
        any_fold = True
        fold_lines.append(
            "| {model} | {fold} | {method} | {brier_pre} | {ece_pre} | {brier_post} | {ece_post} | {rel_pre} | {rel_post} |".format(
                model=_fmt(row.get("model", "-")),
                fold=_fmt(row.get("fold", "-")),
                method=_fmt(row.get("method", "-")).capitalize(),
                brier_pre=_fmt(row.get("brier_pre")),
                ece_pre=_fmt(row.get("ece_pre")),
                brier_post=_fmt(row.get("brier_post")),
                ece_post=_fmt(row.get("ece_post")),
                rel_pre=_fmt(row.get("rel_pre", "-")),
                rel_post=_fmt(row.get("rel_post", "-")),
            )
        )
    if not any_fold:
        fold_lines.append("| - | - | - | - | - | - | - | - | - |")

    overlays = [
        "",
        "## Overlays",
    ]
    overlays.append(f"- plots/reliability_overlay_pre.png: {_fmt(plot_paths.get('pre', '-'))}")
    overlays.append(f"- plots/reliability_overlay_post.png: {_fmt(plot_paths.get('post', '-'))}")

    lines = aggregate_lines + fold_lines + overlays
    with out_report.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    LOGGER.info("Wrote L2 summary report to %s", out_report)
    return out_report


