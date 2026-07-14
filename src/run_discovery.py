from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from .discovery import (
    apriori_runner,
    itemize_from_attention,
    rule_postprocess,
    rule_report,
    rule_visualize,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run L3 discovery pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to discovery YAML config.")
    parser.add_argument("--tau", type=str, help="Override threshold tau (numeric or percentile e.g. p80).")
    parser.add_argument("--percentile", type=str, help="Alternative way to set tau as percentile string (e.g. p80).")
    parser.add_argument("--min_support", type=float, help="Override min support.")
    parser.add_argument("--min_conf", type=float, help="Override min confidence.")
    parser.add_argument("--min_lift", type=float, help="Override min lift for post-processing.")
    parser.add_argument("--max_len", type=int, help="Override Apriori max_len.")
    parser.add_argument("--no_bootstrap", action="store_true", help="Disable stability bootstrap.")
    parser.add_argument("--cohort_file", type=str, help="Override cohort file path and enable cohort mining.")
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = cfg.copy()
    attn_cfg = cfg.setdefault("attention_to_item", {})
    apriori_cfg = cfg.setdefault("apriori", {})
    post_cfg = cfg.setdefault("postprocess", {})
    cohort_cfg = cfg.setdefault("cohort", {})

    if args.percentile:
        attn_cfg["tau"] = args.percentile
    elif args.tau:
        attn_cfg["tau"] = args.tau

    if args.min_support is not None:
        apriori_cfg["min_support"] = args.min_support
        post_cfg["min_support"] = args.min_support
    if args.min_conf is not None:
        apriori_cfg["min_threshold"] = args.min_conf
        post_cfg["min_confidence"] = args.min_conf
    if args.min_lift is not None:
        post_cfg["min_lift"] = args.min_lift
    if args.max_len is not None:
        apriori_cfg["max_len"] = args.max_len
    if args.no_bootstrap:
        post_cfg.setdefault("stability_bootstrap", {})["enabled"] = False
    if args.cohort_file:
        cohort_cfg["enabled"] = True
        cohort_cfg["file"] = args.cohort_file

    cfg["attention_to_item"] = attn_cfg
    cfg["apriori"] = apriori_cfg
    cfg["postprocess"] = post_cfg
    cfg["cohort"] = cohort_cfg
    return cfg


def resolve_paths(cfg: Dict[str, Any], base_dir: Path) -> Dict[str, Path]:
    paths_cfg = cfg.setdefault("paths", {})
    resolved = {}
    for key, value in paths_cfg.items():
        if value is None:
            continue
        path = Path(value)
        resolved[key] = path if path.is_absolute() else (base_dir / path).resolve()
    cfg["paths"] = {k: str(v) for k, v in resolved.items()}
    return resolved


def set_global_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw_cfg = load_config(config_path)
    cfg = apply_overrides(raw_cfg, args)

    base_dir = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    resolved_paths = resolve_paths(cfg, base_dir)

    discovery_dir = resolved_paths.get("discovery_dir")
    if discovery_dir is None:
        raise ValueError("Discovery directory (`paths.discovery_dir`) must be specified in the config.")
    discovery_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(cfg.get("random_seed"))

    logger.info("Running discovery pipeline with config: %s", config_path)

    items_df = itemize_from_attention.build_itemset(
        cfg=cfg["attention_to_item"],
        paths_cfg=cfg["paths"],
        output_dir=discovery_dir,
    )

    frequent_itemsets, raw_rules, cohort_rules = apriori_runner.mine_rules(
        apriori_cfg=cfg["apriori"],
        cohort_cfg=cfg.get("cohort", {}),
        discovery_dir=discovery_dir,
        items_df=items_df,
        target_col="cardio",
    )

    ranking_weights = cfg.get("ranking", {}).get("weights")
    pruned_rules = rule_postprocess.postprocess_rules(
        apriori_cfg=cfg["apriori"],
        post_cfg=cfg.get("postprocess", {}),
        discovery_dir=discovery_dir,
        items_df=items_df,
        target_col="cardio",
        random_seed=int(cfg.get("random_seed", 42)),
        ranking_weights=ranking_weights,
    )

    rule_visualize.generate_figures(
        discovery_dir=discovery_dir,
        rules_df=pruned_rules,
        top_k=int(cfg.get("postprocess", {}).get("topk", 20)),
        target_col="cardio",
        random_seed=int(cfg.get("random_seed", 42)),
    )

    report_path = rule_report.generate_report(
        discovery_dir=discovery_dir,
        config=cfg,
        rules_df=pruned_rules,
        cohort_rules_df=cohort_rules,
        top_n=int(cfg.get("postprocess", {}).get("topk", 20)),
    )

    # Persist config snapshot for reproducibility.
    snapshot_path = discovery_dir / "meta.json"
    snapshot_path.write_text(json.dumps(cfg, indent=2))
    logger.info("Saved config snapshot to %s", snapshot_path)

    logger.info("Discovery pipeline complete.")
    logger.info("Itemset -> %s", discovery_dir / "itemsets" / "binarized_itemset.csv")
    logger.info("Frequent itemsets -> %s", discovery_dir / "apriori" / "frequent_itemsets.csv")
    logger.info("Pruned rules -> %s", discovery_dir / "apriori" / "rules_pruned.csv")
    logger.info("Report -> %s", report_path)


if __name__ == "__main__":
    main()

