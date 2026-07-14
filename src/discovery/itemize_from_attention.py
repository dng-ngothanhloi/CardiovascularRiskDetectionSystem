from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_attention_matrices(
    attn_dir: Path,
    agg: str = "mean",
) -> Tuple[
    np.ndarray,
    List[int],
    Dict[int, int],
    List[str],
    Optional[np.ndarray],
    Optional[List[str]],
]:
    """
    Load per-fold attention matrices produced by L1/L2.

    Parameters
    ----------
    attn_dir:
        Directory containing `fold_*/attention_per_sample.npy`.

    Returns
    -------
    stacked: np.ndarray
        Concatenated attention arrays across folds.
    fold_assignments: list[int]
        Fold index assigned to each row in the stacked matrix.
    fold_counts: dict[int, int]
        Number of samples contributed by each fold.
    sources: list[str]
        List of file paths that were loaded.
    ids: Optional[np.ndarray]
        Concatenated per-sample identifiers if provided in the artifacts.
    feature_names: Optional[List[str]]
        Feature name ordering inferred directly from the artifacts (if available).
    """
    if not attn_dir.exists():
        raise FileNotFoundError(f"Attention directory not found: {attn_dir}")

    arrays: List[np.ndarray] = []
    fold_assignments: List[int] = []
    fold_counts: Dict[int, int] = {}
    sources: List[str] = []
    ids_list: List[np.ndarray] = []
    feature_names_hint: Optional[List[str]] = None

    fold_dirs = sorted(
        [p for p in attn_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")],
        key=lambda p: p.name,
    )

    if not fold_dirs:
        raise ValueError(f"No fold_* directories found inside {attn_dir}")

    for fold_dir in fold_dirs:
        npy_path = fold_dir / "attention_per_sample.npy"
        if not npy_path.exists():
            logger.warning("Skipping %s: %s missing", fold_dir.name, npy_path.name)
            continue

        try:
            raw = np.load(npy_path, allow_pickle=False)
        except ValueError as exc:
            if "Object arrays cannot be loaded" in str(exc):
                logger.warning(
                    "Reloading %s with allow_pickle=True due to object dtype array.", npy_path.name
                )
                raw = np.load(npy_path, allow_pickle=True)
            else:
                raise

        arr, ids, feature_names = _coerce_attention_artifact(raw, npy_path)
        arr = _apply_step_aggregation(arr, agg)

        if arr.ndim < 2:
            raise ValueError(f"Attention array expected >=2 dimensions, got shape {arr.shape}")

        arrays.append(arr)
        fold_idx = _parse_fold_index(fold_dir.name)
        fold_assignments.extend([fold_idx] * arr.shape[0])
        fold_counts[fold_idx] = arr.shape[0]
        sources.append(str(npy_path))
        if ids is not None:
            ids_list.append(ids)
        if feature_names_hint is None and feature_names:
            feature_names_hint = list(feature_names)

    if not arrays:
        raise ValueError(f"No attention_per_sample.npy files loaded from {attn_dir}")

    stacked = np.concatenate(arrays, axis=0)
    ids_concat = np.concatenate(ids_list, axis=0) if ids_list else None
    return stacked, fold_assignments, fold_counts, sources, ids_concat, feature_names_hint


def reduce_and_binarize(
    attention: np.ndarray, feature_names: Sequence[str], cfg: Dict[str, object]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Reduce attention tensors to 2D, normalise, and binarise into an itemset matrix.

    Parameters
    ----------
    attention:
        Attention array with shape (n_samples, n_features[, n_steps]).
    feature_names:
        Ordered feature names corresponding to the feature dimension.
    cfg:
        Configuration dictionary with keys `agg`, `norm`, `tau`, `min_active_features`.

    Returns
    -------
    items_df:
        DataFrame of shape (n_samples, n_features) with 0/1 entries.
    reduced:
        Reduced continuous attention (after aggregation across steps).
    thresholds:
        Per-feature thresholds applied for binarisation.
    used_feature_names:
        Feature names aligned with the returned DataFrame columns (may differ from input size).
    """

    if attention.ndim == 3:
        attention = _ensure_feature_axis(attention, len(feature_names))
        agg = str(cfg.get("agg", "mean")).lower()
        if agg == "mean":
            reduced = attention.mean(axis=2)
        elif agg == "max":
            reduced = attention.max(axis=2)
        else:
            raise ValueError(f"Unsupported aggregation method: {cfg.get('agg')}")
    elif attention.ndim == 2:
        reduced = attention
    else:
        raise ValueError(f"Unsupported attention ndim={attention.ndim}")

    reduced = np.asarray(reduced, dtype=np.float64)
    used_feature_names = list(feature_names)

    if reduced.shape[1] != len(used_feature_names):
        logger.warning(
            "Feature name count (%s) does not match attention dimension (%s); using fallback names.",
            len(used_feature_names),
            reduced.shape[1],
        )
        used_feature_names = [f"feature_{idx}" for idx in range(reduced.shape[1])]

    norm = str(cfg.get("norm", "none")).lower()
    normed = reduced.copy()

    if norm == "per_feature_max":
        max_vals = normed.max(axis=0)
        max_vals[max_vals == 0] = 1.0
        normed = normed / max_vals
    elif norm == "zscore":
        mean = normed.mean(axis=0)
        std = normed.std(axis=0)
        std[std == 0] = 1.0
        normed = (normed - mean) / std
    elif norm == "none":
        pass
    else:
        raise ValueError(f"Unknown normalisation option: {cfg.get('norm')}")

    tau_value = cfg.get("tau", 0.5)
    if isinstance(tau_value, str):
        tau_value = tau_value.strip()
        if tau_value.lower().startswith("p"):
            percentile = float(tau_value[1:])
            thresholds = np.percentile(normed, percentile, axis=0)
        else:
            thresholds = np.full(normed.shape[1], float(tau_value))
    else:
        thresholds = np.full(normed.shape[1], float(tau_value))

    binarized = (normed >= thresholds).astype(np.int8)

    min_active = int(cfg.get("min_active_features", 0) or 0)
    if min_active > 0:
        row_sums = binarized.sum(axis=1)
        for idx, count in enumerate(row_sums):
            if count >= min_active:
                continue
            # Fallback to top-k features based on reduced attention scores.
            top_indices = np.argsort(reduced[idx])[::-1][:min_active]
            binarized[idx, top_indices] = 1

    items_df = pd.DataFrame(binarized, columns=used_feature_names)
    return items_df, reduced, thresholds, used_feature_names


def attach_labels(
    df_items: pd.DataFrame,
    y_series: Optional[pd.Series],
    id_series: Optional[pd.Series] = None,
    target_col: str = "cardio",
) -> pd.DataFrame:
    """
    Attach optional label (`y_true`) and identifier columns to the itemset DataFrame.

    Parameters
    ----------
    df_items:
        Itemset DataFrame containing only feature indicator columns.
    y_series:
        Optional series of labels aligned with df_items.
    id_series:
        Optional series of identifiers aligned with df_items.
    target_col:
        Column name to use for the binary target appended to df_items.

    Returns
    -------
    pd.DataFrame
        DataFrame with optional `id`, `y_true`, and target columns appended.
    """

    df_items = df_items.copy()
    if id_series is not None:
        if len(id_series) != len(df_items):
            logger.warning(
                "ID series length (%s) does not match itemset rows (%s); discarding IDs.",
                len(id_series),
                len(df_items),
            )
        else:
            df_items["id"] = id_series.values

    if y_series is not None:
        if len(y_series) != len(df_items):
            logger.warning(
                "Label series length (%s) does not match itemset rows (%s); skipping labels.",
                len(y_series),
                len(df_items),
            )
        else:
            df_items["y_true"] = y_series.values
            df_items[target_col] = y_series.astype(int).values

    return df_items


def _parse_fold_index(name: str) -> int:
    try:
        return int(name.split("_")[-1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Cannot parse fold index from directory name: {name}") from exc


def _ensure_feature_axis(attention: np.ndarray, expected_features: int) -> np.ndarray:
    """
    Ensure that attention array is ordered as (n_samples, n_features, n_steps).
    """
    if attention.ndim != 3:
        raise ValueError("Expected a 3D attention tensor.")

    if expected_features and attention.shape[1] == expected_features:
        return attention
    if expected_features and attention.shape[2] == expected_features:
        return attention.swapaxes(1, 2)

    if attention.shape[1] >= attention.shape[2]:
        return attention
    return attention.swapaxes(1, 2)


def _infer_feature_names(
    fe_dir: Path,
    global_attention_csv: Path,
    n_features: int,
    preferred: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Infer feature names using FE artifacts or global attention outputs.
    """
    if preferred and len(preferred) == n_features:
        return [str(name) for name in preferred]
    # Priority 1: CSV metadata.
    csv_path = fe_dir / "selected_features_tabnet.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if "feature" in df.columns:
                names = df["feature"].astype(str).tolist()
            else:
                names = df.iloc[:, 0].astype(str).tolist()
            if len(names) == n_features:
                return names
            logger.warning(
                "selected_features_tabnet.csv contained %s features, expected %s.",
                len(names),
                n_features,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to read %s: %s", csv_path, exc)

    # Priority 2: JSON metadata (current artifact format).
    json_path = fe_dir / "selected_features_tabnet.json"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
            if isinstance(data, dict) and "kept_features" in data:
                names = [str(x) for x in data["kept_features"]]
                if len(names) == n_features:
                    return names
                logger.warning(
                    "selected_features_tabnet.json listed %s features, expected %s.",
                    len(names),
                    n_features,
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to read %s: %s", json_path, exc)

    # Priority 3: Global attention CSV (take first fold order).
    if global_attention_csv.exists():
        try:
            df = pd.read_csv(global_attention_csv)
            if "feature" in df.columns:
                if "fold" in df.columns and not df.empty:
                    first_fold = df["fold"].iloc[0]
                    names = df[df["fold"] == first_fold]["feature"].astype(str).tolist()
                else:
                    names = df["feature"].astype(str).tolist()
                # Deduplicate while preserving order.
                deduped = list(dict.fromkeys(names))
                if len(deduped) == n_features:
                    return deduped
                logger.warning(
                    "global_attention.csv provided %s unique features, expected %s.",
                    len(deduped),
                    n_features,
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to parse %s: %s", global_attention_csv, exc)

    logger.warning("Falling back to generic feature names for %s features.", n_features)
    return [f"feature_{idx}" for idx in range(n_features)]


def build_itemset(
    cfg: Dict[str, object],
    paths_cfg: Dict[str, str],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Orchestrate loading, processing, and persisting the binarised itemset.

    Parameters
    ----------
    cfg:
        Attention-to-item configuration dictionary.
    paths_cfg:
        Path configuration dictionary (must include attention_dir, fe_dir, tabnet_model_dir).
    output_dir:
        Base discovery output directory (`artifacts/Discovery`).

    Returns
    -------
    pd.DataFrame
        Binarised itemset with optional labels.
    """
    attn_dir = Path(paths_cfg["attention_dir"]).resolve()
    fe_dir = Path(paths_cfg.get("fe_dir", output_dir)).resolve()

    tabnet_model_dir_cfg = paths_cfg.get("tabnet_model_dir")
    tabnet_model_dir = Path(tabnet_model_dir_cfg).resolve() if tabnet_model_dir_cfg else None

    itemset_dir = output_dir / "itemsets"
    itemset_dir.mkdir(parents=True, exist_ok=True)

    (
        attention,
        fold_assignments,
        fold_counts,
        sources,
        attention_ids,
        feature_names_hint,
    ) = load_attention_matrices(attn_dir, agg=str(cfg.get("agg", "mean")))
    feature_dim = attention.shape[1] if attention.ndim == 2 else max(attention.shape[1], attention.shape[2])
    feature_names = _infer_feature_names(
        fe_dir,
        attn_dir / "global_attention.csv",
        feature_dim,
        preferred=feature_names_hint,
    )

    items_df, reduced, thresholds, used_feature_names = reduce_and_binarize(attention, feature_names, cfg)

    if attention_ids is not None:
        items_df = attach_labels(items_df, y_series=None, id_series=pd.Series(attention_ids), target_col="cardio")

    labels_df = _load_labels(tabnet_model_dir, fold_assignments, fold_counts)
    if labels_df is not None:
        items_df = attach_labels(
            items_df,
            y_series=labels_df.get("y_true"),
            id_series=None if attention_ids is not None else labels_df.get("id"),
            target_col="cardio",
        )
        if len(labels_df) == len(items_df) and "fold" in labels_df.columns:
            items_df["fold"] = labels_df["fold"].values

    itemset_path = itemset_dir / "binarized_itemset.csv"
    items_df.to_csv(itemset_path, index=False)

    meta = {
        "tau": cfg.get("tau"),
        "normalisation": cfg.get("norm"),
        "aggregation": cfg.get("agg"),
        "min_active_features": cfg.get("min_active_features"),
        "feature_names": used_feature_names,
        "thresholds": thresholds.tolist() if isinstance(thresholds, np.ndarray) else list(thresholds),
        "n_samples": int(items_df.shape[0]),
        "n_features": int(len(used_feature_names)),
        "fold_counts": fold_counts,
        "sources": sources,
    }

    meta_path = itemset_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info("Saved binarised itemset to %s", itemset_path)
    logger.info("Itemset metadata saved to %s", meta_path)
    return items_df


def _coerce_attention_artifact(
    raw: object,
    source_path: Path,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Sequence[str]]]:
    """
    Handle historical attention exports that stored dictionaries with metadata.
    """
    ids: Optional[np.ndarray] = None
    feature_names: Optional[Sequence[str]] = None

    if isinstance(raw, dict):
        attention = np.asarray(raw.get("attention"))
        ids_val = raw.get("ids")
        if ids_val is not None:
            ids = np.asarray(ids_val)
        feature_names = raw.get("feature_names")
        return attention, ids, feature_names

    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        obj = raw.item()
        return _coerce_attention_artifact(obj, source_path)

    if isinstance(raw, list):
        raw = np.asarray(raw)

    if isinstance(raw, np.ndarray) and raw.dtype == object:
        if raw.shape == ():
            obj = raw.item()
            return _coerce_attention_artifact(obj, source_path)
        if raw.ndim == 1 and all(hasattr(elem, "shape") for elem in raw):
            stacked = np.stack([np.asarray(elem) for elem in raw], axis=0)
            return stacked, ids, feature_names
        raise ValueError(
            f"Unsupported attention artifact structure in {source_path} with object dtype shape {raw.shape}"
        )

    return np.asarray(raw), ids, feature_names


def _apply_step_aggregation(attention: np.ndarray, agg: str) -> np.ndarray:
    """
    Reduce attention tensors with variable decision steps to 2D.
    """
    if attention.ndim == 2:
        return np.asarray(attention)
    if attention.ndim != 3:
        raise ValueError(f"Unsupported attention tensor with ndim={attention.ndim}")

    agg_lower = agg.lower()
    if agg_lower == "mean":
        return attention.mean(axis=2)
    if agg_lower == "max":
        return attention.max(axis=2)

    raise ValueError(f"Unsupported aggregation method: {agg}")


def _load_labels(
    model_dir: Optional[Path], fold_assignments: Sequence[int], fold_counts: Dict[int, int]
) -> Optional[pd.DataFrame]:
    """
    Load OOF labels (and IDs) to align with attention rows.
    """
    if model_dir is None or not model_dir.exists():
        logger.warning("TabNet model directory not found (%s); skipping label attachment.", model_dir)
        return None

    csv_path = model_dir / "oof_predictions.csv"
    if not csv_path.exists():
        logger.warning("OOF predictions CSV not found (%s); skipping label attachment.", csv_path)
        return None

    oof = pd.read_csv(csv_path)
    required_cols = {"y_true"}
    if not required_cols.issubset(oof.columns):
        logger.warning("OOF CSV missing required columns %s; skipping.", required_cols)
        return None

    fold_col = "fold" if "fold" in oof.columns else None
    if fold_col is None:
        logger.warning("OOF CSV does not contain `fold`; cannot align rows robustly.")
        return None

    aligned_rows: List[pd.DataFrame] = []
    for fold_idx in sorted(fold_counts):
        count = fold_counts[fold_idx]
        fold_rows = oof[oof[fold_col] == fold_idx]
        if len(fold_rows) < count:
            logger.warning(
                "Fold %s has %s attention rows but only %s OOF rows; truncating to available rows.",
                fold_idx,
                count,
                len(fold_rows),
            )
            count = len(fold_rows)
        aligned_rows.append(fold_rows.iloc[:count])

    aligned = pd.concat(aligned_rows, axis=0, ignore_index=True)
    if len(aligned) != len(fold_assignments):
        logger.warning(
            "Aligned OOF rows (%s) do not match attention rows (%s); discarding labels.",
            len(aligned),
            len(fold_assignments),
        )
        return None

    # Ensure order matches attention stacking order.
    aligned["fold"] = list(fold_assignments)
    return aligned[["id", "fold", "y_true"]] if "id" in aligned.columns else aligned[["fold", "y_true"]]


