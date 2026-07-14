"""
Utility package exposing legacy helpers plus submodules (io, metrics, ...).
"""

from __future__ import annotations

import importlib.util as _importlib_util
import sys as _sys
from pathlib import Path as _Path

# Load legacy module (src/utils.py) under temporary name and re-export symbols.
_legacy_path = _Path(__file__).resolve().parent.parent / "utils.py"
_legacy_spec = _importlib_util.spec_from_file_location("src._utils_legacy", _legacy_path)
_legacy_module = _importlib_util.module_from_spec(_legacy_spec)  # type: ignore[arg-type]
assert _legacy_spec.loader is not None
_legacy_spec.loader.exec_module(_legacy_module)  # type: ignore[attr-defined]
_sys.modules.setdefault("src._utils_legacy", _legacy_module)

for _name in dir(_legacy_module):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_legacy_module, _name)

# Re-export utility submodules
from .io import read_oof_csv, write_json, read_json, save_pickle, load_pickle  # noqa: F401
from .metrics import brier_score, ece_score, reliability_bins  # noqa: F401

__all__ = [name for name in globals() if not name.startswith("_")]

