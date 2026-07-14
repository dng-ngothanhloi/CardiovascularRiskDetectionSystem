"""
Integration tests for CLI modeling commands.
Tests --model tabnet and --model vae_tabnet with synthetic data.
"""
import numpy as np
import os
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _prepare_fe(dir_path: Path, n=120, d=8, seed=0):
    """Prepare synthetic FE artifacts (numpy files)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype("float32")
    y = (rng.random(n) > 0.7).astype("int64")
    
    # Simple split
    n_tr = int(n * 0.8)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save numpy files
    np.save(dir_path / "X_train.npy", X[:n_tr])
    np.save(dir_path / "y_train.npy", y[:n_tr])
    np.save(dir_path / "X_test.npy", X[n_tr:])
    np.save(dir_path / "y_test.npy", y[n_tr:])
    
    # Also create CSV files for compatibility (if needed by load_fe_artifacts)
    import pandas as pd
    feature_names = [f"feature_{i}" for i in range(d)]
    train_df = pd.DataFrame(X[:n_tr], columns=feature_names)
    train_df['cardio'] = y[:n_tr]
    train_df.to_csv(dir_path / "Train_data_tabnet.csv", index=False)
    
    test_df = pd.DataFrame(X[n_tr:], columns=feature_names)
    test_df['cardio'] = y[n_tr:]
    test_df.to_csv(dir_path / "Test_data_tabnet.csv", index=False)
    
    # Create selected_features JSON
    selected_features = {
        'kept_features': feature_names,
        'selected_features': feature_names
    }
    with open(dir_path / "selected_features_tabnet.json", 'w') as f:
        json.dump(selected_features, f)
    
    # Create selected_features CSV
    pd.DataFrame({'feature': feature_names}).to_csv(
        dir_path / "selected_features_tabnet.csv", index=False
    )


def _run(cmd, cwd):
    """Run command and return result."""
    print("RUN:", " ".join(cmd))
    result = subprocess.run(
        cmd, cwd=cwd, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if result.returncode != 0:
        print("STDOUT/STDERR:")
        print(result.stdout)
    return result


def test_tabnet_standalone(tmp_path, monkeypatch):
    """Test TabNet standalone mode."""
    # FE for TabNet
    fe_tab = tmp_path / "artifacts/fe/tabnet"
    _prepare_fe(fe_tab, n=100, d=6, seed=1)
    
    # Config
    cfg = tmp_path / "configs/default.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(f"""
random_seed: 123

paths:
  fe_dir: {tmp_path}/artifacts/fe
  fe_dir_tabnet: {tmp_path}/artifacts/fe/tabnet
  model_dir: {tmp_path}/artifacts/Model

modeling:
  hpo:
    enabled: false
  calibration:
    enabled: false
  test_scoring: false

evaluation:
  cv_folds: 2

tabnet:
  epochs: 2
  batch_size: 32
  n_d: 8
  n_a: 8
  n_steps: 3
  lr: 1e-3

preprocessing:
  target: cardio
""")
    
    # Run CLI
    cmd = [
        sys.executable, "-m", "src.modeling",
        "--config", str(cfg),
        "--model", "tabnet"
    ]
    result = _run(cmd, cwd=ROOT)
    
    # Check results
    assert result.returncode == 0, f"Command failed: {result.stdout}"
    
    outdir = tmp_path / "artifacts/Model/tabnet/fold_0"
    assert (outdir / "model.h5").exists(), "TabNet model.h5 must exist"
    assert not (outdir / "encoder.h5").exists(), "encoder.h5 must NOT exist in tabnet mode"
    assert not (outdir / "decoder.h5").exists(), "decoder.h5 must NOT exist in tabnet mode"


def test_vae_tabnet_two_stage(tmp_path, monkeypatch):
    """Test VAE-TabNet two-stage mode."""
    fe_tab = tmp_path / "artifacts/fe/tabnet"
    _prepare_fe(fe_tab, n=100, d=6, seed=2)
    
    cfg = tmp_path / "configs/default.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(f"""
random_seed: 123

paths:
  fe_dir: {tmp_path}/artifacts/fe
  fe_dir_tabnet: {tmp_path}/artifacts/fe/tabnet
  model_dir: {tmp_path}/artifacts/Model

modeling:
  hpo:
    enabled: false
  calibration:
    enabled: false
  test_scoring: false

evaluation:
  cv_folds: 2

tabnet:
  epochs: 2
  batch_size: 32
  n_d: 8
  n_a: 8
  n_steps: 3
  lr: 1e-3

vae:
  epochs: 2
  batch_size: 32
  latent_dim: 8

preprocessing:
  target: cardio
""")
    
    cmd = [
        sys.executable, "-m", "src.modeling",
        "--config", str(cfg),
        "--model", "vae_tabnet"
    ]
    result = _run(cmd, cwd=ROOT)
    
    # Check results
    assert result.returncode == 0, f"Command failed: {result.stdout}"
    
    outdir = tmp_path / "artifacts/Model/vae_tabnet/fold_0"
    assert (outdir / "encoder.h5").exists(), "encoder.h5 must exist in vae_tabnet mode"
    assert (outdir / "decoder.h5").exists(), "decoder.h5 must exist in vae_tabnet mode"
    # TabNet classifier should be saved (as classifier.h5 or in tabnet subdirectory)
    assert (outdir / "classifier.h5").exists() or (outdir / "tabnet" / "model.h5").exists(), \
        "TabNet classifier must be saved (as classifier.h5 or tabnet/model.h5)"

