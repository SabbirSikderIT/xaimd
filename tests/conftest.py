"""Pytest fixtures for XMalDetect."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_csv_dataset(tmp_path):
    """Create a minimal single-file dataset in a temp data dir."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    n, f = 40, 8
    rng = np.random.default_rng(0)
    X = (rng.random((n, f)) > 0.7).astype(int)
    y = (rng.random(n) > 0.6).astype(int)
    feat_cols = [f"feat_{i}" for i in range(f)]
    pd.DataFrame(X, columns=feat_cols).to_csv(data_dir / "tiny_features.csv", index=False)
    pd.DataFrame({"label": y}).to_csv(data_dir / "tiny_labels.csv", index=False)
    return data_dir, feat_cols


@pytest.fixture
def mock_modeling_summary(tmp_path):
    """Write a fake modeling_summary.json for aggregation tests."""
    ds_dir = tmp_path / "outputs" / "testds"
    (ds_dir / "tables").mkdir(parents=True)
    (ds_dir / "processed").mkdir(parents=True)

    summary = {
        "dataset": "testds",
        "display_name": "Test Dataset",
        "domain": "android",
        "status": "complete",
        "metrics": [
            {"model": "Random Forest + SHAP", "accuracy": 0.95, "precision": 0.94,
             "recall": 0.93, "f1": 0.935, "auc_roc": 0.98, "avg_prec": 0.97,
             "explainable": "Full (SHAP)"},
            {"model": "SVM (Black-Box Baseline)", "accuracy": 0.92, "precision": 0.91,
             "recall": 0.90, "f1": 0.905, "auc_roc": 0.95, "avg_prec": 0.94,
             "explainable": "None"},
        ],
        "preprocessing": {
            "train_shape": [100, 50],
            "test_shape": [25, 50],
            "n_features": 50,
            "smote_applied": True,
            "scaled": False,
        },
    }
    with open(ds_dir / "modeling_summary.json", "w") as f:
        json.dump(summary, f)
    with open(ds_dir / "processed" / "preprocessing_summary.json", "w") as f:
        json.dump(summary["preprocessing"], f)

    # ROC curves
    fpr = np.linspace(0, 1, 20)
    tpr = np.sqrt(fpr)
    np.savez_compressed(
        ds_dir / "tables" / "roc_curves.npz",
        random_forest_shap_fpr=fpr,
        random_forest_shap_tpr=tpr,
        random_forest_shap_auc=np.array([0.98]),
    )
    pd.DataFrame({
        "feature": ["feat_0", "feat_1"],
        "mean_abs_shap": [0.5, 0.3],
    }).to_csv(ds_dir / "tables" / "table4_shap_top20.csv", index=False)

    return tmp_path
