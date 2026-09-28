"""Tests for preprocessing and metrics."""

import numpy as np
import pytest

from src.core.evaluation import compute_metrics
from src.core.model import RandomForestModel
from src.preprocessing.pipeline import _should_smote, _imbalance_ratio


def test_imbalance_ratio_balanced():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert _imbalance_ratio(y) == 1.0


def test_imbalance_ratio_imbalanced():
    y = np.array([0] * 9 + [1])
    assert _imbalance_ratio(y) == 9.0


def test_should_smote_auto():
    from src.io.registry import DatasetConfig
    cfg = DatasetConfig(
        id="t", display_name="T", domain="android", split_type="single_file",
        description="", feature_type="", apply_smote="auto",
    )
    y_balanced = np.array([0] * 50 + [1] * 50)
    y_imbal = np.array([0] * 90 + [1] * 10)
    assert not _should_smote(cfg, y_balanced)
    assert _should_smote(cfg, y_imbal)


def test_compute_metrics_perfect():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, y, y.astype(float), "test")
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0


def test_random_forest_train_predict():
    rng = np.random.default_rng(42)
    X = rng.random((60, 10))
    y = (X[:, 0] > 0.5).astype(int)
    model = RandomForestModel(n_estimators=10)
    model.train(X[:50], y[:50], [f"f{i}" for i in range(10)])
    preds = model.predict(X[50:])
    assert len(preds) == 10
    assert set(preds).issubset({0, 1})


def test_baseline_subsample_caps_large_training_set():
    from src.pipeline.runner import _subsample_stratified

    rng = np.random.default_rng(0)
    X = rng.random((50_000, 8))
    y = np.array([0] * 25_000 + [1] * 25_000)
    X_sub, y_sub, original_n = _subsample_stratified(X, y, 20_000)
    assert original_n == 50_000
    assert len(y_sub) == 20_000
    assert abs(y_sub.mean() - 0.5) < 0.02


def test_model_suite_includes_baselines_when_not_fast():
    from src.pipeline.runner import _model_suite

    assert len(_model_suite(fast=False)) == 4
    assert len(_model_suite(fast=True)) == 2
