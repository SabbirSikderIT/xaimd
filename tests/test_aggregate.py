"""Tests for cross-dataset aggregation."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.aggregate.collect import collect_dataset
from src.aggregate.tables import (
    build_dataset_summary,
    build_model_comparison_long,
    build_master_tables,
)
from src.aggregate.report import aggregate_all


def test_collect_dataset_from_fixture(mock_modeling_summary, monkeypatch):
    monkeypatch.setattr(
        "src.aggregate.collect.get_dataset",
        lambda ds_id: type("C", (), {
            "id": "testds", "display_name": "Test", "domain": "android"
        })(),
    )
    rec = collect_dataset("testds", mock_modeling_summary)
    assert rec.status == "complete"
    assert rec.rf_metrics["f1"] == 0.935
    assert rec.roc_curves is not None
    assert len(rec.roc_curves) >= 1


def test_build_model_comparison_long(mock_modeling_summary, monkeypatch):
    monkeypatch.setattr(
        "src.aggregate.collect.get_dataset",
        lambda ds_id: type("C", (), {
            "id": "testds", "display_name": "Test", "domain": "android"
        })(),
    )
    from src.aggregate.collect import collect_all_results
    with patch("src.aggregate.collect.ACTIVE_DATASETS", ["testds"]):
        records = [collect_dataset("testds", mock_modeling_summary)]
    long_df = build_model_comparison_long(records)
    assert len(long_df) == 2
    assert "f1" in long_df.columns
    assert set(long_df["model"]) == {"Random Forest + SHAP", "SVM (Black-Box Baseline)"}


def test_build_dataset_summary(mock_modeling_summary, monkeypatch):
    monkeypatch.setattr(
        "src.aggregate.collect.get_dataset",
        lambda ds_id: type("C", (), {
            "id": "testds", "display_name": "Test", "domain": "android"
        })(),
    )
    rec = collect_dataset("testds", mock_modeling_summary)
    summary = build_dataset_summary([rec])
    assert summary.loc["testds", "rf_f1"] == 0.935
    assert summary.loc["testds", "train_samples"] == 100


def test_aggregate_all_produces_master_files(mock_modeling_summary, monkeypatch):
    monkeypatch.setattr(
        "src.aggregate.collect.get_dataset",
        lambda ds_id: type("C", (), {
            "id": "testds", "display_name": "Test Dataset", "domain": "android"
        })(),
    )
    with patch("src.aggregate.collect.ACTIVE_DATASETS", ["testds"]):
        result = aggregate_all(mock_modeling_summary, verbose=False)

    master = Path(result["master_dir"])
    assert (master / "collection_manifest.json").exists()
    assert (master / "tables" / "master_dataset_summary.csv").exists()
    assert (master / "figures" / "fig_master_dashboard.png").exists()
