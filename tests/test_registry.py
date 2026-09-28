"""Tests for dataset registry."""

import pytest

from src.io.registry import (
    ACTIVE_DATASETS, DATASETS, get_dataset, list_datasets, master_output_dir,
)


def test_all_eight_datasets_registered():
    assert len(DATASETS) >= 8


def test_active_datasets_are_six():
    assert len(ACTIVE_DATASETS) == 6
    assert set(ACTIVE_DATASETS).issubset(set(DATASETS.keys()))


def test_get_dataset_normalizes_underscores():
    cfg = get_dataset("nsl_kdd")
    assert cfg.id == "nsl-kdd"


def test_unknown_dataset_raises():
    with pytest.raises(KeyError, match="Unknown dataset"):
        get_dataset("not-a-real-dataset")


def test_list_datasets_returns_all():
    assert len(list_datasets()) == 8


def test_master_output_dir():
    path = master_output_dir()
    assert path.name == "benchmark"
    assert path.parent.name == "reports"


def test_dataset_has_required_fields():
    for cfg in DATASETS.values():
        assert cfg.id
        assert cfg.display_name
        assert cfg.domain in ("android", "network", "pe", "behavioral")
        assert cfg.split_type in ("single_file", "pre_split", "official_test")
