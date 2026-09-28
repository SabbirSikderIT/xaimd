"""Tests for CSV loaders."""

import numpy as np
import pandas as pd
import pytest

from src.io.loaders import normalize_labels, _resolve_file
from src.preprocessing.pipeline import _drop_constant


def test_normalize_labels_binary_int():
    y = normalize_labels(pd.Series([0, 1, 0, 1]), object())
    assert list(y) == [0, 1, 0, 1]


def test_normalize_labels_strings():
    y = normalize_labels(pd.Series(["benign", "malware", "benign"]), object())
    assert list(y) == [0, 1, 0]


def test_resolve_file_first_match(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1")
    (tmp_path / "b.csv").write_text("x\n1")
    path = _resolve_file(tmp_path, ("missing.csv", "b.csv"))
    assert path.name == "b.csv"


def test_resolve_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _resolve_file(tmp_path, "nope.csv")


def test_drop_constant_features():
    X = np.array([[1, 0, 1], [1, 1, 0], [1, 0, 1]], dtype=np.float32)
    names = ["a", "b", "c"]
    X_out, names_out = _drop_constant(X, names)
    assert "b" in names_out
    assert "a" not in names_out  # constant column
