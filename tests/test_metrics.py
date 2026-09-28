"""Tests for metric helpers."""

import numpy as np

from src.core.metrics import positive_class_proba


def test_positive_class_proba_binary():
    proba = np.array([[0.2, 0.8], [0.6, 0.4]])
    assert np.allclose(positive_class_proba(proba), [0.8, 0.4])


def test_positive_class_proba_single_column():
    proba = np.array([[0.9], [0.1]])
    assert np.allclose(positive_class_proba(proba), [0.9, 0.1])
