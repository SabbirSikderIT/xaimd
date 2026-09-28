"""Shared metric helpers."""

from __future__ import annotations

import numpy as np


def positive_class_proba(proba: np.ndarray) -> np.ndarray:
    """
    Extract P(malware) from predict_proba output.

    Handles edge cases where only one class was seen during training
    (proba has shape (n, 1)).
    """
    if proba.ndim == 1:
        return proba
    if proba.shape[1] == 1:
        return proba[:, 0]
    return proba[:, 1]
