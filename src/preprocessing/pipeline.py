"""Shared preprocessing: clean features, split, SMOTE, save per-dataset artifacts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.io.loaders import load_raw
from src.io.paths import ensure_dataset_dirs
from src.io.registry import DatasetConfig, get_dataset


def _imbalance_ratio(y: np.ndarray) -> float:
    counts = np.bincount(y.astype(int))
    if len(counts) < 2 or counts.min() == 0:
        return float("inf")
    return counts.max() / counts.min()


def _should_smote(cfg: DatasetConfig, y: np.ndarray) -> bool:
    if cfg.apply_smote is True:
        return True
    if cfg.apply_smote is False:
        return False
    return _imbalance_ratio(y) > 1.5


def _drop_constant(X: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    df = pd.DataFrame(X, columns=names)
    keep = df.std() > 0
    if keep.sum() == 0:
        return X.astype(np.float32), names
    return df.loc[:, keep].values.astype(np.float32), df.columns[keep].tolist()


def run_preprocessing(
    dataset_id: str,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_smote: bool | None = None,
    max_rows: int | None = None,
) -> dict:
    """
    Load raw CSVs, preprocess, and save arrays under ``outputs/{dataset}/processed/``.

    Returns a summary dict with shapes and paths.
    """
    cfg = get_dataset(dataset_id)
    paths = ensure_dataset_dirs(cfg)
    proc = paths["processed"]

    raw = load_raw(dataset_id, max_rows=max_rows)

    if raw.X_test is not None:
        for split_name, y in [("train", raw.y_train), ("test", raw.y_test)]:
            if len(np.unique(y)) < 2:
                raise ValueError(
                    f"[{dataset_id}] {split_name} split has only one class. "
                    "Increase --max-rows or check labels."
                )
    elif len(np.unique(raw.y_train)) < 2:
        raise ValueError(f"[{dataset_id}] dataset has only one class. Check labels.")

    if raw.X_test is not None:
        X_train, X_test = raw.X_train, raw.X_test
        y_train, y_test = raw.y_train, raw.y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            raw.X_train, raw.y_train,
            test_size=test_size,
            random_state=random_state,
            stratify=raw.y_train,
        )

    names = raw.feature_names
    if cfg.drop_constant:
        X_train, names = _drop_constant(X_train, names)
        df_test = pd.DataFrame(X_test, columns=raw.feature_names)
        X_test = df_test[names].values.astype(np.float32)

    if cfg.scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
        np.save(proc / "scaler_mean.npy", scaler.mean_.astype(np.float32))
        np.save(proc / "scaler_scale.npy", scaler.scale_.astype(np.float32))

    use_smote = apply_smote if apply_smote is not None else _should_smote(cfg, y_train)
    smote_applied = False
    if use_smote:
        k = max(1, min(5, int((y_train == 1).sum()) - 1))
        smote = SMOTE(random_state=random_state, k_neighbors=k)
        try:
            X_train, y_train = smote.fit_resample(X_train, y_train)
            smote_applied = True
        except ValueError as exc:
            print(f"[Preprocess] SMOTE skipped: {exc}")

    np.save(proc / "X_train.npy", X_train.astype(np.float32))
    np.save(proc / "X_test.npy", X_test.astype(np.float32))
    np.save(proc / "y_train.npy", y_train.astype(int))
    np.save(proc / "y_test.npy", y_test.astype(int))
    pd.Series(names).to_csv(proc / "feature_names.csv", index=False, header=False)

    summary = {
        "dataset": cfg.id,
        "display_name": cfg.display_name,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "n_features": len(names),
        "train_malware_pct": float((y_train == 1).mean()),
        "test_malware_pct": float((y_test == 1).mean()),
        "smote_applied": smote_applied,
        "scaled": cfg.scale_features,
        "processed_dir": str(proc),
    }
    with open(proc / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{cfg.id}] Preprocessing complete -> {proc}")
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}  SMOTE: {smote_applied}")
    return summary
