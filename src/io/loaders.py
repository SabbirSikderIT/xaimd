"""
Load processed CSV artifacts from ``data/`` using the dataset registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .registry import DATA_DIR, DatasetConfig, get_dataset


class RawSplit(NamedTuple):
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    family_train: np.ndarray | None = None
    family_test: np.ndarray | None = None


class ProcessedSplit(NamedTuple):
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


BENIGN_TERMS = {
    "benign", "normal", "goodware", "legitimate", "clean", "0",
}


def _resolve_file(data_dir: Path, name: str | tuple[str, ...]) -> Path:
    candidates = (name,) if isinstance(name, str) else name
    for candidate in candidates:
        path = data_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"None of {candidates} found in {data_dir}. "
        "Run: python data/download_datasets.py --process <dataset>"
    )


def _has_index_column(path: Path) -> bool:
    with open(path, encoding="utf-8", errors="replace") as f:
        header = f.readline()
    first = header.split(",")[0].strip()
    return first in {"", "Unnamed: 0"} or first.startswith("app_") or first.isdigit()


def normalize_labels(series: pd.Series, cfg: DatasetConfig) -> np.ndarray:
    """Map labels to binary 0=benign, 1=malware."""
    col = series
    if col.dtype == object or str(col.dtype) in ("string", "str"):
        lowered = col.astype(str).str.strip().str.lower()
        if set(lowered.unique()) <= BENIGN_TERMS | {"malware", "1", "attack", "anomaly"}:
            return (lowered != "benign").astype(int).values
        return pd.factorize(col)[0].astype(int)
    numeric = col.astype(float)
    unique = set(np.unique(numeric))
    if unique <= {0, 1}:
        return numeric.astype(int).values
    if unique <= {0, 1, -1}:
        return (numeric == 1).astype(int).values
    return (numeric != numeric.min()).astype(int).values


def _valid_label_mask(series: pd.Series) -> np.ndarray:
    """Mask rows with valid labels (exclude EMBER -1 unlabeled)."""
    numeric = pd.to_numeric(series, errors="coerce")
    return (~numeric.isna()) & (numeric != -1)


def _stratified_row_indices(y: np.ndarray, max_rows: int, seed: int = 42) -> np.ndarray:
    """Pick up to max_rows indices preserving class ratio."""
    n = len(y)
    if n <= max_rows:
        return np.arange(n)
    if len(np.unique(y)) < 2:
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n, max_rows, replace=False))
    idx = np.arange(n)
    _, sample = train_test_split(
        idx, train_size=max_rows, stratify=y, random_state=seed,
    )
    return np.sort(sample)


def _read_feature_rows(path: Path, row_indices: np.ndarray, use_index_col: bool) -> pd.DataFrame:
    """Read specific data rows from a large feature CSV (0-based row indices)."""
    keep_lines = {int(i) + 1 for i in row_indices}  # +1 for header line
    return pd.read_csv(
        path,
        index_col=0 if use_index_col else None,
        skiprows=lambda x: x > 0 and x not in keep_lines,
        low_memory=False,
    )


def _load_xy(
    features_path: Path,
    labels_path: Path,
    cfg: DatasetConfig,
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    y_df = pd.read_csv(labels_path)
    label_col = cfg.label_column
    if label_col and label_col in y_df.columns:
        raw_labels = y_df[label_col]
    elif "label" in y_df.columns:
        raw_labels = y_df["label"]
    else:
        raw_labels = y_df.iloc[:, -1]

    valid = _valid_label_mask(raw_labels)
    y_df = y_df.loc[valid].reset_index(drop=True)
    y = normalize_labels(raw_labels.loc[valid].reset_index(drop=True), cfg)

    if max_rows and len(y_df) > max_rows:
        row_idx = _stratified_row_indices(y, max_rows)
        y_df = y_df.iloc[row_idx].reset_index(drop=True)
        y = y[row_idx]
        original_rows = np.where(valid)[0][row_idx]
    else:
        original_rows = np.where(valid)[0]

    use_index = _has_index_column(features_path)
    if max_rows and len(y_df) < len(valid):
        X_df = _read_feature_rows(features_path, original_rows, use_index)
    else:
        X_df = pd.read_csv(
            features_path,
            index_col=0 if use_index else None,
            low_memory=False,
        )
        if len(X_df) != len(y_df):
            X_df = X_df.iloc[: len(y_df)]

    family = None
    if cfg.family_column and cfg.family_column in y_df.columns:
        family = y_df[cfg.family_column].values

    id_cols = {"sample_id", "id", "sha256", "md5", "hash"}
    drop_cols = [c for c in X_df.columns if c in id_cols]
    if drop_cols:
        X_df = X_df.drop(columns=drop_cols)

    feature_names = X_df.columns.astype(str).tolist()
    X = X_df.values.astype(np.float32)
    if len(X) != len(y):
        n = min(len(X), len(y))
        X, y = X[:n], y[:n]
        if family is not None:
            family = family[:n]
    return X, y, feature_names, family


def load_raw(
    dataset_id: str,
    data_dir: Path | None = None,
    max_rows: int | None = None,
) -> RawSplit:
    cfg = get_dataset(dataset_id)
    data_dir = data_dir or DATA_DIR
    row_cap = max_rows if max_rows is not None else cfg.max_rows

    if cfg.split_type in ("pre_split", "official_test"):
        assert cfg.train_features and cfg.train_labels
        assert cfg.test_features and cfg.test_labels
        X_train, y_train, fn_train, fam_train = _load_xy(
            _resolve_file(data_dir, cfg.train_features),
            _resolve_file(data_dir, cfg.train_labels),
            cfg, row_cap,
        )
        X_test, y_test, fn_test, fam_test = _load_xy(
            _resolve_file(data_dir, cfg.test_features),
            _resolve_file(data_dir, cfg.test_labels),
            cfg, row_cap,
        )
        if fn_train != fn_test:
            raise ValueError(
                f"Train/test feature mismatch for {cfg.id}: "
                f"{len(fn_train)} vs {len(fn_test)} columns"
            )
        return RawSplit(X_train, X_test, y_train, y_test, fn_train, fam_train, fam_test)

    assert cfg.features and cfg.labels
    X, y, feature_names, family = _load_xy(
        _resolve_file(data_dir, cfg.features),
        _resolve_file(data_dir, cfg.labels),
        cfg, row_cap,
    )
    return RawSplit(X, None, y, None, feature_names, family, None)


def load_processed(dataset_id: str, processed_dir: Path | None = None) -> ProcessedSplit:
    from .paths import data_processed_dir, project_root

    cfg = get_dataset(dataset_id)
    if processed_dir:
        proc = processed_dir
    else:
        proc = data_processed_dir(cfg.id)
        if not (proc / "X_train.npy").exists():
            proc = project_root() / "outputs" / cfg.id / "processed"
    for name in ("X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy", "feature_names.csv"):
        if not (proc / name).exists():
            raise FileNotFoundError(
                f"Missing {proc / name}. Run notebook 02_Preprocess for '{cfg.id}' first."
            )
    feature_names = pd.read_csv(proc / "feature_names.csv", header=None)[0].tolist()
    return ProcessedSplit(
        np.load(proc / "X_train.npy"),
        np.load(proc / "X_test.npy"),
        np.load(proc / "y_train.npy"),
        np.load(proc / "y_test.npy"),
        feature_names,
    )
