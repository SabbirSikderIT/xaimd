"""Canonical project paths — DevOps-style layout."""

from __future__ import annotations

import shutil
from pathlib import Path

from .registry import ACTIVE_DATASETS, DatasetConfig, _project_root, get_dataset


def project_root() -> Path:
    return _project_root()


def data_raw_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "data"


def data_processed_dir(dataset_id: str, root: Path | None = None) -> Path:
    return (root or project_root()) / "data" / "processed" / dataset_id


def models_dir(dataset_id: str, root: Path | None = None) -> Path:
    return (root or project_root()) / "models" / dataset_id


def reports_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "reports"


def reports_figures_dir(root: Path | None = None) -> Path:
    return reports_dir(root) / "figures"


def reports_tables_dir(root: Path | None = None) -> Path:
    return reports_dir(root) / "tables"


def benchmark_dir(root: Path | None = None) -> Path:
    return reports_dir(root) / "benchmark"


def dataset_paths(cfg: DatasetConfig, root: Path | None = None) -> dict[str, Path]:
    """Runtime artifact paths (canonical layout)."""
    base = root or project_root()
    return {
        "root": base / "outputs" / cfg.id,
        "figures": reports_figures_dir(base) / cfg.id,
        "models": models_dir(cfg.id, base),
        "reports": reports_dir(base) / "datasets" / cfg.id,
        "processed": data_processed_dir(cfg.id, base),
        "tables": reports_tables_dir(base) / cfg.id,
    }


def ensure_dataset_dirs(cfg: DatasetConfig, root: Path | None = None) -> dict[str, Path]:
    paths = dataset_paths(cfg, root)
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def sync_processed_to_data(dataset_id: str, root: Path | None = None) -> None:
    """Copy processed arrays from legacy outputs/ to data/processed/."""
    base = root or project_root()
    legacy = base / "outputs" / dataset_id / "processed"
    target = data_processed_dir(dataset_id, base)
    if not legacy.exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    for f in legacy.glob("*"):
        if f.is_file():
            shutil.copy2(f, target / f.name)


def sync_model_to_models_dir(dataset_id: str, root: Path | None = None) -> None:
    """Copy trained model to models/{dataset}/."""
    base = root or project_root()
    src = base / "outputs" / dataset_id / "models" / "rf_shap_model.pkl"
    if not src.exists():
        src = models_dir(dataset_id, base) / "rf_shap_model.pkl"
    if src.exists():
        dst = models_dir(dataset_id, base)
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst / "rf_shap_model.pkl")


def ensure_project_dirs(root: Path | None = None) -> None:
    """Create top-level project directories."""
    base = root or project_root()
    for d in ("data/processed", "models", "reports/figures", "reports/tables",
              "reports/benchmark", "notebooks", "tests"):
        (base / d).mkdir(parents=True, exist_ok=True)
    for ds in ACTIVE_DATASETS:
        ensure_dataset_dirs(get_dataset(ds), base)
