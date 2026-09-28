"""Collect per-dataset pipeline outputs into structured records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.io.registry import ACTIVE_DATASETS, DatasetConfig, _project_root, get_dataset, master_output_dir


@dataclass
class DatasetRunRecord:
    """All artifacts from one completed dataset pipeline run."""

    dataset_id: str
    display_name: str
    domain: str
    status: str  # complete | partial | not_run | error
    metrics: list[dict] = field(default_factory=list)
    preprocessing: dict = field(default_factory=dict)
    roc_curves: dict[str, dict] = field(default_factory=dict)  # model -> {fpr, tpr, auc}
    shap_top20: pd.DataFrame | None = None
    error: str | None = None
    paths: dict[str, str] = field(default_factory=dict)

    @property
    def rf_metrics(self) -> dict | None:
        for m in self.metrics:
            if "SHAP" in m.get("model", ""):
                return m
        return self.metrics[0] if self.metrics else None


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_roc_npz(path: Path) -> dict[str, dict] | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    curves = {}
    for key in data.files:
        if key.endswith("_fpr"):
            model = key[:-4]
            curves.setdefault(model, {})["fpr"] = data[key]
            curves[model]["tpr"] = data[f"{model}_tpr"]
            if f"{model}_auc" in data.files:
                curves[model]["auc"] = float(np.asarray(data[f"{model}_auc"]).ravel()[0])
    return curves


def collect_dataset(dataset_id: str, root: Path | None = None) -> DatasetRunRecord:
    """Load all available outputs for one dataset."""
    cfg = get_dataset(dataset_id)
    base = (root or _project_root()) / "outputs" / dataset_id

    record = DatasetRunRecord(
        dataset_id=cfg.id,
        display_name=cfg.display_name,
        domain=cfg.domain,
        status="not_run",
        paths={"root": str(base)},
    )

    modeling = _load_json(base / "modeling_summary.json")
    preprocess = _load_json(base / "processed" / "preprocessing_summary.json")
    roc = _load_roc_npz(base / "tables" / "roc_curves.npz")
    shap_path = base / "tables" / "table4_shap_top20.csv"

    if preprocess:
        record.preprocessing = preprocess
        record.paths["preprocessing"] = str(base / "processed")

    if modeling:
        record.metrics = modeling.get("metrics", [])
        record.status = modeling.get("status", "complete")
        record.error = modeling.get("error")
        record.paths["modeling"] = str(base / "modeling_summary.json")
    elif preprocess:
        record.status = "partial"

    if roc:
        record.roc_curves = roc
        record.paths["roc"] = str(base / "tables" / "roc_curves.npz")

    if shap_path.exists():
        record.shap_top20 = pd.read_csv(shap_path)
        record.paths["shap"] = str(shap_path)

    return record


def collect_all_results(root: Path | None = None) -> list[DatasetRunRecord]:
    """Collect outputs for every active benchmark dataset."""
    return [collect_dataset(ds_id, root) for ds_id in ACTIVE_DATASETS]


def save_collection_manifest(records: list[DatasetRunRecord], root: Path | None = None) -> Path:
    """Write a JSON manifest of collection status."""
    out_dir = master_output_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": [
            {
                "id": r.dataset_id,
                "display_name": r.display_name,
                "domain": r.domain,
                "status": r.status,
                "rf_f1": r.rf_metrics.get("f1") if r.rf_metrics else None,
                "rf_auc": r.rf_metrics.get("auc_roc") if r.rf_metrics else None,
                "train_shape": r.preprocessing.get("train_shape"),
                "test_shape": r.preprocessing.get("test_shape"),
                "error": r.error,
            }
            for r in records
        ],
    }
    path = out_dir / "collection_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path
