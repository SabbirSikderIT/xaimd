#!/usr/bin/env python3
"""
XMalDetect — Full benchmark pipeline for 6 validated datasets.

Runs: preprocess -> train -> test metrics -> SHAP -> IEEE figures -> benchmark report.

Usage:
    python run_benchmark.py
    python run_benchmark.py --only drebin ember
    python run_benchmark.py --report-only
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve

from src.core.ieee_style import (
    ieee_bar_chart, ieee_confusion_matrix, ieee_roc_curve, ieee_scatter,
)
from src.core.metrics import positive_class_proba
from src.io.paths import (
    benchmark_dir, data_processed_dir, ensure_project_dirs,
    models_dir, reports_figures_dir, sync_model_to_models_dir,
    sync_processed_to_data,
)
from src.io.registry import ACTIVE_DATASETS, get_dataset
from src.pipeline.runner import run_forensic, run_shap, train_and_evaluate
from src.preprocessing.pipeline import run_preprocessing
from src.reporting.benchmark_report import generate_benchmark_report


def _export_ieee_figures(dataset_id: str) -> None:
    """Generate IEEE-compliant figures for one dataset."""
    from src.core.model import RandomForestModel
    from src.io.loaders import load_processed

    cfg = get_dataset(dataset_id)
    data = load_processed(dataset_id)
    model_path = models_dir(dataset_id) / "rf_shap_model.pkl"
    if not model_path.exists():
        model_path = Path(f"outputs/{dataset_id}/models/rf_shap_model.pkl")
    if not model_path.exists():
        return

    rf = RandomForestModel.load(str(model_path))
    y_pred = rf.predict(data.X_test)
    y_proba = positive_class_proba(rf.predict_proba(data.X_test))

    stem_base = reports_figures_dir() / dataset_id
    metrics = {
        "Accuracy": float((y_pred == data.y_test).mean()),
        "Precision": 0, "Recall": 0, "F1": 0,
    }
    from src.core.evaluation import compute_metrics
    m = compute_metrics(data.y_test, y_pred, y_proba, cfg.display_name)
    ieee_bar_chart(
        ["Acc", "Prec", "Rec", "F1", "AUC"],
        [m["accuracy"], m["precision"], m["recall"], m["f1"], m["auc_roc"]],
        f"{cfg.display_name} — Test Metrics",
        "Score",
        stem_base / "bar_metrics",
    )

    cm = confusion_matrix(data.y_test, y_pred, labels=[0, 1])
    ieee_confusion_matrix(cm, list(cfg.class_names),
                          f"{cfg.display_name} — Confusion Matrix",
                          stem_base / "confusion_matrix")

    fpr, tpr, _ = roc_curve(data.y_test, y_proba)
    ieee_roc_curve(fpr, tpr, auc(fpr, tpr),
                   f"{cfg.display_name} — ROC Curve",
                   stem_base / "roc_curve")

    ieee_scatter(y_proba, y_pred.astype(float),
                 f"{cfg.display_name} — Confidence vs Prediction",
                 "P(Malware)", "Predicted Class",
                 stem_base / "scatter_confidence")


def run_one(dataset_id: str, max_rows: int | None, fast: bool, baselines_only: bool = False) -> dict:
    cfg = get_dataset(dataset_id)
    cap = max_rows if max_rows is not None else cfg.pipeline_max_rows
    result = {"dataset": dataset_id, "status": "ok", "error": None}
    try:
        if not baselines_only:
            run_preprocessing(dataset_id, max_rows=cap)
            sync_processed_to_data(dataset_id)
        train_and_evaluate(dataset_id, fast=fast, baselines_only=baselines_only)
        sync_model_to_models_dir(dataset_id)
        if not baselines_only:
            run_shap(dataset_id, max_shap_samples=300)
            if cfg.forensic_plugin != "none":
                run_forensic(dataset_id, max_reports=5)
            _export_ieee_figures(dataset_id)
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        traceback.print_exc()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="XMalDetect 6-dataset benchmark")
    parser.add_argument("--only", nargs="*", help="Subset of datasets")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--fast", action="store_true",
                        help="Train RF models only (skip SVM/MLP baselines)")
    parser.add_argument("--full-models", action="store_true",
                        help="Include SVM/MLP baselines (default behaviour)")
    parser.add_argument("--baselines-only", action="store_true",
                        help="Retrain SVM/MLP only; keep existing RF metrics/checkpoints")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    ensure_project_dirs()
    targets = args.only or list(ACTIVE_DATASETS)

    if args.report_only:
        generate_benchmark_report()
        return

    print("=" * 70)
    print("  XMALDETECT BENCHMARK — 6 DATASETS")
    print("=" * 70)

    fast = args.fast and not args.full_models
    log = []
    for i, ds_id in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] {ds_id}")
        log.append(run_one(ds_id, args.max_rows, fast, args.baselines_only))

    log_path = Path("reports") / "benchmark_run_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"results": log, "time": datetime.now(timezone.utc).isoformat()},
                  f, indent=2)

    print("\n  Generating benchmark report...")
    generate_benchmark_report()

    n_fail = sum(1 for r in log if r["status"] != "ok")
    print(f"\n  Done. Failures: {n_fail}/{len(log)}")
    print(f"  Report: reports/XMalDetect_Benchmark_Report.pdf")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
