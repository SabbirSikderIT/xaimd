"""
End-to-end pipeline runner for one dataset.

Usage:
    python -m src.pipeline.runner drebin --step all
    python -m src.pipeline.runner ember --step preprocess
    python -m src.pipeline.runner --aggregate
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import auc, roc_curve

from src.core.metrics import positive_class_proba
from src.core.evaluation import (
    build_comparison_table,
    compute_metrics,
    plot_confusion_matrix,
    plot_gini_importance,
    plot_model_comparison_bar,
    plot_roc_curves,
)
from src.core.model import MLPModel, RandomForestModel, RFGiniModel, SVMModel
from src.core.shap_engine import SHAPEngine
from src.forensic_report import ForensicReport
from src.io.loaders import load_processed
from src.io.paths import ensure_dataset_dirs
from src.io.registry import get_dataset, list_datasets
from src.preprocessing.pipeline import run_preprocessing

BASELINE_MODELS = (
    "SVM (Black-Box Baseline)",
    "MLP Neural Network (Black-Box Baseline)",
)
MODEL_ORDER = (
    "Random Forest + SHAP",
    "RF + Gini Importance (Partial Baseline)",
    *BASELINE_MODELS,
)
BASELINE_MAX_SAMPLES = 20_000


def _slug(name: str) -> str:
    """Safe filename key from model name."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def _save_roc_curves(roc_data: list[dict], path) -> None:
    """Persist ROC curves as .npz for cross-dataset aggregation."""
    arrays = {}
    for entry in roc_data:
        key = _slug(entry["name"])
        fpr, tpr, _ = roc_curve(entry["y_true"], entry["y_proba"])
        arrays[f"{key}_fpr"] = fpr
        arrays[f"{key}_tpr"] = tpr
        arrays[f"{key}_auc"] = np.array([auc(fpr, tpr)])
    np.savez_compressed(path, **arrays)


def _model_suite(fast: bool = False) -> dict:
    """Return model instances; fast=True skips SVM/MLP baselines."""
    suite = {
        "Random Forest + SHAP": RandomForestModel(),
        "RF + Gini Importance (Partial Baseline)": RFGiniModel(),
    }
    if not fast:
        suite["SVM (Black-Box Baseline)"] = SVMModel()
        suite["MLP Neural Network (Black-Box Baseline)"] = MLPModel()
    return suite


def _subsample_stratified(X, y, max_samples: int, random_state: int = 42):
    """Stratified subsample for baseline models on large training sets."""
    n = len(y)
    if n <= max_samples:
        return X, y, n
    from sklearn.model_selection import train_test_split

    X_sub, _, y_sub, _ = train_test_split(
        X, y,
        train_size=max_samples,
        stratify=y,
        random_state=random_state,
    )
    return X_sub, y_sub, n


def _training_arrays(name: str, X_train, y_train) -> tuple:
    """RF uses full data; SVM/MLP subsample when training set is very large."""
    if name not in BASELINE_MODELS:
        return X_train, y_train, len(y_train)
    X_sub, y_sub, original_n = _subsample_stratified(
        X_train, y_train, BASELINE_MAX_SAMPLES,
    )
    if original_n > len(y_sub):
        print(
            f"  [{name}] Subsampled training: {len(y_sub):,} / {original_n:,} "
            f"(cap={BASELINE_MAX_SAMPLES:,} for baseline speed)"
        )
    return X_sub, y_sub, original_n


def _load_existing_metrics(summary_path) -> list[dict]:
    if not summary_path.exists():
        return []
    with open(summary_path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("metrics", [])


def _roc_entry(name: str, y_true, y_proba) -> dict:
    return {"name": name, "y_true": y_true, "y_proba": y_proba}


def _append_rf_roc_from_checkpoint(
    roc_data: list[dict],
    model_path,
    data,
    metrics_by_name: dict[str, dict],
) -> None:
    """Rebuild RF ROC curves from saved checkpoint without retraining."""
    if not model_path.exists():
        return
    rf = RandomForestModel.load(str(model_path))
    y_pred = rf.predict(data.X_test)
    y_proba = positive_class_proba(rf.predict_proba(data.X_test))
    for label in ("Random Forest + SHAP", "RF + Gini Importance (Partial Baseline)"):
        if label not in metrics_by_name:
            continue
        roc_data.append(_roc_entry(label, data.y_test, y_proba))


def train_and_evaluate(
    dataset_id: str,
    fast: bool = False,
    baselines_only: bool = False,
) -> dict:
    cfg = get_dataset(dataset_id)
    paths = ensure_dataset_dirs(cfg)
    data = load_processed(dataset_id)
    summary_path = paths["root"] / "modeling_summary.json"
    model_path = paths["models"] / "rf_shap_model.pkl"

    existing = _load_existing_metrics(summary_path)
    metrics_by_name = {m["model"]: m for m in existing}
    has_rf = "Random Forest + SHAP" in metrics_by_name and model_path.exists()

    if baselines_only and has_rf and not fast:
        models = {k: v for k, v in _model_suite(fast=False).items() if k in BASELINE_MODELS}
        results = [metrics_by_name[m] for m in metrics_by_name if m not in BASELINE_MODELS]
        print(f"[{dataset_id}] Refreshing SVM/MLP baselines (keeping existing RF metrics)")
    else:
        models = _model_suite(fast=fast)
        results = []

    roc_data: list[dict] = []
    if baselines_only and has_rf:
        _append_rf_roc_from_checkpoint(roc_data, model_path, data, metrics_by_name)

    for name, model in models.items():
        print(f"\n--- Training: {name} ---")
        X_tr, y_tr, _ = _training_arrays(name, data.X_train, data.y_train)
        model.train(X_tr, y_tr, data.feature_names)
        y_pred = model.predict(data.X_test)
        y_proba = positive_class_proba(model.predict_proba(data.X_test))
        metrics = compute_metrics(data.y_test, y_pred, y_proba, model_name=name)
        metrics["explainable"] = (
            "Full (SHAP)" if "SHAP" in name
            else "Partial (Gini)" if "Gini" in name else "None"
        )
        results.append(metrics)
        roc_data.append(_roc_entry(name, data.y_test, y_proba))

    ordered = {m["model"]: m for m in results}
    results = [ordered[name] for name in MODEL_ORDER if name in ordered]

    fig = paths["figures"]
    plot_model_comparison_bar(results, save_path=str(fig / "fig6_model_comparison_bar.png"))
    plot_roc_curves(roc_data, save_path=str(fig / "fig7_roc_curves.png"))

    if baselines_only and has_rf:
        rf = RandomForestModel.load(str(model_path))
    else:
        rf = models["Random Forest + SHAP"]
        rf.save(str(model_path))

    y_pred = rf.predict(data.X_test)
    plot_confusion_matrix(
        data.y_test, y_pred,
        model_name="Random Forest + SHAP",
        class_names=list(cfg.class_names),
        save_path=str(fig / "fig8_confusion_matrix.png"),
    )
    plot_gini_importance(
        rf.gini_importance(),
        save_path=str(fig / "gini_importance.png"),
    )

    table_path = paths["tables"] / "comparison_table3.csv"
    build_comparison_table(results, save_path=str(table_path))

    roc_path = paths["tables"] / "roc_curves.npz"
    _save_roc_curves(roc_data, roc_path)

    prep_path = paths["processed"] / "preprocessing_summary.json"
    preprocessing = {}
    if prep_path.exists():
        with open(prep_path, encoding="utf-8") as f:
            preprocessing = json.load(f)

    summary = {
        "dataset": dataset_id,
        "display_name": cfg.display_name,
        "domain": cfg.domain,
        "status": "complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": results,
        "preprocessing": preprocessing,
        "paths": {
            "table": str(table_path),
            "roc": str(roc_path),
            "model": str(model_path),
        },
    }
    with open(paths["root"] / "modeling_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def run_shap(dataset_id: str, max_shap_samples: int = 500) -> None:
    cfg = get_dataset(dataset_id)
    paths = ensure_dataset_dirs(cfg)
    data = load_processed(dataset_id)

    rf = RandomForestModel.load(str(paths["models"] / "rf_shap_model.pkl"))
    engine = SHAPEngine(rf, feature_names=data.feature_names)

    n = min(max_shap_samples, len(data.X_test))
    idx = np.random.default_rng(42).choice(len(data.X_test), n, replace=False)
    X_shap = data.X_test[idx]

    engine.fit(X_shap)
    fig = paths["figures"]
    engine.plot_global_summary(save_path=str(fig / "fig3_shap_summary.png"))
    engine.plot_waterfall(sample_idx=0, save_path=str(fig / "fig4_waterfall_sample0.png"))
    engine.compare_with_gini(
        rf.gini_importance(),
        save_path=str(fig / "fig_shap_vs_gini.png"),
    )
    engine.mean_abs_shap().to_csv(paths["tables"] / "table4_shap_top20.csv", index=False)


def run_forensic(dataset_id: str, max_reports: int = 10) -> None:
    cfg = get_dataset(dataset_id)
    if cfg.forensic_plugin == "none":
        print(f"[{dataset_id}] No forensic plugin - skipping.")
        return

    paths = ensure_dataset_dirs(cfg)
    data = load_processed(dataset_id)
    rf = RandomForestModel.load(str(paths["models"] / "rf_shap_model.pkl"))
    engine = SHAPEngine(rf, feature_names=data.feature_names)

    n = min(200, len(data.X_test))
    engine.fit(data.X_test[:n])

    y_proba = positive_class_proba(rf.predict_proba(data.X_test))
    y_pred = rf.predict(data.X_test)
    malware_idx = np.where(y_pred == 1)[0][:max_reports]

    reporter = ForensicReport(data.feature_names, engine, domain=cfg.forensic_plugin)
    reporter.build_cwe_table(save_path=str(paths["reports"] / "cwe_mapping_table.csv"))
    reporter.generate_batch_reports(
        malware_indices=malware_idx.tolist(),
        y_pred=y_pred,
        confidences=y_proba,
        y_true=data.y_test,
        output_dir=str(paths["reports"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="XMalDetect per-dataset pipeline")
    parser.add_argument("dataset", nargs="?", help="Dataset id (e.g. drebin, ember)")
    parser.add_argument(
        "--step",
        choices=["preprocess", "model", "shap", "forensic", "all"],
        default="all",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--aggregate", action="store_true",
                        help="Build master cross-dataset tables and figures")
    parser.add_argument("--max-rows", type=int, default=None, help="Cap rows for large CSVs")
    parser.add_argument("--fast", action="store_true",
                        help="Train RF models only (skip SVM/MLP)")
    args = parser.parse_args()

    if args.list:
        for cfg in list_datasets():
            cap = f" (cap={cfg.pipeline_max_rows})" if cfg.pipeline_max_rows else ""
            print(f"  {cfg.id:12}  {cfg.display_name}{cap}")
        return

    if args.aggregate:
        from src.aggregate.report import aggregate_all
        aggregate_all()
        return

    if not args.dataset:
        parser.error("dataset required (or use --list / --aggregate)")

    if args.step in ("preprocess", "all"):
        run_preprocessing(args.dataset, max_rows=args.max_rows)
    if args.step in ("model", "all"):
        train_and_evaluate(args.dataset, fast=args.fast)
    if args.step in ("shap", "all"):
        run_shap(args.dataset)
    if args.step in ("forensic", "all"):
        run_forensic(args.dataset)


if __name__ == "__main__":
    main()
