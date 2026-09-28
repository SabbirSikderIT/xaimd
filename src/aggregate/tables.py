"""Build unified cross-dataset comparison tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.io.registry import master_output_dir

from .collect import DatasetRunRecord, collect_all_results


METRIC_COLS = ["accuracy", "precision", "recall", "f1", "auc_roc", "avg_prec"]


def build_model_comparison_long(records: list[DatasetRunRecord]) -> pd.DataFrame:
    """Long-format: one row per (dataset, model) with all metrics."""
    rows = []
    for rec in records:
        if rec.status == "not_run" or not rec.metrics:
            continue
        for m in rec.metrics:
            row = {
                "dataset": rec.dataset_id,
                "display_name": rec.display_name,
                "domain": rec.domain,
                "model": m.get("model", ""),
                "explainable": m.get("explainable", ""),
            }
            for col in METRIC_COLS:
                row[col] = m.get(col)
            rows.append(row)
    return pd.DataFrame(rows)


def build_dataset_summary(records: list[DatasetRunRecord]) -> pd.DataFrame:
    """One row per dataset: RF+SHAP metrics + preprocessing info."""
    rows = []
    for rec in records:
        rf = rec.rf_metrics
        prep = rec.preprocessing
        rows.append({
            "dataset": rec.dataset_id,
            "display_name": rec.display_name,
            "domain": rec.domain,
            "status": rec.status,
            "train_samples": prep.get("train_shape", [None, None])[0],
            "test_samples": prep.get("test_shape", [None, None])[0],
            "n_features": prep.get("n_features"),
            "smote_applied": prep.get("smote_applied"),
            "scaled": prep.get("scaled"),
            "rf_accuracy": rf.get("accuracy") if rf else None,
            "rf_precision": rf.get("precision") if rf else None,
            "rf_recall": rf.get("recall") if rf else None,
            "rf_f1": rf.get("f1") if rf else None,
            "rf_auc_roc": rf.get("auc_roc") if rf else None,
            "rf_avg_prec": rf.get("avg_prec") if rf else None,
        })
    return pd.DataFrame(rows).set_index("dataset")


def build_domain_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Average RF+SHAP metrics grouped by malware domain."""
    rf = long_df[long_df["model"].str.contains("SHAP", na=False)].copy()
    if rf.empty:
        return pd.DataFrame()
    agg = rf.groupby("domain")[METRIC_COLS].mean().round(4)
    agg["n_datasets"] = rf.groupby("domain")["dataset"].nunique()
    return agg


def build_model_pivot(long_df: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    """Pivot: rows=datasets, columns=models, values=metric."""
    if long_df.empty:
        return pd.DataFrame()
    return long_df.pivot_table(
        index="dataset", columns="model", values=metric, aggfunc="first"
    ).round(4)


def build_shap_consolidated(records: list[DatasetRunRecord]) -> pd.DataFrame:
    """Top SHAP features across all datasets."""
    frames = []
    for rec in records:
        if rec.shap_top20 is None:
            continue
        df = rec.shap_top20.copy()
        df.insert(0, "dataset", rec.dataset_id)
        df.insert(1, "domain", rec.domain)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_master_tables(
    records: list[DatasetRunRecord] | None = None,
    root: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build and save all master tables to ``reports/benchmark/tables/``.

    Returns dict of table_name -> DataFrame.
    """
    if records is None:
        records = collect_all_results(root)

    out_dir = master_output_dir(root) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df = build_model_comparison_long(records)
    summary_df = build_dataset_summary(records)
    domain_df = build_domain_summary(long_df)
    pivot_f1 = build_model_pivot(long_df, "f1")
    pivot_auc = build_model_pivot(long_df, "auc_roc")
    shap_df = build_shap_consolidated(records)

    tables = {
        "master_model_comparison": long_df,
        "master_dataset_summary": summary_df,
        "master_domain_summary": domain_df,
        "master_f1_pivot": pivot_f1,
        "master_auc_pivot": pivot_auc,
        "master_shap_top20": shap_df,
    }

    for name, df in tables.items():
        if df is not None and not (isinstance(df, pd.DataFrame) and df.empty):
            path = out_dir / f"{name}.csv"
            df.to_csv(path)
            print(f"[Aggregate] Table saved -> {path}")

    return tables
