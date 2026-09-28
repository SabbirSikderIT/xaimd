"""Compare results across datasets — delegates to aggregate module."""

from __future__ import annotations

from pathlib import Path

from src.aggregate.report import aggregate_all
from src.aggregate.tables import build_dataset_summary
from src.aggregate.collect import collect_all_results
from src.io.registry import _project_root, master_output_dir


def print_comparison(root: Path | None = None) -> None:
    """Build master outputs and print RF+SHAP summary table."""
    result = aggregate_all(root)
    records = collect_all_results(root)
    df = build_dataset_summary(records)
    print("\n  RF+SHAP Quick Summary:")
    cols = ["domain", "status", "rf_f1", "rf_auc_roc", "train_samples", "test_samples"]
    print(df[[c for c in cols if c in df.columns]].to_string())
    print(f"\n  Full outputs -> {result['master_dir']}")


if __name__ == "__main__":
    print_comparison()
