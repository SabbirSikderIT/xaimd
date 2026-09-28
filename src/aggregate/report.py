"""Orchestrate full cross-dataset aggregation."""

from __future__ import annotations

from pathlib import Path

from src.io.registry import master_output_dir

from .collect import collect_all_results, save_collection_manifest
from .figures import build_master_figures
from .tables import build_master_tables


def aggregate_all(root: Path | None = None, verbose: bool = True) -> dict:
    """
    Collect all per-dataset outputs and produce unified tables + figures.

    Outputs land in ``reports/benchmark/``:
      tables/master_*.csv
      figures/fig_master_*.png
      collection_manifest.json

    Returns summary dict with paths and record counts.
    """
    records = collect_all_results(root)
    n_complete = sum(1 for r in records if r.status == "complete")
    n_total = len(records)

    if verbose:
        print("=" * 70)
        print("  XMALDETECT — CROSS-DATASET AGGREGATION")
        print("=" * 70)
        print(f"  Datasets with results: {n_complete}/{n_total}")

    manifest_path = save_collection_manifest(records, root)
    tables = build_master_tables(records, root)
    figure_paths = build_master_figures(records, root)

    master = master_output_dir(root)
    result = {
        "master_dir": str(master),
        "manifest": str(manifest_path),
        "tables": {k: str(master / "tables" / f"{k}.csv") for k in tables if not tables[k].empty},
        "figures": [str(p) for p in figure_paths],
        "n_complete": n_complete,
        "n_total": n_total,
    }

    if verbose:
        print(f"\n  Master output directory: {master}")
        print(f"  Tables : {len(result['tables'])}")
        print(f"  Figures: {len(result['figures'])}")
        if figure_paths:
            print(f"  Dashboard: {figure_paths[-1]}")
        print("=" * 70)

    return result
