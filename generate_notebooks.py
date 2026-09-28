"""
Generate per-dataset notebooks for the 6 active benchmark datasets.

Structure per dataset:
  01_EDA.ipynb
  02_Preprocess.ipynb
  03_Train.ipynb
  04_Test.ipynb

Plus:
  notebooks/benchmark/05_Benchmark_Report.ipynb

Run: python generate_notebooks.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.io.registry import ACTIVE_DATASETS, DATASETS, get_dataset

ROOT = Path(__file__).resolve().parent


def nb(cells: list) -> dict:
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }


def md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": t,
            "id": "md" + str(abs(hash(t)))[-6:]}


def code(t: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": t, "outputs": [],
            "execution_count": None, "id": "co" + str(abs(hash(t)))[-6:]}


def setup(ds_id: str) -> str:
    return f'''import sys, os
sys.path.insert(0, os.path.abspath('../..'))
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
from pathlib import Path
from src.io.registry import get_dataset, ACTIVE_DATASETS
from src.io.paths import ensure_dataset_dirs, data_processed_dir, models_dir, reports_figures_dir

DATASET_ID = "{ds_id}"
cfg = get_dataset(DATASET_ID)
paths = ensure_dataset_dirs(cfg)
np.random.seed(42)
print(cfg.display_name)
'''


def nb_eda(ds_id: str, cfg) -> dict:
    return nb([
        md(f"# 01 — EDA\n## {cfg.display_name}\n\nExplore raw data before preprocessing."),
        code(setup(ds_id) + "from src.io.loaders import load_raw\n"),
        code("raw = load_raw(DATASET_ID)\nX = raw.X_train if raw.X_test is None else np.vstack([raw.X_train, raw.X_test])\ny = raw.y_train if raw.y_test is None else np.concatenate([raw.y_train, raw.y_test])\nprint(f'Samples: {X.shape[0]:,}  Features: {X.shape[1]:,}')\nprint(f'Malware: {(y==1).mean():.1%}')"),
        code("from src.core.evaluation import plot_class_distribution\nplot_class_distribution(y, cfg.display_name, str(reports_figures_dir()/DATASET_ID/'eda_class_distribution.png'))"),
    ])


def nb_preprocess(ds_id: str, cfg) -> dict:
    cap = cfg.pipeline_max_rows or "None"
    return nb([
        md(f"# 02 — Preprocess\n## {cfg.display_name}\n\nSaves arrays to `data/processed/{ds_id}/`"),
        code(setup(ds_id) + "from src.preprocessing.pipeline import run_preprocessing\nfrom src.io.paths import sync_processed_to_data\n"),
        code(f"summary = run_preprocessing(DATASET_ID, max_rows={cap})\nsync_processed_to_data(DATASET_ID)\nprint(summary)"),
    ])


def nb_train(ds_id: str, cfg) -> dict:
    return nb([
        md(f"# 03 — Train\n## {cfg.display_name}\n\nTrain Random Forest + SHAP model. Saves to `models/{ds_id}/`"),
        code(setup(ds_id) + "from src.pipeline.runner import train_and_evaluate\nfrom src.io.paths import sync_model_to_models_dir\n"),
        code("result = train_and_evaluate(DATASET_ID, fast=False)\nsync_model_to_models_dir(DATASET_ID)\nprint('Model:', models_dir(DATASET_ID)/'rf_shap_model.pkl')"),
    ])


def nb_test(ds_id: str, cfg) -> dict:
    return nb([
        md(f"# 04 — Test & Evaluate\n## {cfg.display_name}\n\nEvaluate on test set + IEEE figures."),
        code(setup(ds_id) + """
from src.io.loaders import load_processed
from src.core.model import RandomForestModel
from src.core.evaluation import compute_metrics
from src.core.metrics import positive_class_proba
from src.core.ieee_style import ieee_bar_chart, ieee_confusion_matrix, ieee_roc_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc

data = load_processed(DATASET_ID)
rf = RandomForestModel.load(str(models_dir(DATASET_ID)/'rf_shap_model.pkl'))
y_pred = rf.predict(data.X_test)
y_proba = positive_class_proba(rf.predict_proba(data.X_test))
metrics = compute_metrics(data.y_test, y_pred, y_proba, cfg.display_name)
print(metrics)
"""),
        code("""
stem = reports_figures_dir()/DATASET_ID
ieee_bar_chart(['Acc','Prec','Rec','F1','AUC'],
    [metrics['accuracy'],metrics['precision'],metrics['recall'],metrics['f1'],metrics['auc_roc']],
    f'{cfg.display_name} Test Metrics', 'Score', stem/'bar_metrics')
cm = confusion_matrix(data.y_test, y_pred, labels=[0,1])
ieee_confusion_matrix(cm, list(cfg.class_names), 'Confusion Matrix', stem/'confusion_matrix')
fpr,tpr,_ = roc_curve(data.y_test, y_proba)
ieee_roc_curve(fpr, tpr, auc(fpr,tpr), 'ROC', stem/'roc_curve')
print('IEEE figures saved to', stem)
"""),
    ])


def nb_benchmark() -> dict:
    return nb([
        md("# 05 — Cross-Dataset Benchmark Report\n\nAggregates all 6 datasets into benchmark tables and PDF."),
        code("import sys, os\nsys.path.insert(0, os.path.abspath('..'))\nfrom src.reporting.benchmark_report import generate_benchmark_report\nresult = generate_benchmark_report()\nprint(result)"),
        code("from pathlib import Path\nfrom IPython.display import Image, display\npdf = Path('reports/XMalDetect_Benchmark_Report.pdf')\ndash = Path(result['benchmark_dir'])/'figures'/'fig_master_dashboard.png'\nif dash.exists(): display(Image(filename=str(dash)))\nprint('PDF:', pdf)"),
    ])


def generate(ds_id: str) -> None:
    cfg = get_dataset(ds_id)
    out = ROOT / "notebooks" / ds_id
    out.mkdir(parents=True, exist_ok=True)
    for name, builder in [
        ("01_EDA.ipynb", lambda: nb_eda(ds_id, cfg)),
        ("02_Preprocess.ipynb", lambda: nb_preprocess(ds_id, cfg)),
        ("03_Train.ipynb", lambda: nb_train(ds_id, cfg)),
        ("04_Test.ipynb", lambda: nb_test(ds_id, cfg)),
    ]:
        with open(out / name, "w", encoding="utf-8") as f:
            json.dump(builder(), f, indent=1)
        print(f"  {out.name}/{name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    args = parser.parse_args()
    targets = [args.dataset] if args.dataset else list(ACTIVE_DATASETS)
    print(f"Generating notebooks for {len(targets)} datasets...")
    for ds in targets:
        if ds not in DATASETS:
            raise SystemExit(f"Unknown: {ds}")
        print(f"\n[{ds}]")
        generate(ds)
    bench = ROOT / "notebooks" / "benchmark"
    bench.mkdir(parents=True, exist_ok=True)
    with open(bench / "05_Benchmark_Report.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb_benchmark(), f, indent=1)
    print("  benchmark/05_Benchmark_Report.ipynb")
    print("Done.")


if __name__ == "__main__":
    main()
