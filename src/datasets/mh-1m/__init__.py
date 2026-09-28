"""MH-1M — large-scale Android hypergraph dataset."""

DATASET_ID = "mh-1m"

from src.pipeline.runner import run_forensic, run_shap, train_and_evaluate
from src.preprocessing.pipeline import run_preprocessing

__all__ = ["DATASET_ID", "run_preprocessing", "train_and_evaluate", "run_shap", "run_forensic"]
