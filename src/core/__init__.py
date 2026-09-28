"""Shared ML, SHAP, and evaluation modules."""

from .model import (
    BaseModel,
    RandomForestModel,
    RFGiniModel,
    SVMModel,
    MLPModel,
)
from .evaluation import (
    compute_metrics,
    cross_val_metrics,
    plot_model_comparison_bar,
    plot_roc_curves,
    plot_confusion_matrix,
    build_comparison_table,
    run_full_evaluation,
    plot_class_distribution,
    plot_feature_correlation,
    plot_gini_importance,
)
from .shap_engine import SHAPEngine

__all__ = [
    "BaseModel",
    "RandomForestModel",
    "RFGiniModel",
    "SVMModel",
    "MLPModel",
    "compute_metrics",
    "cross_val_metrics",
    "plot_model_comparison_bar",
    "plot_roc_curves",
    "plot_confusion_matrix",
    "build_comparison_table",
    "run_full_evaluation",
    "plot_class_distribution",
    "plot_feature_correlation",
    "plot_gini_importance",
    "SHAPEngine",
]
