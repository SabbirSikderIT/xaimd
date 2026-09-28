"""
evaluation.py
=============
XMalDetect — Metrics, Evaluation & Paper Figure Generator

Produces every figure and table referenced in the paper:
  Fig. 6  — Grouped bar chart: F1 + AUC across all models
  Fig. 7  — ROC curves (all models, single plot)
  Fig. 8  — Confusion matrix for RF + SHAP
  Table 3 — Classification performance comparison CSV
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, auc,
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi"     : 150,
    "font.family"    : "DejaVu Sans",
    "axes.titlesize" : 13,
    "axes.labelsize" : 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

MODEL_COLORS = {
    "Random Forest + SHAP"                     : "#d62728",
    "RF + Gini Importance (Partial Baseline)"  : "#ff7f0e",
    "SVM (Black-Box Baseline)"                 : "#1f77b4",
    "MLP Neural Network (Black-Box Baseline)"  : "#9467bd",
}
DEFAULT_COLORS = list(MODEL_COLORS.values())

DEFAULT_FIG_DIR = "outputs/figures"


# ──────────────────────────────────────────────────────────────────────────────
# Core metric computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray, model_name: str = "") -> dict:
    """
    Compute all paper metrics for one model.
    Returns a dict with keys matching Table 3 column headers.
    """
    return {
        "model"    : model_name,
        "accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall"   : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1"       : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc"  : round(roc_auc_score(y_true, y_proba), 4),
        "avg_prec" : round(average_precision_score(y_true, y_proba), 4),
    }


def cross_val_metrics(estimator, X: np.ndarray, y: np.ndarray,
                      cv: int = 5, model_name: str = "") -> pd.DataFrame:
    """
    Run stratified k-fold CV and return mean ± std for key metrics.
    Used to populate the CV column in Table 3.
    """
    skf     = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = cross_validate(
        estimator, X, y,
        cv=skf, scoring=scoring,
        return_train_score=False, n_jobs=-1,
    )
    summary = {"model": model_name}
    for s in scoring:
        vals = results[f"test_{s}"]
        summary[f"cv_{s}"] = f"{vals.mean():.4f} ± {vals.std():.4f}"
    return pd.DataFrame([summary])


# ──────────────────────────────────────────────────────────────────────────────
# Fig. 6 — Grouped Bar Chart: F1 + AUC  (paper §6.1)
# ──────────────────────────────────────────────────────────────────────────────

def plot_model_comparison_bar(
    results_list : list[dict],
    metrics      : list[str] = ["f1", "auc_roc", "accuracy", "precision", "recall"],
    save_path    : str = f"{DEFAULT_FIG_DIR}/fig6_model_comparison_bar.png",
    dpi          : int = 300,
) -> None:
    """
    Fig. 6 — Grouped bar chart comparing all models across multiple metrics.

    Parameters
    ----------
    results_list : list of dicts from compute_metrics()
    metrics      : which metrics to plot (columns from results dict)
    """
    df = pd.DataFrame(results_list).set_index("model")
    df = df[[m for m in metrics if m in df.columns]]

    n_models  = len(df)
    n_metrics = len(df.columns)
    x         = np.arange(n_metrics)
    width     = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (model_name, row) in enumerate(df.iterrows()):
        color  = MODEL_COLORS.get(model_name, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        offset = (i - n_models / 2 + 0.5) * width
        bars   = ax.bar(x + offset, row.values, width * 0.9,
                        label=model_name, color=color, alpha=0.85,
                        edgecolor="white", linewidth=0.5)
        # value labels
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=7.5, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper().replace("_", "-") for m in df.columns],
                       fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Fig. 6 — Model Performance Comparison (Table 3)\n"
        "RF + SHAP achieves competitive accuracy WITH full explainability",
        fontsize=11, pad=10,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Fig 6 saved -> {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Fig. 7 — ROC Curves  (paper §6.1)
# ──────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    roc_data  : list[dict],
    save_path : str = f"{DEFAULT_FIG_DIR}/fig7_roc_curves.png",
    dpi       : int = 300,
) -> None:
    """
    Fig. 7 — ROC curves for all models on a single plot.

    Parameters
    ----------
    roc_data : list of dicts, each with keys:
               'name'   : model name string
               'y_true' : ground-truth labels
               'y_proba': predicted probabilities for class 1
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, entry in enumerate(roc_data):
        fpr, tpr, _ = roc_curve(entry["y_true"], entry["y_proba"])
        roc_auc     = auc(fpr, tpr)
        color       = MODEL_COLORS.get(entry["name"],
                                       DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        lw          = 2.5 if "SHAP" in entry["name"] else 1.5
        ls          = "-"  if "SHAP" in entry["name"] else "--"
        ax.plot(fpr, tpr, color=color, lw=lw, ls=ls,
                label=f"{entry['name']} (AUC = {roc_auc:.4f})")

    # Random baseline
    ax.plot([0, 1], [0, 1], "k:", lw=1.2, label="Random Classifier (AUC = 0.5)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title(
        "Fig. 7 — ROC Curves: All Models\n"
        "Solid line = RF + SHAP (primary model)",
        fontsize=11, pad=10,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Fig 7 saved -> {save_path}")


def plot_precision_recall_curves(
    pr_data   : list[dict],
    save_path : str = f"{DEFAULT_FIG_DIR}/fig7b_pr_curves.png",
    dpi       : int = 300,
) -> None:
    """
    Precision-Recall curves — important for imbalanced datasets (supplement to ROC).
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, entry in enumerate(pr_data):
        prec, rec, _ = precision_recall_curve(entry["y_true"], entry["y_proba"])
        ap           = average_precision_score(entry["y_true"], entry["y_proba"])
        color        = MODEL_COLORS.get(entry["name"],
                                        DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{entry['name']} (AP = {ap:.4f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves — All Models\n"
                 "(Supplement to Fig. 7 for imbalanced dataset context)",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Eval] PR curves saved -> {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Fig. 8 — Confusion Matrix  (paper §6.1)
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true      : np.ndarray,
    y_pred      : np.ndarray,
    model_name  : str = "RF + SHAP",
    class_names : list[str] = ["Benign", "Malware"],
    save_path   : str = f"{DEFAULT_FIG_DIR}/fig8_confusion_matrix.png",
    dpi         : int = 300,
    normalize   : bool = False,
) -> np.ndarray:
    """
    Fig. 8 — Annotated confusion matrix with counts and percentages.

    Parameters
    ----------
    normalize : if True, show percentages; if False, show raw counts
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt        = ".2%"
        title_sfx  = "(Normalised)"
    else:
        cm_display = cm
        fmt        = "d"
        title_sfx  = "(Raw Counts)"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_display,
        annot       = True,
        fmt         = fmt,
        cmap        = "Blues",
        xticklabels = class_names,
        yticklabels = class_names,
        linewidths  = 0.8,
        linecolor   = "white",
        ax          = ax,
        cbar_kws    = {"shrink": 0.8},
    )

    # Annotate with both count and % if not normalized
    if not normalize:
        total = cm.sum()
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            pct = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.72, f"({pct:.1f}%)",
                    ha="center", va="center", fontsize=9, color="grey")

    # Metric annotations
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    ax.set_xlabel(
        f"Predicted Label\n\n"
        f"False Positive Rate: {fpr:.4f}  |  False Negative Rate: {fnr:.4f}",
        fontsize=10,
    )
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(
        f"Fig. 8 — Confusion Matrix: {model_name} {title_sfx}\n"
        f"TP={tp}  TN={tn}  FP={fp}  FN={fn}",
        fontsize=11, pad=12,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Fig 8 saved -> {save_path}")
    return cm


# ──────────────────────────────────────────────────────────────────────────────
# Table 3 — Full comparison table builder
# ──────────────────────────────────────────────────────────────────────────────

def build_comparison_table(
    results_list: list[dict],
    cv_list     : list[pd.DataFrame] | None = None,
    save_path   : str = "outputs/comparison_table3.csv",
) -> pd.DataFrame:
    """
    Build Table 3 from the paper — full model comparison.

    Parameters
    ----------
    results_list : list of compute_metrics() dicts
    cv_list      : optional list of cross_val_metrics() DataFrames
    """
    df = pd.DataFrame(results_list)

    if cv_list:
        cv_combined = pd.concat(cv_list, ignore_index=True)
        df = df.merge(cv_combined, on="model", how="left")

    # Add explainability column
    def _explainability(name):
        if "SHAP" in name:
            return "Full (TreeSHAP)"
        if "Gini" in name:
            return "Partial (Gini)"
        return "None"

    df["explainable"] = df["model"].apply(_explainability)
    df = df.set_index("model")

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                exist_ok=True)
    df.to_csv(save_path)
    print(f"[Eval] Table 3 saved -> {save_path}")
    print("\n" + "="*70)
    print("  TABLE 3 — PAPER: MODEL COMPARISON")
    print("="*70)
    print(df.to_string())
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Feature importance plot (Gini) — for §6.5 comparison baseline
# ──────────────────────────────────────────────────────────────────────────────

def plot_gini_importance(
    gini_df   : pd.DataFrame,
    top_n     : int = 20,
    save_path : str = f"{DEFAULT_FIG_DIR}/gini_importance.png",
    dpi       : int = 300,
) -> None:
    """Horizontal bar chart of Gini (MDI) feature importances."""
    df = gini_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        df["feature"][::-1],
        df["gini_importance"][::-1],
        color="#1f77b4", alpha=0.8, edgecolor="white",
    )
    ax.set_xlabel("Gini Importance (Mean Decrease in Impurity)", fontsize=11)
    ax.set_title(
        f"Gini Feature Importance — Top {top_n} Features\n"
        "(Baseline for §6.5 SHAP vs. Gini Comparison)",
        fontsize=11,
    )
    for bar, val in zip(bars, df["gini_importance"][::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Gini importance saved -> {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Class distribution plot (for EDA notebook)
# ──────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(
    y            : np.ndarray,
    dataset_name : str = "Dataset",
    save_path    : str = f"{DEFAULT_FIG_DIR}/class_distribution.png",
    dpi          : int = 300,
) -> None:
    """Bar chart of class balance. Shows imbalance ratio."""
    labels, counts = np.unique(y, return_counts=True)
    names  = ["Benign" if l == 0 else "Malware" for l in labels]
    colors = ["#1f77b4", "#d62728"][:len(labels)]
    total  = counts.sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(names, counts, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, cnt + total * 0.005,
                f"{cnt:,}\n({cnt/total:.1%})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ratio = max(counts) / min(counts) if min(counts) > 0 else float("inf")
    ax.set_title(
        f"Class Distribution — {dataset_name}\n"
        f"Imbalance Ratio: {ratio:.1f}:1 "
        f"({'Balanced' if ratio < 1.5 else 'Apply SMOTE / class_weight'})",
        fontsize=11,
    )
    ax.set_ylabel("Sample Count", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Class distribution saved -> {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Feature correlation heat map (for EDA notebook)
# ──────────────────────────────────────────────────────────────────────────────

def plot_feature_correlation(
    X            : np.ndarray,
    feature_names: list[str],
    top_n        : int = 30,
    save_path    : str = f"{DEFAULT_FIG_DIR}/feature_correlation.png",
    dpi          : int = 300,
) -> None:
    """
    Pearson correlation heat map for top_n most-variance features.
    Helps identify redundant / collinear permission pairs.
    """
    df   = pd.DataFrame(X, columns=feature_names)
    # Select top_n by variance (most informative for binary features)
    variances  = df.var().sort_values(ascending=False)
    top_feats  = variances.head(top_n).index.tolist()
    corr       = df[top_feats].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle mask
    sns.heatmap(
        corr, mask=mask, cmap="coolwarm", center=0,
        vmin=-1, vmax=1, annot=False,
        linewidths=0.3, ax=ax,
        cbar_kws={"label": "Pearson Correlation", "shrink": 0.8},
    )
    ax.set_title(
        f"Feature Correlation Heat Map (Top {top_n} by Variance)\n"
        "Highly correlated pairs may be redundant — compare with SHAP ranking",
        fontsize=11,
    )
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Correlation heatmap saved -> {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Full evaluation pipeline runner
# ──────────────────────────────────────────────────────────────────────────────

def run_full_evaluation(
    models_dict  : dict,
    X_train      : np.ndarray,
    y_train      : np.ndarray,
    X_test       : np.ndarray,
    y_test       : np.ndarray,
    feature_names: list[str],
    out_dir      : str = "outputs/figures",
) -> pd.DataFrame:
    """
    End-to-end evaluation runner.

    Parameters
    ----------
    models_dict : {model_name: trained_model_object}
                  Each object must have .predict() and .predict_proba() methods.

    Returns
    -------
    comparison_df : Table 3 DataFrame
    """
    os.makedirs(out_dir, exist_ok=True)
    results_list = []
    roc_data     = []
    pr_data      = []

    for name, model in models_dict.items():
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba, model_name=name)
        results_list.append(metrics)
        roc_data.append({"name": name, "y_true": y_test, "y_proba": y_proba})
        pr_data.append ({"name": name, "y_true": y_test, "y_proba": y_proba})

    # Generate all paper figures
    plot_model_comparison_bar(results_list,
                              save_path=f"{out_dir}/fig6_model_comparison_bar.png")
    plot_roc_curves(roc_data, save_path=f"{out_dir}/fig7_roc_curves.png")
    plot_precision_recall_curves(pr_data, save_path=f"{out_dir}/fig7b_pr_curves.png")

    # Confusion matrix for primary RF + SHAP model
    primary_name = [n for n in models_dict if "SHAP" in n]
    if primary_name:
        pm      = models_dict[primary_name[0]]
        y_pred  = pm.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, model_name=primary_name[0],
                              save_path=f"{out_dir}/fig8_confusion_matrix.png")

    # Comparison table
    comparison_df = build_comparison_table(
        results_list, save_path="outputs/comparison_table3.csv"
    )
    return comparison_df