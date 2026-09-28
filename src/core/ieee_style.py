"""
IEEE-compliant matplotlib styling and multi-format figure export.

IEEE figure requirements implemented:
  - Min 300 DPI (600 for text-heavy)
  - Vector formats: PDF, SVG, EPS
  - High-quality raster: PNG, TIFF
  - Serif fonts (Times-like), minimal gridlines
  - Readable axis labels (8-10 pt equivalent)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IEEE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
}

IEEE_COLORS = {
    "primary": "#000000",
    "secondary": "#404040",
    "accent1": "#0072BD",
    "accent2": "#D95319",
    "accent3": "#77AC30",
    "accent4": "#7E2F8E",
    "benign": "#0072BD",
    "malware": "#D95319",
}


def apply_ieee_style() -> None:
    plt.rcParams.update(IEEE_RC)


def save_ieee_figure(
    fig: plt.Figure,
    stem: str | Path,
    dpi_raster: int = 300,
    dpi_text_heavy: int = 600,
    text_heavy: bool = False,
) -> dict[str, Path]:
    """
    Export figure to PDF, SVG, EPS, PNG, and TIFF.

    Parameters
    ----------
    stem : path without extension, e.g. reports/figures/drebin/confusion_matrix
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    dpi = dpi_text_heavy if text_heavy else dpi_raster
    saved = {}
    for fmt, ext in [("pdf", ".pdf"), ("svg", ".svg"), ("eps", ".eps"),
                     ("png", ".png"), ("tiff", ".tiff")]:
        path = stem.with_suffix(ext)
        try:
            fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            saved[fmt] = path
        except Exception:
            pass
    return saved


def ieee_bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    stem: Path,
    colors: list[str] | None = None,
) -> dict[str, Path]:
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    cols = colors or [IEEE_COLORS["accent1"]] * len(labels)
    ax.bar(range(len(labels)), values, color=cols, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, min(1.15, max(values) * 1.2 + 0.05) if values else 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    saved = save_ieee_figure(fig, stem)
    plt.close(fig)
    return saved


def ieee_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    stem: Path,
) -> dict[str, Path]:
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    saved = save_ieee_figure(fig, stem, text_heavy=True)
    plt.close(fig)
    return saved


def ieee_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    title: str,
    stem: Path,
) -> dict[str, Path]:
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot(fpr, tpr, color=IEEE_COLORS["accent1"], lw=1.5,
            label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    saved = save_ieee_figure(fig, stem)
    plt.close(fig)
    return saved


def ieee_scatter(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    stem: Path,
) -> dict[str, Path]:
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.scatter(x, y, c=IEEE_COLORS["accent1"], s=12, alpha=0.6, edgecolors="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    saved = save_ieee_figure(fig, stem)
    plt.close(fig)
    return saved
