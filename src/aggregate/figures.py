"""Unified cross-dataset figures and master dashboard."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve

from src.core.evaluation import MODEL_COLORS, DEFAULT_COLORS
from src.io.registry import master_output_dir

from .collect import DatasetRunRecord, collect_all_results
from .tables import build_dataset_summary, build_model_comparison_long, build_model_pivot

DOMAIN_COLORS = {
    "android": "#2ca02c",
    "network": "#1f77b4",
    "pe": "#9467bd",
    "behavioral": "#ff7f0e",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})


def _short_name(name: str, max_len: int = 18) -> str:
  return name if len(name) <= max_len else name[: max_len - 1] + "."


def plot_rf_metrics_by_dataset(
    summary_df: pd.DataFrame,
    save_path: Path,
    dpi: int = 300,
) -> None:
    """Grouped bar: RF+SHAP F1 and AUC per dataset."""
    done = summary_df[summary_df["status"] == "complete"].copy()
    if done.empty:
        return

    x = np.arange(len(done))
    w = 0.35
    colors = [DOMAIN_COLORS.get(d, "#888888") for d in done["domain"]]

    fig, ax = plt.subplots(figsize=(max(10, len(done) * 1.2), 6))
    b1 = ax.bar(x - w / 2, done["rf_f1"], w, label="F1", color=colors, alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w / 2, done["rf_auc_roc"], w, label="AUC-ROC", color=colors, alpha=0.55,
                edgecolor="white", hatch="//")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([_short_name(i) for i in done.index], rotation=35, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Random Forest + SHAP: F1 and AUC-ROC Across All Datasets")
    ax.legend(loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # domain legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=DOMAIN_COLORS[d]) for d in sorted(set(done["domain"]))]
    labels = sorted(set(done["domain"]))
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.legend(handles, labels, title="Domain", loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Aggregate] Figure saved -> {save_path}")


def plot_model_heatmap(
    pivot_df: pd.DataFrame,
    metric_label: str,
    save_path: Path,
    dpi: int = 300,
) -> None:
    """Heatmap: datasets (rows) x models (cols)."""
    if pivot_df.empty:
        return

    # Shorten model column names
    short_cols = {c: _short_name(c.replace(" (Black-Box Baseline)", "").replace(" (Partial Baseline)", ""), 22)
                  for c in pivot_df.columns}
    plot_df = pivot_df.rename(columns=short_cols)

    fig_h = max(5, len(plot_df) * 0.55)
    fig_w = max(10, len(plot_df.columns) * 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        plot_df.astype(float),
        annot=True, fmt=".3f", cmap="YlOrRd",
        vmin=0.5, vmax=1.0,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": metric_label},
    )
    ax.set_title(f"Model Comparison Heatmap — {metric_label} (All Datasets)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Dataset")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Aggregate] Figure saved -> {save_path}")


def plot_domain_summary(
    long_df: pd.DataFrame,
    save_path: Path,
    dpi: int = 300,
) -> None:
    """Bar chart: mean RF+SHAP F1/AUC by domain."""
    rf = long_df[long_df["model"].str.contains("SHAP", na=False)]
    if rf.empty:
        return

    domain_stats = rf.groupby("domain")[["f1", "auc_roc"]].mean()
    x = np.arange(len(domain_stats))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [DOMAIN_COLORS.get(d, "#888") for d in domain_stats.index]
    ax.bar(x - w / 2, domain_stats["f1"], w, label="Mean F1", color=colors, alpha=0.9)
    ax.bar(x + w / 2, domain_stats["auc_roc"], w, label="Mean AUC", color=colors, alpha=0.5, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(domain_stats.index.str.title())
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Mean RF+SHAP Performance by Malware Domain")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Aggregate] Figure saved -> {save_path}")


def plot_roc_grid(
    records: list[DatasetRunRecord],
    save_path: Path,
    dpi: int = 200,
) -> None:
    """Grid of ROC curves (RF+SHAP only) — one subplot per dataset."""
    done = [r for r in records if r.roc_curves and r.status == "complete"]
    if not done:
        return

    n = len(done)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, rec in enumerate(done):
        ax = axes[i]
        # Prefer RF+SHAP curve
        model_key = next(
            (k for k in rec.roc_curves if "shap" in k.lower()),
            next(iter(rec.roc_curves)),
        )
        curve = rec.roc_curves[model_key]
        fpr, tpr = curve["fpr"], curve["tpr"]
        roc_auc = curve.get("auc", auc(fpr, tpr))
        color = DOMAIN_COLORS.get(rec.domain, "#d62728")
        ax.plot(fpr, tpr, color=color, lw=2, label=f"AUC={roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.5)
        ax.set_title(f"{rec.dataset_id}\n({rec.domain})", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        if i % cols == 0:
            ax.set_ylabel("TPR")
        if i >= (rows - 1) * cols:
            ax.set_xlabel("FPR")
        ax.legend(fontsize=7, loc="lower right")

    for j in range(len(done), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("ROC Curves — Random Forest + SHAP (All Datasets)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Aggregate] Figure saved -> {save_path}")


def plot_all_models_comparison(
    long_df: pd.DataFrame,
    save_path: Path,
    dpi: int = 300,
) -> None:
    """Grouped bar: each dataset shows all 4 models' F1 scores."""
    if long_df.empty:
        return

    datasets = long_df["dataset"].unique()
    models = long_df["model"].unique()
    x = np.arange(len(datasets))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(max(12, len(datasets) * 1.5), 6))
    for i, model in enumerate(models):
        sub = long_df[long_df["model"] == model].set_index("dataset")
        vals = [sub.loc[d, "f1"] if d in sub.index else 0 for d in datasets]
        color = MODEL_COLORS.get(model, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, label=_short_name(model, 25), color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=35, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 Score")
    ax.set_title("All Models — F1 Score Across All Datasets")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Aggregate] Figure saved -> {save_path}")


def plot_master_dashboard(
    records: list[DatasetRunRecord],
    summary_df: pd.DataFrame,
    long_df: pd.DataFrame,
    pivot_f1: pd.DataFrame,
    save_path: Path,
    dpi: int = 250,
) -> None:
    """
    Single comprehensive figure combining:
      A) RF F1/AUC by dataset
      B) F1 heatmap (datasets x models)
      C) Domain summary
      D) Dataset status overview
    """
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    # Panel A — RF metrics
    ax_a = fig.add_subplot(gs[0, 0])
    done = summary_df[summary_df["status"] == "complete"]
    if not done.empty:
        x = np.arange(len(done))
        w = 0.35
        colors = [DOMAIN_COLORS.get(d, "#888") for d in done["domain"]]
        ax_a.bar(x - w / 2, done["rf_f1"], w, label="F1", color=colors, alpha=0.85)
        ax_a.bar(x + w / 2, done["rf_auc_roc"], w, label="AUC", color=colors, alpha=0.5, hatch="//")
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(done.index, rotation=40, ha="right", fontsize=8)
        ax_a.set_ylim(0, 1.1)
        ax_a.set_title("A) RF+SHAP: F1 & AUC by Dataset", fontweight="bold")
        ax_a.legend(fontsize=8)
        ax_a.yaxis.grid(True, linestyle="--", alpha=0.4)

    # Panel B — heatmap
    ax_b = fig.add_subplot(gs[0, 1])
    if not pivot_f1.empty:
        short_cols = {c: _short_name(c, 16) for c in pivot_f1.columns}
        plot_hm = pivot_f1.rename(columns=short_cols)
        sns.heatmap(plot_hm.astype(float), annot=True, fmt=".2f", cmap="YlOrRd",
                    vmin=0.5, vmax=1.0, ax=ax_b, cbar_kws={"shrink": 0.8})
        ax_b.set_title("B) F1 Heatmap: Datasets x Models", fontweight="bold")
        ax_b.set_xlabel("")
        plt.setp(ax_b.get_xticklabels(), rotation=35, ha="right", fontsize=7)

    # Panel C — all models F1 grouped
    ax_c = fig.add_subplot(gs[1, 0])
    if not long_df.empty:
        datasets = long_df["dataset"].unique()
        models = long_df["model"].unique()
        x = np.arange(len(datasets))
        width = 0.8 / max(len(models), 1)
        for i, model in enumerate(models):
            sub = long_df[long_df["model"] == model].set_index("dataset")
            vals = [sub.loc[d, "f1"] if d in sub.index else np.nan for d in datasets]
            color = MODEL_COLORS.get(model, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
            offset = (i - len(models) / 2 + 0.5) * width
            ax_c.bar(x + offset, vals, width * 0.9, label=_short_name(model, 18),
                     color=color, alpha=0.85)
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(datasets, rotation=40, ha="right", fontsize=8)
        ax_c.set_ylim(0, 1.1)
        ax_c.set_title("C) All Models F1 Comparison", fontweight="bold")
        ax_c.legend(fontsize=6, ncol=2, loc="upper right")
        ax_c.yaxis.grid(True, linestyle="--", alpha=0.4)

    # Panel D — status table as text
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")
    table_data = []
    for ds_id, row in summary_df.iterrows():
        rf_f1 = f"{row['rf_f1']:.4f}" if pd.notna(row.get("rf_f1")) else "—"
        rf_auc = f"{row['rf_auc_roc']:.4f}" if pd.notna(row.get("rf_auc_roc")) else "—"
        train_n = row.get("train_samples", "—")
        test_n = row.get("test_samples", "—")
        table_data.append([ds_id, row["domain"], row["status"], train_n, test_n, rf_f1, rf_auc])

    if table_data:
        tbl = ax_d.table(
            cellText=table_data,
            colLabels=["Dataset", "Domain", "Status", "Train N", "Test N", "RF F1", "RF AUC"],
            loc="center", cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.4)
        ax_d.set_title("D) Dataset Run Summary", fontweight="bold", pad=20)

    n_complete = (summary_df["status"] == "complete").sum()
    n_total = len(summary_df)
    fig.suptitle(
        f"XMalDetect — Master Cross-Dataset Dashboard  "
        f"({n_complete}/{n_total} datasets complete)",
        fontsize=15, fontweight="bold", y=1.01,
    )
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Aggregate] MASTER DASHBOARD saved -> {save_path}")


def build_master_figures(
    records: list[DatasetRunRecord] | None = None,
    root: Path | None = None,
) -> list[Path]:
    """Generate all unified figures. Returns list of saved paths."""
    if records is None:
        records = collect_all_results(root)

    out_dir = master_output_dir(root) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df = build_model_comparison_long(records)
    summary_df = build_dataset_summary(records)
    pivot_f1 = build_model_pivot(long_df, "f1")
    pivot_auc = build_model_pivot(long_df, "auc_roc")

    paths = []
    specs = [
        ("fig_master_rf_metrics.png", lambda p: plot_rf_metrics_by_dataset(summary_df, p)),
        ("fig_master_heatmap_f1.png", lambda p: plot_model_heatmap(pivot_f1, "F1", p)),
        ("fig_master_heatmap_auc.png", lambda p: plot_model_heatmap(pivot_auc, "AUC-ROC", p)),
        ("fig_master_domain_summary.png", lambda p: plot_domain_summary(long_df, p)),
        ("fig_master_all_models_f1.png", lambda p: plot_all_models_comparison(long_df, p)),
        ("fig_master_roc_grid.png", lambda p: plot_roc_grid(records, p)),
        ("fig_master_dashboard.png", lambda p: plot_master_dashboard(
            records, summary_df, long_df, pivot_f1, p)),
    ]

    for fname, fn in specs:
        path = out_dir / fname
        fn(path)
        if path.exists():
            paths.append(path)

    return paths
