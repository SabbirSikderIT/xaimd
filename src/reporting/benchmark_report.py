"""
Generate IEEE-style benchmark report (Markdown + PDF).

Outputs:
  reports/XMalDetect_Benchmark_Report.md
  reports/XMalDetect_Benchmark_Report.pdf
  reports/benchmark/ (tables + master figures)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from src.aggregate.report import aggregate_all
from src.aggregate.tables import build_dataset_summary
from src.aggregate.collect import collect_all_results
from src.core.ieee_style import apply_ieee_style
from src.io.paths import benchmark_dir, reports_dir
from src.io.registry import ACTIVE_DATASETS, _project_root


def _write_markdown(summary_df: pd.DataFrame, benchmark_path: Path) -> Path:
    md_path = reports_dir() / "XMalDetect_Benchmark_Report.md"
    done = summary_df[summary_df["status"] == "complete"]

    lines = [
        "# XMalDetect: Cross-Dataset Malware Detection Benchmark",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## 1. Project Overview",
        "",
        "XMalDetect is an explainable malware detection framework using "
        "Random Forest + SHAP TreeExplainer. This benchmark compares "
        f"performance across **{len(ACTIVE_DATASETS)}** validated datasets.",
        "",
        "### Active Datasets",
        "",
        "| ID | Domain | Train N | Test N | RF F1 | RF AUC |",
        "|----|--------|---------|--------|-------|--------|",
    ]
    for ds_id, row in summary_df.iterrows():
        f1 = f"{row['rf_f1']:.4f}" if pd.notna(row.get("rf_f1")) else "N/A"
        auc = f"{row['rf_auc_roc']:.4f}" if pd.notna(row.get("rf_auc_roc")) else "N/A"
        tr = int(row["train_samples"]) if pd.notna(row.get("train_samples")) else "-"
        te = int(row["test_samples"]) if pd.notna(row.get("test_samples")) else "-"
        lines.append(f"| {ds_id} | {row['domain']} | {tr} | {te} | {f1} | {auc} |")

    lines += [
        "",
        "## 2. Methodology",
        "",
        "1. **Preprocessing:** constant-feature removal, optional StandardScaler, "
        "SMOTE for imbalanced splits.",
        "2. **Training:** Random Forest (200 trees, balanced class weights).",
        "3. **Testing:** stratified hold-out or official test split.",
        "4. **Explainability:** TreeSHAP on primary RF model.",
        "",
        "## 3. Benchmark Summary",
        "",
    ]
    if not done.empty:
        lines.append(f"- **Mean RF F1:** {done['rf_f1'].mean():.4f}")
        lines.append(f"- **Mean RF AUC:** {done['rf_auc_roc'].mean():.4f}")
        lines.append(f"- **Datasets completed:** {len(done)}/{len(ACTIVE_DATASETS)}")

    lines += [
        "",
        "## 4. Figures",
        "",
        f"Master dashboard: `{benchmark_path / 'figures' / 'fig_master_dashboard.png'}`",
        "",
        "Per-dataset IEEE figures: `reports/figures/{dataset}/`",
        "",
        "## 5. Conclusions",
        "",
        "RF+SHAP achieves competitive F1/AUC across Android, network, PE, "
        "and behavioral malware domains while providing per-sample SHAP "
        "explanations for forensic analysis.",
        "",
        "## 6. Future Work",
        "",
        "- Integrate corrected DMBD/BODMAS label files",
        "- Hyperparameter tuning per domain",
        "- Deep learning baselines with calibration",
    ]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def _add_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.92, title, ha="center", fontsize=14, fontweight="bold")
    y = 0.84
    for line in lines:
        fig.text(0.08, y, line, ha="left", va="top", fontsize=10, wrap=True)
        y -= 0.045
        if y < 0.05:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8.5, 11))
            y = 0.92
    pdf.savefig(fig)
    plt.close(fig)


def _embed_image_page(pdf: PdfPages, image_path: Path, title: str) -> None:
    if not image_path.exists():
        return
    img = plt.imread(str(image_path))
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    pdf.savefig(fig)
    plt.close(fig)


def _write_pdf(summary_df: pd.DataFrame, benchmark_path: Path) -> Path:
    pdf_path = reports_dir() / "XMalDetect_Benchmark_Report.pdf"
    apply_ieee_style()
    root = _project_root()
    figures_root = reports_dir() / "figures"

    with PdfPages(str(pdf_path)) as pdf:
        # Title page
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.7, "XMalDetect Benchmark Report",
                 ha="center", fontsize=18, fontweight="bold")
        fig.text(0.5, 0.6, f"Datasets: {', '.join(ACTIVE_DATASETS)}",
                 ha="center", fontsize=10)
        fig.text(0.5, 0.5, datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                 ha="center", fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

        _add_text_page(pdf, "1. Project Overview", [
            "XMalDetect is an explainable malware detection framework using",
            "Random Forest with SHAP TreeExplainer for per-sample forensic attribution.",
            f"This benchmark evaluates {len(ACTIVE_DATASETS)} validated datasets spanning",
            "Android, network intrusion, PE static, and behavioral ransomware domains.",
        ])

        _add_text_page(pdf, "2. Methodology", [
            "Preprocessing: remove constant features, optional StandardScaler,",
            "SMOTE for imbalanced training splits (when enabled per dataset).",
            "Training: Random Forest (200 estimators, balanced class weights).",
            "Testing: stratified hold-out or official test split per dataset registry.",
            "Explainability: TreeSHAP on the primary RF model; top-20 features exported.",
            "Figures: IEEE-compliant vector exports (PDF/SVG/EPS) at 300+ DPI.",
        ])

        # Summary table page
        done = summary_df[summary_df["status"] == "complete"]
        if not done.empty:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            cols = ["domain", "train_samples", "test_samples",
                    "rf_accuracy", "rf_precision", "rf_recall", "rf_f1", "rf_auc_roc"]
            tbl_data = []
            for ds_id, row in done.iterrows():
                tbl_data.append([ds_id] + [
                    f"{row.get(c):.4f}" if c.startswith("rf_") and pd.notna(row.get(c))
                    else str(row.get(c, ""))
                    for c in cols
                ])
            table = ax.table(
                cellText=tbl_data,
                colLabels=["Dataset"] + cols,
                loc="center", cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1, 1.3)
            ax.set_title("3. Benchmark Table — Random Forest + SHAP", fontsize=12, pad=20)
            pdf.savefig(fig)
            plt.close(fig)

        # Master figures
        for fname, title in [
            ("fig_master_dashboard.png", "4. Cross-Dataset Performance Dashboard"),
            ("fig_master_rf_metrics.png", "5. RF+SHAP F1 and AUC by Dataset"),
            ("fig_master_heatmap_f1.png", "6. F1 Heatmap (Datasets x Models)"),
            ("fig_master_all_models_f1.png", "7. All Models F1 Comparison"),
            ("fig_master_roc_grid.png", "8. ROC Curves Grid"),
        ]:
            _embed_image_page(pdf, benchmark_path / "figures" / fname, title)

        # Per-dataset IEEE figures
        for ds_id, row in summary_df.iterrows():
            if row.get("status") != "complete":
                continue
            ds_fig = figures_root / ds_id
            if not ds_fig.exists():
                continue
            display = row.get("display_name", ds_id)
            for stem, caption in [
                ("bar_metrics", "Test Metrics Bar Chart"),
                ("confusion_matrix", "Confusion Matrix"),
                ("roc_curve", "ROC Curve"),
                ("scatter_confidence", "Confidence vs Prediction"),
            ]:
                for ext in (".png", ".pdf"):
                    path = ds_fig / f"{stem}{ext}"
                    if path.exists():
                        _embed_image_page(
                            pdf, path,
                            f"{display} — {caption}",
                        )
                        break

        _add_text_page(pdf, "9. Conclusions", [
            f"Completed datasets: {(summary_df['status'] == 'complete').sum()}/{len(ACTIVE_DATASETS)}.",
            f"Mean RF F1: {done['rf_f1'].mean():.4f}" if not done.empty else "Mean RF F1: N/A",
            f"Mean RF AUC: {done['rf_auc_roc'].mean():.4f}" if not done.empty else "Mean RF AUC: N/A",
            "RF+SHAP delivers strong discrimination with interpretable feature attributions.",
            "Future work: corrected DMBD/BODMAS labels, per-domain HPO, deep baselines.",
        ])

    return pdf_path


def generate_benchmark_report(root: Path | None = None, verbose: bool = True) -> dict:
    """Full pipeline: aggregate results, write MD + PDF report."""
    root = root or _project_root()
    agg = aggregate_all(root, verbose=verbose)
    records = collect_all_results(root)
    summary_df = build_dataset_summary(records)
    bench = benchmark_dir(root)

    md_path = _write_markdown(summary_df, bench)
    pdf_path = _write_pdf(summary_df, bench)

    # Save benchmark CSV
    csv_path = bench / "tables" / "benchmark_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(csv_path)

    result = {
        "markdown": str(md_path),
        "pdf": str(pdf_path),
        "benchmark_csv": str(csv_path),
        "benchmark_dir": str(bench),
        **agg,
    }
    if verbose:
        print(f"[Report] Markdown -> {md_path}")
        print(f"[Report] PDF      -> {pdf_path}")
    return result
