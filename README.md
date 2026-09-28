# Explainable AI in Malware Detection: Bridging AI and Cybersecurity

**XMalDetect** is a reproducible machine learning research framework for **explainable malware and intrusion detection**. It combines Random Forest classification with **SHAP (SHapley Additive exPlanations)** to deliver strong detection performance *and* per-sample forensic attribution — connecting modern AI with practical cybersecurity workflows.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)
[![Tests](https://img.shields.io/badge/tests-23%20passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

---

## Abstract

Malware detection systems often operate as opaque black boxes, limiting trust and incident response. This project benchmarks an **explainable-by-design** pipeline across **six validated security datasets** spanning Android permissions, Windows PE static analysis, network intrusion, and ransomware behavioral traces. Each dataset is processed, trained, evaluated, and explained independently; results are aggregated into a unified benchmark with IEEE-compliant visualizations and a comprehensive PDF report.

**Key contributions:**
- End-to-end pipeline: preprocessing → RF training → SHAP explainability → forensic reporting
- Cross-domain benchmark (Android, PE, network, behavioral) with unified metrics
- Publication-ready figures (300+ DPI, vector PDF/SVG/EPS)
- Modular `src/` library with per-dataset Jupyter notebooks and 23 automated tests

---

## Benchmark Results (Random Forest + SHAP)

| Dataset | Domain | Train N | Test N | F1 | AUC-ROC |
|---------|--------|---------|--------|-----|---------|
| DREBIN | Android | 2,560 | 400 | **0.981** | **0.999** |
| EMBER | PE (Windows) | 560,000 | 160,000 | **0.936** | **0.983** |
| NSL-KDD | Network | 30,000 | 22,544 | **0.747** | **0.961** |
| MLRan | Behavioral | 3,905 | 975 | **0.966** | **0.995** |
| MH-1M | Android | 8,000 | 2,000 | **0.992** | **1.000** |
| CICIDS-2017 | Network | 40,000 | 10,000 | **0.999** | **1.000** |

**Mean F1:** 0.937 · **Mean AUC:** 0.990 · **Datasets completed:** 6/6

Full tables: [`reports/benchmark/tables/benchmark_summary.csv`](reports/benchmark/tables/benchmark_summary.csv)  
PDF report: [`reports/XMalDetect_Benchmark_Report.pdf`](reports/XMalDetect_Benchmark_Report.pdf)

---

## Repository Structure

```
xmaldetect/
├── data/processed/{dataset}/     # Preprocessed arrays metadata (summaries, feature names)
├── models/{dataset}/             # Trained rf_shap_model.pkl (Git LFS)
├── notebooks/{dataset}/          # 01_EDA, 02_Preprocess, 03_Train, 04_Test
├── notebooks/benchmark/          # 05_Benchmark_Report.ipynb
├── reports/
│   ├── figures/{dataset}/        # IEEE vector + raster figures
│   ├── benchmark/                # Master tables + cross-dataset dashboard
│   └── XMalDetect_Benchmark_Report.pdf
├── src/                          # Core library (io, core, aggregate, reporting)
├── tests/                        # pytest suite (23 tests)
├── run_benchmark.py              # Full pipeline entry point
├── generate_notebooks.py         # Regenerate all notebooks
├── README.md                     # This file
├── GUIDE.md                      # Step-by-step beginner walkthrough
└── docs/MASTER_OUTPUTS.md        # Artifact reference
```

---

## Quick Start

```powershell
git clone https://github.com/m09xd/explainable-ai-malware-detection-bridging-ai-and-cybersecurity.git
cd explainable-ai-malware-detection-bridging-ai-and-cybersecurity
git lfs #pull to download the model weights.

git lfs #pull to download the model weights.

python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Run full benchmark (preprocess → train → SHAP → IEEE figures → PDF)
python run_benchmark.py

# Regenerate report only (from existing artifacts)
python run_benchmark.py --report-only

# Run tests
python -m pytest tests/ -v
```

> **Note:** Trained models are stored with **Git LFS** (EMBER model ≈ 756 MB). After cloning, run `git lfs pull` to download model weights.

---

## Methodology

1. **Preprocessing** — constant-feature removal, optional `StandardScaler`, SMOTE for imbalanced splits
2. **Training** — Random Forest (200 trees, balanced class weights)
3. **Evaluation** — accuracy, precision, recall, F1, AUC-ROC, average precision
4. **Explainability** — TreeSHAP feature attributions; top-20 features exported per dataset
5. **Visualization** — IEEE-style exports via `src/core/ieee_style.py` (300 DPI, serif fonts, minimal gridlines)

---

## Datasets

| ID | Domain | Description |
|----|--------|-------------|
| `drebin` | Android | Permission-based APK malware (545 binary features) |
| `ember` | PE | Windows executable static features |
| `nsl-kdd` | Network | Classic intrusion detection benchmark |
| `mlran` | Behavioral | Ransomware API / behavioral traces |
| `mh-1m` | Android | Large-scale hypergraph Android features |
| `cicids-2017` | Network | Modern IDS flow features |

Raw CSV inputs are **not** bundled (licensing/size). Place files under `data/` and see `GUIDE.md` for naming conventions, or re-run preprocessing notebooks.

---

## Notebook Workflow

For each dataset in `notebooks/{dataset}/`:

| Notebook | Purpose |
|----------|---------|
| `01_EDA` | Class balance, feature overview |
| `02_Preprocess` | Clean, split, save to `data/processed/` |
| `03_Train` | Train RF+SHAP, save to `models/` |
| `04_Test` | Metrics + IEEE confusion matrix, ROC, bar charts |

Cross-dataset comparison: `notebooks/benchmark/05_Benchmark_Report.ipynb`

---

## Security & AI Context

This project sits at the intersection of **machine learning** and **computer security**:

- **Detection** — supervised binary classification across heterogeneous malware representations
- **Explainability** — SHAP values map predictions to human-readable features (permissions, PE headers, flow statistics)
- **Forensics** — Android datasets produce HTML attribution reports for analyst review
- **Benchmarking** — rigorous cross-dataset comparison for research reproducibility

---

## References

- Lundberg & Lee (2017) — *A Unified Approach to Interpreting Model Predictions* (SHAP)
- Arp et al. (2014) — *DREBIN: Effective and Explainable Detection of Android Malware*
- Anderson & Roth (2018) — *EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models*
- Tavallaee et al. (2009) — *A Detailed Analysis of the KDD Cup 99 Data Set* (NSL-KDD)

---

## Author

**m09xd** — Machine Learning & Cybersecurity Research  
UIU · BSc Computer Science & Engineering

---

## License

MIT License — see [LICENSE](LICENSE).
