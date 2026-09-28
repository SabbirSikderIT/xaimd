# XMalDetect — Complete Beginner's Guide

This guide assumes you have **never run a machine-learning project before**. Follow it step by step.

---

## What is this project?

**XMalDetect** trains machine-learning models to detect **malware** (malicious software) and explains **why** each sample was flagged using **SHAP** (a popular explainability method).

You will:
1. Load a dataset (CSV files in `data/`)
2. Train models (Random Forest, SVM, MLP)
3. Measure accuracy (F1, AUC, confusion matrix)
4. Generate SHAP plots (which features caused each prediction)
5. Produce forensic HTML reports (for Android datasets)

---

## Project layout (what each folder means)

```
xmaldetect/
├── data/                          # INPUT: processed CSV files (one set per dataset)
│   ├── drebin_features.csv        # Example: DREBIN feature matrix
│   ├── drebin_labels.csv          # Example: DREBIN labels (0=benign, 1=malware)
│   ├── ember_train_features.csv   # EMBER has separate train/test files
│   └── ...
│
├── notebooks/                     # Jupyter notebooks — one folder PER dataset
│   ├── drebin/
│   │   ├── 01_EDA.ipynb           # Step 1: explore data
│   │   ├── 02_Preprocess.ipynb    # Step 2: clean & split
│   │   ├── 03_Train.ipynb         # Step 3: train RF+SHAP
│   │   └── 04_Test.ipynb          # Step 4: metrics + IEEE figures
│   ├── ember/, nsl-kdd/, mlran/, mh-1m/, cicids-2017/
│   └── benchmark/05_Benchmark_Report.ipynb
│
├── data/processed/{dataset}/      # Preprocessed .npy arrays (canonical)
├── models/{dataset}/              # Trained rf_shap_model.pkl
├── reports/
│   ├── figures/{dataset}/         # IEEE PDF/SVG/EPS figures
│   ├── benchmark/                 # Master tables + dashboard
│   └── XMalDetect_Benchmark_Report.pdf
│
├── src/                           # Python library (core, io, preprocessing, aggregate)
├── run_benchmark.py               # Full 6-dataset pipeline + PDF report
└── outputs/{dataset}/             # Legacy pipeline artifacts (still written)
```

**Golden rule:** Each dataset has isolated artifacts. Cross-dataset comparison lives in `reports/benchmark/`.

---

## Step 0 — Install Python & dependencies

### 0.1 Open a terminal in the project folder

```powershell
cd "D:\University Bullshitz\9th Semester\Machine Learning\Paper\xmaldetect"
```

### 0.2 Create a virtual environment (recommended)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 0.3 Verify installation

```powershell
python -c "import sklearn, shap, pandas; print('OK')"
```

---

## Step 1 — Understand your datasets

These **8 datasets** are already processed in `data/`:

| Folder key | CSV files in `data/` | Domain | Train/test |
|------------|----------------------|--------|------------|
| **drebin** | `drebin_features.csv`, `drebin_labels.csv` | Android permissions | Single file → auto 80/20 split |
| **ember** | `ember_train/test_features.csv`, `ember_train/test_labels.csv` | Windows PE files | Pre-split |
| **nsl-kdd** | `nslkdd_train/test_features.csv`, `nslkdd_train/test_labels.csv` | Network intrusion | Official split |
| **mlran** | `mlran_train_feature.csv`*, `mlran_test_features.csv`, labels | Ransomware behavior | Pre-split |
| **mh-1m** | `mh1m_train_feature.csv`*, `mh1m_test_feature.csv`, labels | Android (large) | Pre-split |
| **dmbd** | `dmbd_features.csv`, `dmbd_labels.csv` | Dynamic behavior | Single file → split |
| **cicids-2017** | `cicids2017_features.csv`, `cicids2017_labels.csv` | Network flows | Single file (large) |
| **bodmas** | `bodmas_features.csv`, `bodmas_labels.csv` | PE multi-family | Single file (large) |

\*Some files use `feature` (singular) instead of `features` — the loader handles both automatically.

### Labels meaning (all datasets)

| Value | Meaning |
|-------|---------|
| `0` | Benign (safe / normal traffic) |
| `1` | Malware (or attack) |

---

## Step 2 — Choose how to run

### Option A — Jupyter notebooks (best for learning)

```powershell
jupyter notebook
```

Open `notebooks/drebin/01_EDA.ipynb` and run cells top-to-bottom. Then 02, 03, 04, 05 **in order**.

Repeat for any other dataset folder (e.g. `notebooks/ember/`).

### Option B — Command line (fastest)

```powershell
# List datasets
python -m src.pipeline.runner --list

# Run full pipeline for DREBIN
python -m src.pipeline.runner drebin --step all

# Run only one step
python -m src.pipeline.runner ember --step preprocess
python -m src.pipeline.runner ember --step model
python -m src.pipeline.runner ember --step shap
python -m src.pipeline.runner ember --step forensic
```

For **very large** datasets (CICIDS, BODMAS, EMBER), cap rows while testing:

```powershell
python -m src.pipeline.runner cicids-2017 --step preprocess --max-rows 50000
```

---

## Step 3 — What each notebook / step produces

### 01 — EDA (Exploratory Data Analysis)

**Input:** Raw CSV in `data/`  
**Output:** `outputs/{dataset}/figures/eda_class_distribution.png`

You learn: sample count, feature count, class balance.

### 02 — Preprocessing

**Input:** Raw CSV  
**Output:** `outputs/{dataset}/processed/`

| File | What it is |
|------|------------|
| `X_train.npy` | Training features (numbers) |
| `X_test.npy` | Test features |
| `y_train.npy` | Training labels |
| `y_test.npy` | Test labels |
| `feature_names.csv` | Human-readable column names |
| `preprocessing_summary.json` | Shapes, SMOTE applied?, etc. |

### 03 — Modeling

**Input:** Processed `.npy` files  
**Output:**

| File | Paper reference |
|------|-----------------|
| `tables/comparison_table3.csv` | Table 3 — model comparison |
| `figures/fig6_model_comparison_bar.png` | Fig 6 |
| `figures/fig7_roc_curves.png` | Fig 7 |
| `figures/fig8_confusion_matrix.png` | Fig 8 |
| `models/rf_shap_model.pkl` | Saved Random Forest |

### 04 — SHAP

**Output:**

| File | Meaning |
|------|---------|
| `figures/fig3_shap_summary.png` | Global feature importance |
| `figures/fig4_waterfall_sample0.png` | Why one sample was classified |
| `tables/table4_shap_top20.csv` | Top 20 SHAP features |

### 05 — Forensic report

**Output:** HTML reports in `outputs/{dataset}/reports/`  
Maps top features to **CWE** security vulnerabilities (best for Android datasets).

---

## Step 4 — Run ALL datasets + merge into one table & figure

```powershell
# Run every dataset (smart memory caps) + build master outputs
python run_all.py

# Or merge existing results only (no retraining)
python run_all.py --aggregate-only
```

**Master outputs** land in `outputs/master/`:

| File | What it is |
|------|------------|
| `tables/master_model_comparison.csv` | Every dataset x every model x every metric |
| `tables/master_dataset_summary.csv` | One row per dataset (RF+SHAP F1/AUC, sample counts) |
| `tables/master_f1_pivot.csv` | Pivot: datasets (rows) x models (cols) |
| `tables/master_domain_summary.csv` | Average metrics by domain (android/network/pe) |
| `tables/master_shap_top20.csv` | Top SHAP features from all datasets |
| `figures/fig_master_dashboard.png` | **THE main figure** — 4-panel cross-dataset dashboard |
| `figures/fig_master_heatmap_f1.png` | Heatmap: datasets x models |
| `figures/fig_master_roc_grid.png` | ROC curves grid (one per dataset) |
| `collection_manifest.json` | Run status for every dataset |

**Jupyter:** `notebooks/master/06_Cross_Dataset_Aggregation.ipynb`

```powershell
python -m src.compare_results          # same as aggregate-only shortcut
python -m src.pipeline.runner --aggregate
```

---

## Step 5 — Identify samples in your data

### In notebooks

After loading in `01_EDA.ipynb`:

```python
# Row index 0 = first sample
print(f"Label: {y[0]}")           # 0 or 1
print(f"Features: {feature_names[:5]}")
print(f"Values: {X[0, :5]}")
```

### After modeling

```python
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# Sample index 42 in test set
i = 42
print(f"True label: {y_test[i]}")
print(f"Predicted:  {y_pred[i]}")
print(f"Confidence: {y_proba[i]:.2%}")
```

### False positives / false negatives

```python
fp = np.where((y_test == 0) & (y_pred == 1))[0]  # benign called malware
fn = np.where((y_test == 1) & (y_pred == 0))[0]  # malware missed
print(f"False positives: {len(fp)}")
print(f"False negatives: {len(fn)}")
```

---

## Step 6 — Regenerate notebooks

If you change the generator:

```powershell
python generate_notebooks.py              # all datasets
python generate_notebooks.py --dataset drebin
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError` for CSV | Check `data/` — run `python data/download_datasets.py --process <name>` |
| Out of memory | Use `--max-rows 50000` or subsample in notebook 02 |
| `ModuleNotFoundError: shap` | `pip install -r requirements.txt` |
| Old results mixed up | Delete `outputs/{dataset}/` and re-run |
| SVM very slow | Normal on large datasets — use `--max-rows` for testing |

---

## Quick reference — one command per dataset

```powershell
python -m src.pipeline.runner drebin --step all
python -m src.pipeline.runner ember --step all
python -m src.pipeline.runner nsl-kdd --step all
python -m src.pipeline.runner mlran --step all
python -m src.pipeline.runner mh-1m --step all
python -m src.pipeline.runner dmbd --step all
python -m src.pipeline.runner cicids-2017 --step all --max-rows 100000
python -m src.pipeline.runner bodmas --step all --max-rows 50000
```

---

## Next steps for your paper

1. Run **drebin** fully (primary Android dataset)
2. Run **nsl-kdd** or **cicids-2017** for network IDS comparison
3. Run **ember** or **bodmas** for PE malware comparison
4. Use `cross_dataset_comparison.csv` for multi-dataset discussion
5. Copy figures from `outputs/{dataset}/figures/` into your paper
