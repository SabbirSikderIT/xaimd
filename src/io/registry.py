"""
Dataset registry for XMalDetect.

Every processed CSV under ``data/`` is described here so notebooks, CLI tools,
and comparison scripts can load the correct files without hard-coding paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Domain = Literal["android", "network", "pe", "behavioral"]
SplitType = Literal["single_file", "pre_split", "official_test"]
MemoryProfile = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class DatasetConfig:
    """Static metadata for one malware / intrusion dataset."""

    id: str
    display_name: str
    domain: Domain
    split_type: SplitType
    description: str
    feature_type: str
    class_names: tuple[str, str] = ("Benign", "Malware")
    memory_profile: MemoryProfile = "medium"
    forensic_plugin: Literal["android", "network", "pe", "behavioral", "none"] = "none"
    scale_features: bool = False
    apply_smote: bool | str = "auto"
    drop_constant: bool = True
    label_column: str | None = None
    family_column: str | None = None
    # File names relative to data/ (supports alternate spellings via tuples)
    train_features: str | tuple[str, ...] | None = None
    train_labels: str | tuple[str, ...] | None = None
    test_features: str | tuple[str, ...] | None = None
    test_labels: str | tuple[str, ...] | None = None
    features: str | tuple[str, ...] | None = None
    labels: str | tuple[str, ...] | None = None
    max_rows: int | None = None
    pipeline_max_rows: int | None = None  # default cap for run_all (None = use all)
    cite: str = ""

    def artifact_names(self) -> list[str]:
        """Return expected processed CSV filenames."""
        names: list[str] = []
        for attr in (
            "train_features", "train_labels", "test_features", "test_labels",
            "features", "labels",
        ):
            val = getattr(self, attr)
            if val is None:
                continue
            if isinstance(val, tuple):
                names.append(val[0])
            else:
                names.append(val)
        return names


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


DATA_DIR = _project_root() / "data"


DATASETS: dict[str, DatasetConfig] = {
    "drebin": DatasetConfig(
        id="drebin",
        display_name="DREBIN — Android Malware Permissions",
        domain="android",
        split_type="single_file",
        description="129K Android APKs with 545 binary permission features.",
        feature_type="Android permissions (binary 0/1)",
        forensic_plugin="android",
        memory_profile="low",
        features="drebin_features.csv",
        labels="drebin_labels.csv",
        label_column="label",
        cite="Arp et al. (NDSS 2014)",
    ),
    "ember": DatasetConfig(
        id="ember",
        display_name="EMBER — Windows PE Malware",
        domain="pe",
        split_type="pre_split",
        description="900K train + 200K test PE files, 2381 static features.",
        feature_type="PE header / section / import features",
        forensic_plugin="pe",
        scale_features=True,
        memory_profile="high",
        train_features="ember_train_features.csv",
        train_labels="ember_train_labels.csv",
        test_features="ember_test_features.csv",
        test_labels="ember_test_labels.csv",
        label_column="label",
        pipeline_max_rows=40_000,
        cite="Anderson & Roth (2018)",
    ),
    "nsl-kdd": DatasetConfig(
        id="nsl-kdd",
        display_name="NSL-KDD — Network Intrusion Detection",
        domain="network",
        split_type="official_test",
        description="125K train + 22K test network flows (41 features, one-hot expanded).",
        feature_type="Network flow statistics",
        forensic_plugin="network",
        scale_features=True,
        memory_profile="medium",
        train_features="nslkdd_train_features.csv",
        train_labels="nslkdd_train_labels.csv",
        test_features="nslkdd_test_features.csv",
        test_labels="nslkdd_test_labels.csv",
        label_column="label",
        pipeline_max_rows=30_000,
        cite="Tavallaee et al. (2009)",
    ),
    "mlran": DatasetConfig(
        id="mlran",
        display_name="MLRan — Ransomware Behavioral Features",
        domain="behavioral",
        split_type="pre_split",
        description="Ransomware vs benign with RFE-selected behavioral features.",
        feature_type="Behavioral / API trace features (RFE)",
        forensic_plugin="behavioral",
        memory_profile="low",
        train_features=("mlran_train_features.csv", "mlran_train_feature.csv"),
        train_labels="mlran_train_labels.csv",
        test_features="mlran_test_features.csv",
        test_labels="mlran_test_labels.csv",
        label_column="sample_type",
        family_column="family_label",
        cite="MLRan (faithfulco/mlran, 2025)",
    ),
    "mh-1m": DatasetConfig(
        id="mh-1m",
        display_name="MH-1M — Large-Scale Android Hypergraph",
        domain="android",
        split_type="pre_split",
        description="Large Android malware dataset with chi-squared feature selection.",
        feature_type="Hyperedge / API-call co-occurrence features",
        forensic_plugin="android",
        memory_profile="high",
        train_features=("mh1m_train_features.csv", "mh1m_train_feature.csv"),
        train_labels="mh1m_train_labels.csv",
        test_features=("mh1m_test_features.csv", "mh1m_test_feature.csv"),
        test_labels="mh1m_test_labels.csv",
        label_column="label",
        pipeline_max_rows=20_000,
        cite="Buchko (Kaggle MH-1M, 2024)",
    ),
    "dmbd": DatasetConfig(
        id="dmbd",
        display_name="DMBD — Dynamic Malware Behavior Dataset",
        domain="behavioral",
        split_type="single_file",
        description="Dynamic behavioral features from malware execution traces.",
        feature_type="Dynamic behavioral metrics",
        forensic_plugin="behavioral",
        scale_features=True,
        memory_profile="medium",
        features="dmbd_features.csv",
        labels="dmbd_labels.csv",
        label_column="label",
        pipeline_max_rows=25_000,
        cite="LLNL DMBD",
    ),
    "cicids-2017": DatasetConfig(
        id="cicids-2017",
        display_name="CICIDS-2017 — Modern Network Intrusion",
        domain="network",
        split_type="single_file",
        description="~2.8M network flows, 78 CICFlowMeter features, binarized labels.",
        feature_type="Network flow features (CICFlowMeter)",
        forensic_plugin="network",
        scale_features=True,
        memory_profile="high",
        features="cicids2017_features.csv",
        labels="cicids2017_labels.csv",
        label_column="label",
        max_rows=500_000,
        pipeline_max_rows=50_000,
        cite="Sharafaldin et al. (2018)",
    ),
    "bodmas": DatasetConfig(
        id="bodmas",
        display_name="BODMAS — Multi-Family PE Malware",
        domain="pe",
        split_type="single_file",
        description="57K PE samples with family metadata and static features.",
        feature_type="PE static / hash-derived features",
        forensic_plugin="pe",
        scale_features=True,
        memory_profile="high",
        features="bodmas_features.csv",
        labels="bodmas_labels.csv",
        label_column="label",
        pipeline_max_rows=30_000,
        cite="BODMAS (2021)",
    ),
}


# Six validated datasets with binary labels suitable for training/benchmarking.
ACTIVE_DATASETS: tuple[str, ...] = (
    "drebin",
    "ember",
    "nsl-kdd",
    "mlran",
    "mh-1m",
    "cicids-2017",
)


def master_output_dir(root: Path | None = None) -> Path:
    """Unified cross-dataset output directory."""
    base = root or _project_root()
    return base / "reports" / "benchmark"


def get_dataset(dataset_id: str) -> DatasetConfig:
    key = dataset_id.lower().replace("_", "-")
    if key not in DATASETS:
        available = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset '{dataset_id}'. Available: {available}")
    return DATASETS[key]


def list_datasets() -> list[DatasetConfig]:
    return list(DATASETS.values())
