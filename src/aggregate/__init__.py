"""Cross-dataset aggregation: unified tables and master figures."""

from .collect import collect_all_results, DatasetRunRecord
from .tables import build_master_tables
from .figures import build_master_figures
from .report import aggregate_all

__all__ = [
    "collect_all_results",
    "DatasetRunRecord",
    "build_master_tables",
    "build_master_figures",
    "aggregate_all",
]
