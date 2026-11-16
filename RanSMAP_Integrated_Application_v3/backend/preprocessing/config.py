from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

STORAGE_FILES = ("ata_read.csv", "ata_write.csv")
MEMORY_FILES = ("mem_read.csv", "mem_write.csv", "mem_readwrite.csv", "mem_exec.csv")
REQUIRED_FILES = STORAGE_FILES + MEMORY_FILES

# Memory access page types (Section 2.3)
PAGE_TYPE_4KB = 1
PAGE_TYPE_2MB = 2
PAGE_TYPE_1GB = 3
PAGE_TYPE_MMIO = 4
VALID_MEMORY_TYPES = {PAGE_TYPE_4KB, PAGE_TYPE_2MB, PAGE_TYPE_1GB, PAGE_TYPE_MMIO}

# Exact 23-feature ordering from Appendix D (Figs. D.1 and D.2)
FEATURE_NAMES_23 = [
    # Storage features x_s (5 features: Fig. D.1 line 13)
    "T_s_r",        # 0: Average read throughput [byte/s]
    "T_s_w",        # 1: Average write throughput [byte/s]
    "V_s_r",        # 2: Variance of read LBAs
    "V_s_w",        # 3: Variance of write LBAs
    "H_s_w",        # 4: Average Shannon entropy of written sectors
    # Memory features x_m (18 features: Fig. D.2 lines 17-18)
    "H_m_w",        # 5: Average Shannon entropy of mem_write
    "H_m_rw",       # 6: Average Shannon entropy of mem_readwrite
    "C_4KB_r",      # 7: 4KiB page access count (read)
    "C_4KB_w",      # 8: 4KiB page access count (write)
    "C_4KB_rw",     # 9: 4KiB page access count (readwrite)
    "C_4KB_x",      # 10: 4KiB page access count (exec)
    "C_2MB_r",      # 11: 2MiB page access count (read)
    "C_2MB_w",      # 12: 2MiB page access count (write)
    "C_2MB_rw",     # 13: 2MiB page access count (readwrite)
    "C_2MB_x",      # 14: 2MiB page access count (exec)
    "C_MMIO_r",     # 15: MMIO access count (read)
    "C_MMIO_w",     # 16: MMIO access count (write)
    "C_MMIO_rw",    # 17: MMIO access count (readwrite)
    "C_MMIO_x",     # 18: MMIO access count (exec)
    "V_m_r",        # 19: Variance of GPA (read)
    "V_m_w",        # 20: Variance of GPA (write)
    "V_m_rw",       # 21: Variance of GPA (readwrite)
    "V_m_x",        # 22: Variance of GPA (exec)
]

# Canonical technical metadata for all 23 features per RanSMAP specification
FEATURE_METADATA_23: Dict[str, Dict[str, str]] = {
    "T_s_r": {
        "index": 0,
        "name": "T_s_r",
        "desc": "Average read throughput",
        "unit": "byte/s",
        "category": "Storage (x_s)",
        "paper_ref": "Fig. D.1 line 6",
    },
    "T_s_w": {
        "index": 1,
        "name": "T_s_w",
        "desc": "Average write throughput",
        "unit": "byte/s",
        "category": "Storage (x_s)",
        "paper_ref": "Fig. D.1 line 7",
    },
    "V_s_r": {
        "index": 2,
        "name": "V_s_r",
        "desc": "Variance of read LBAs",
        "unit": "LBA^2",
        "category": "Storage (x_s)",
        "paper_ref": "Fig. D.1 line 8",
    },
    "V_s_w": {
        "index": 3,
        "name": "V_s_w",
        "desc": "Variance of write LBAs",
        "unit": "LBA^2",
        "category": "Storage (x_s)",
        "paper_ref": "Fig. D.1 line 9",
    },
    "H_s_w": {
        "index": 4,
        "name": "H_s_w",
        "desc": "Average Shannon entropy of written sectors",
        "unit": "[0, 1]",
        "category": "Storage (x_s)",
        "paper_ref": "Fig. D.1 line 10",
    },
    "H_m_w": {
        "index": 5,
        "name": "H_m_w",
        "desc": "Average Shannon entropy of mem_write",
        "unit": "[0, 1]",
        "category": "Memory (x_m)",
        "paper_ref": "Fig. D.2 line 8",
    },
    "H_m_rw": {
        "index": 6,
        "name": "H_m_rw",
        "desc": "Average Shannon entropy of mem_readwrite",
        "unit": "[0, 1]",
        "category": "Memory (x_m)",
        "paper_ref": "Fig. D.2 line 9",
    },
    "C_4KB_r": {
        "index": 7,
        "name": "C_4KB_r",
        "desc": "4KiB page access count (read)",
        "unit": "count",
        "category": "Memory (x_m)",
        "paper_ref": "Fig. D.2 line 10",
    },
    "C_4KB_w": {
        "index": 8,
        "name": "C_4KB_w",
        "desc": "4KiB page access count (write)",
        "unit": "count",
        "category": "Memory (x_m)",
        "paper_ref": "Fig. D.2 line 10",
    },
    "C_4KB_rw": {
        "index": 9,
        "name": "C_4KB_rw",
        "desc": "4KiB page access count (readwrite)",
        "unit": "count",
        "category": "Memory (x_m)",
        "paper_ref": "Fig. D.2 line 10",
    },
    "C_4KB_x": {
        "index": 10,
        "name": "C_4KB_x",
        "desc": "4KiB page access count (exec)",
        "unit": "count",
        "category": "Memory (x_m)",
        "paper_ref": "Fig. D.2 line 10",
    },
    "C_2MB_r": {
        "index": 11,
        "name": "C_2MB_r",
        "desc": "2MiB page access count (read)",
        "unit": "count",
        "category": "Memory (x_m)",
        "paper_ref": "Fig. D.2 line 11",
    },
    "C_2MB_w": {
        "index": 12,
        "name": "C_2MB_w",
        "desc": "2MiB page access count (write)",
        "unit": "count",
        "category": "Memory (x_m)",
        "paper_ref": "Fig. D.2 line 11",
    },
    "C_2MB_rw": {
        "index": 13,
        "name": "C_2MB_rw",
        "desc": "2MiB page access count (readwrite)",
        "unit": "count",
        "category": "Memory (x_m)",
        "paper_ref": "Fig. D.2 line 11",
    },
    "C_2MB_x": {
        "index": 14,
        "name": "C_2MB_x",
        "desc": "2MiB page access count (exec)",
        "unit": "count",
        "category": "Memory (x_m)",
        "paper_ref": "Fig. D.2 line 11",
    },
    "C_MMIO_r": {
        "index": 15,
        "name": "C_MMIO_r",
        "desc": "MMIO access count (read)",
        "unit": "count",
        "category": "Memory (x_m)",
        "paper_ref": "Fig. D.2 line 12",
    },
    "C_MMIO_w": {
        "index": 16,
        "name": "C_MMIO_w",
        "desc": "MMIO access count (write)",
        "unit": "count",
        "category": "Memory (x_m)",
        "paper_ref": "Fig. D.2 line 12",
    },
    "C_MMIO_rw": {
        "index": 17,
        "name": "C_MMIO_rw",
        "desc": "MMIO access count (readwrite)",
        "unit": "count",
        "category": "Memory (x_m)",
        "paper_ref": "Fig. D.2 line 12",
    },
    "C_MMIO_x": {
        "index": 18,
        "name": "C_MMIO_x",
        "desc": "MMIO access count (exec)",
        "unit": "count",
        "category": "Memory (x_m)",
        "paper_ref": "Fig. D.2 line 12",
    },
    "V_m_r": {
        "index": 19,
        "name": "V_m_r",
        "desc": "Variance of GPA (read)",
        "unit": "GPA^2",
        "category": "Memory (x_m)",
        "paper_ref": "Fig. D.2 line 13",
    },
    "V_m_w": {
        "index": 20,
        "name": "V_m_w",
        "desc": "Variance of GPA (write)",
        "unit": "GPA^2",
        "category": "Memory (x_m)",
        "paper_ref": "Fig. D.2 line 14",
    },
    "V_m_rw": {
        "index": 21,
        "name": "V_m_rw",
        "desc": "Variance of GPA (readwrite)",
        "unit": "GPA^2",
        "category": "Memory (x_m)",
        "paper_ref": "Fig. D.2 line 15",
    },
    "V_m_x": {
        "index": 22,
        "name": "V_m_x",
        "desc": "Variance of GPA (exec)",
        "unit": "GPA^2",
        "category": "Memory (x_m)",
        "paper_ref": "Fig. D.2 line 16",
    },
}

# Default label mappings for ML classification
DEFAULT_BINARY_LABELS: Dict[str, int] = {
    "benign": 0,
    "goodware": 0,
    "clean": 0,
    "ransomware": 1,
    "malware": 1,
}

DEFAULT_FAMILY_LABELS: Dict[str, int] = {
    "benign": 0,
    "locky": 1,
    "wannacry": 2,
    "cerber": 3,
    "ryuk": 4,
    "maze": 5,
    "gandcrab": 6,
    "revil": 7,
    "sodinokibi": 7,
    "conti": 8,
    "blackcat": 9,
    "generic_ransomware": 10,
}


class LabelMapper:
    """
    Manages string-to-integer and reverse label mappings for benign vs ransomware
    or specific malware families.
    """

    def __init__(self, mapping: Optional[Dict[str, int]] = None):
        self._mapping: Dict[str, int] = dict(mapping or DEFAULT_BINARY_LABELS)
        self._reverse: Dict[int, str] = {v: k for k, v in self._mapping.items()}

    @property
    def mapping(self) -> Dict[str, int]:
        return dict(self._mapping)

    def map_label(self, label: Union[str, int]) -> int:
        """Converts a label name or integer to standardized integer class id."""
        if isinstance(label, int):
            return label
        normalized = str(label).strip().lower()
        if normalized in self._mapping:
            return self._mapping[normalized]
        # Auto-register new label with next available integer id
        next_id = max(self._mapping.values(), default=-1) + 1
        self._mapping[normalized] = next_id
        self._reverse[next_id] = normalized
        return next_id

    def get_name(self, label_id: int) -> str:
        """Returns the canonical name for a class id."""
        return self._reverse.get(label_id, f"class_{label_id}")

    def register(self, label: str, label_id: int) -> None:
        normalized = label.strip().lower()
        self._mapping[normalized] = label_id
        self._reverse[label_id] = normalized

@dataclass
class TimestampInfo:
    storage_t0_ns: Optional[int] = None
    memory_t0_ns: Optional[int] = None
    
class ExecutionMetadata(BaseModel):
    sample_id: Optional[str] = Field(default=None, description="Hash or identifier of executed sample")
    experiment_id: Optional[str] = Field(default=None, description="Experiment or batch run ID")
    label: Optional[str] = Field(default="benign", description="Ground truth label name (e.g. benign, ransomware)")
    label_id: Optional[int] = Field(default=0, description="Encoded numerical class label for ML models")
    family: Optional[str] = Field(default=None, description="Specific malware family if applicable")
    duration_sec: float = Field(default=30.0, description="Trace duration T_d in seconds")
    window_size_sec: float = Field(default=0.1, description="Sliding window size T_window in seconds")
    step_size_sec: float = Field(default=0.1, description="Sliding step epsilon in seconds")
    global_t0_ns: Optional[int] = Field(default=None, description="Earliest execution timestamp in nanoseconds")
    t0_storage_ns: Optional[int] = Field(default=None, description="Earliest storage timestamp (ata_*) in nanoseconds")
    t0_memory_ns: Optional[int] = Field(default=None, description="Earliest memory timestamp (mem_*) in nanoseconds")
    total_events: int = Field(default=0, description="Total events across all 6 CSV files")
    event_counts_per_file: Dict[str, int] = Field(default_factory=dict, description="Event count per CSV")
    environment: Optional[Dict[str, Any]] = Field(default=None, description="Sandbox / VM execution environment")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata attributes")


class FileQualityReport(BaseModel):
    filename: str
    is_valid: bool
    row_count: int
    column_count: int
    is_empty: bool
    non_monotonic_timestamps: int
    duplicate_rows: int
    invalid_entropy_rows: int
    unexpected_sentinel_rows: int
    unexpected_page_type_rows: int
    negative_value_rows: int
    min_timestamp_s: Optional[float] = None
    max_timestamp_s: Optional[float] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class PreprocessResponse(BaseModel):
    is_valid: bool
    rejected: bool = Field(default=False, description="Explicit flag indicating whether the backend rejected the CSVs")
    rejection_reasons: List[str] = Field(default_factory=list, description="List of reasons why CSVs were rejected")
    execution_metadata: Optional[ExecutionMetadata] = None
    t_d_sec: float
    t_window_sec: float
    step_epsilon_sec: float
    global_t0_ns: int
    t0_storage_ns: Optional[int] = Field(default=None, description="Earliest storage timestamp (ata_*) in nanoseconds")
    t0_memory_ns: Optional[int] = Field(default=None, description="Earliest memory timestamp (mem_*) in nanoseconds")
    num_windows: int
    num_features: int
    feature_names: List[str]
    quality_reports: Dict[str, FileQualityReport]
    features: List[List[float]]  # Shape: (num_windows, 23)

    def to_numpy(self) -> np.ndarray:
        """Returns the (num_windows, 23) numpy feature matrix."""
        if not self.features:
            return np.empty((0, len(FEATURE_NAMES_23)), dtype=np.float64)
        return np.asarray(self.features, dtype=np.float64)

    def to_dataframe(self, include_timestamps: bool = True, include_metadata: bool = True) -> pd.DataFrame:
        """Converts the extracted features into a ready-to-use pandas DataFrame."""
        if not self.features:
            cols = (["window_idx", "t_start_s", "t_end_s"] if include_timestamps else []) + FEATURE_NAMES_23
            return pd.DataFrame(columns=cols)

        arr = self.to_numpy()
        rows = []
        for i in range(self.num_windows):
            row: Dict[str, Any] = {}
            if include_timestamps:
                row["window_idx"] = i
                row["t_start_s"] = i * self.step_epsilon_sec
                row["t_end_s"] = row["t_start_s"] + self.t_window_sec
            for col_idx, col_name in enumerate(FEATURE_NAMES_23):
                row[col_name] = arr[i, col_idx]
            if include_metadata and self.execution_metadata:
                row["sample_id"] = self.execution_metadata.sample_id
                row["label"] = self.execution_metadata.label
                row["label_id"] = self.execution_metadata.label_id
            rows.append(row)
        return pd.DataFrame(rows)

    def to_csv(self, filepath: str, include_timestamps: bool = True, include_metadata: bool = True) -> None:
        """Exports the feature dataset directly to CSV on disk."""
        df = self.to_dataframe(include_timestamps=include_timestamps, include_metadata=include_metadata)
        df.to_csv(filepath, index=False)

    def get_feature_breakdown(self, window_idx: int) -> List[Dict[str, Any]]:
        """Returns structured metadata + values for each of the 23 features in a window."""
        if window_idx < 0 or window_idx >= self.num_windows:
            raise IndexError(f"Window index {window_idx} out of range (0..{self.num_windows - 1})")
        arr = self.to_numpy()
        items = []
        for idx, name in enumerate(FEATURE_NAMES_23):
            meta = FEATURE_METADATA_23.get(name, {})
            items.append({
                "index": idx,
                "name": name,
                "value": float(arr[window_idx, idx]),
                "unit": meta.get("unit", ""),
                "category": meta.get("category", ""),
                "desc": meta.get("desc", ""),
                "paper_ref": meta.get("paper_ref", ""),
            })
        return items

    def summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Computes summary statistics (min, max, mean, std) across all windows."""
        arr = self.to_numpy()
        stats = {}
        if arr.shape[0] == 0:
            return stats
        for idx, name in enumerate(FEATURE_NAMES_23):
            col_vals = arr[:, idx]
            stats[name] = {
                "min": float(np.min(col_vals)),
                "max": float(np.max(col_vals)),
                "mean": float(np.mean(col_vals)),
                "std": float(np.std(col_vals)),
            }
        return stats


class PipelineRejectionError(ValueError):
    """
    Raised when uploaded/provided CSV files fail validation checks,
    alerting the backend to reject the dataset.
    """

    def __init__(self, message: str, rejection_reasons: List[str], quality_reports: Dict[str, FileQualityReport]):
        super().__init__(message)
        self.message = message
        self.rejection_reasons = rejection_reasons
        self.quality_reports = quality_reports

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": "PipelineRejectionError",
            "message": self.message,
            "rejection_reasons": self.rejection_reasons,
            "quality_reports": {k: v.model_dump() for k, v in self.quality_reports.items()},
        }
