import io
import os
from typing import BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import (
    FEATURE_NAMES_23,
    FileQualityReport,
    ExecutionMetadata,
    LabelMapper,
    TimestampInfo,
    MEMORY_FILES,
    PAGE_TYPE_2MB,
    PAGE_TYPE_4KB,
    PAGE_TYPE_MMIO,
    PipelineRejectionError,
    PreprocessResponse,
    REQUIRED_FILES,
    STORAGE_FILES,
    VALID_MEMORY_TYPES,
)


def parse_and_validate_csv(
    filename: str,
    raw_bytes: bytes,
) -> Tuple[pd.DataFrame, FileQualityReport]:
    """Validate one RanSMAP CSV and return its standardized DataFrame and report."""
    errors: List[str] = []
    warnings: List[str] = []
    is_storage = filename in STORAGE_FILES
    expected_cols = 5 if is_storage else 6

    if not raw_bytes or len(raw_bytes.strip()) == 0:
        warnings.append(f"{filename} is empty (0 events logged).")
        empty_cols = ["ts", "tns", "addr", "size", "entropy"]
        if not is_storage:
            empty_cols.append("type")

        df_empty = pd.DataFrame(columns=empty_cols)
        return df_empty, FileQualityReport(
            filename=filename,
            is_valid=True,
            row_count=0,
            column_count=expected_cols,
            is_empty=True,
            non_monotonic_timestamps=0,
            duplicate_rows=0,
            invalid_entropy_rows=0,
            unexpected_sentinel_rows=0,
            unexpected_page_type_rows=0,
            negative_value_rows=0,
            warnings=warnings,
            errors=[],
        )

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), header=None)
    except Exception as exc:
        errors.append(f"CSV parse failure: {exc}")
        return pd.DataFrame(), FileQualityReport(
            filename=filename,
            is_valid=False,
            row_count=0,
            column_count=0,
            is_empty=True,
            non_monotonic_timestamps=0,
            duplicate_rows=0,
            invalid_entropy_rows=0,
            unexpected_sentinel_rows=0,
            unexpected_page_type_rows=0,
            negative_value_rows=0,
            errors=errors,
            warnings=warnings,
        )

    actual_cols = df.shape[1]

    # Allow a trailing sixth column in storage CSV exports.
    if is_storage and actual_cols == 6:
        warnings.append("Storage CSV has 6 columns; slicing first 5 per Table 3.")
        df = df.iloc[:, :5]
    elif actual_cols != expected_cols:
        errors.append(
            f"Schema violation: Expected {expected_cols} columns for "
            f"{filename}, got {actual_cols}."
        )
        return pd.DataFrame(), FileQualityReport(
            filename=filename,
            is_valid=False,
            row_count=len(df),
            column_count=actual_cols,
            is_empty=False,
            non_monotonic_timestamps=0,
            duplicate_rows=0,
            invalid_entropy_rows=0,
            unexpected_sentinel_rows=0,
            unexpected_page_type_rows=0,
            negative_value_rows=0,
            errors=errors,
            warnings=warnings,
        )

    col_names = (
        ["ts", "tns", "addr", "size", "entropy"]
        if is_storage
        else ["ts", "tns", "addr", "size", "entropy", "type"]
    )
    df.columns = col_names

    # Convert each column to numeric and reject corrupted or missing values.
    for column in col_names:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    nan_rows = int(df.isna().any(axis=1).sum())
    if nan_rows:
        errors.append(
            f"Data corruption: Found {nan_rows} rows with non-numeric or NaN values."
        )
        return df, FileQualityReport(
            filename=filename,
            is_valid=False,
            row_count=len(df),
            column_count=actual_cols,
            is_empty=False,
            non_monotonic_timestamps=0,
            duplicate_rows=0,
            invalid_entropy_rows=0,
            unexpected_sentinel_rows=0,
            unexpected_page_type_rows=0,
            negative_value_rows=0,
            errors=errors,
            warnings=warnings,
        )

    # Compute timestamps as integer nanoseconds.
    df["time_ns"] = (
        df["ts"].astype(np.int64) * 1_000_000_000
        + df["tns"].astype(np.int64)
    )

    # Check negative values and invalid nanosecond components.
    neg_mask = (
        (df["ts"] < 0)
        | (df["tns"] < 0)
        | (df["tns"] >= 1_000_000_000)
        | (df["addr"] < 0)
        | (df["size"] < 0)
    )
    negative_value_rows = int(neg_mask.sum())
    if negative_value_rows:
        errors.append(
            f"{negative_value_rows} rows have negative values or invalid tns >= 1e9."
        )

    # Check timestamp monotonicity.
    diffs = np.diff(df["time_ns"].to_numpy())
    non_monotonic = int(np.sum(diffs < 0))
    if non_monotonic:
        warnings.append(
            f"{non_monotonic} out-of-order timestamp transitions detected."
        )

    # Check duplicate records.
    duplicate_rows = int(df.duplicated(subset=col_names).sum())
    if duplicate_rows:
        warnings.append(f"{duplicate_rows} exact duplicate rows detected.")

    # Check entropy and sentinel rules.
    invalid_entropy_rows = 0
    unexpected_sentinel_rows = 0

    if filename in ("ata_read.csv", "mem_read.csv", "mem_exec.csv"):
        bad_sentinel = df["entropy"] != -1
        bad_ent_count = int(bad_sentinel.sum())
        if bad_ent_count:
            invalid_entropy_rows += bad_ent_count
            errors.append(
                f"{bad_ent_count} rows have invalid sentinel entropy "
                f"(expected -1 for {filename})."
            )

        if filename in ("mem_read.csv", "mem_exec.csv"):
            bad_size = int((df["size"] != 0).sum())
            if bad_size:
                unexpected_sentinel_rows += bad_size
                warnings.append(
                    f"{bad_size} rows have non-zero size for {filename} "
                    "(Table 3 specifies size=0)."
                )
    else:
        bad_entropy = (df["entropy"] < 0.0) | (df["entropy"] > 1.0)
        invalid_entropy_rows = int(bad_entropy.sum())
        if invalid_entropy_rows:
            errors.append(
                f"{invalid_entropy_rows} rows have entropy outside "
                "valid range [0.0, 1.0]."
            )

        if filename in ("mem_write.csv", "mem_readwrite.csv"):
            over_4k = int((df["size"] > 4096).sum())
            if over_4k:
                warnings.append(f"{over_4k} rows have size > 4096 bytes.")

    # Check memory access type enum.
    unexpected_page_type_rows = 0
    if not is_storage:
        bad_types = ~df["type"].isin(VALID_MEMORY_TYPES)
        unexpected_page_type_rows = int(bad_types.sum())
        if unexpected_page_type_rows:
            errors.append(
                f"{unexpected_page_type_rows} rows have invalid memory page type "
                "(must be 1, 2, 3, or 4)."
            )

    min_ts = float(df["time_ns"].min() / 1e9) if len(df) else None
    max_ts = float(df["time_ns"].max() / 1e9) if len(df) else None

    report = FileQualityReport(
        filename=filename,
        is_valid=len(errors) == 0,
        row_count=len(df),
        column_count=actual_cols,
        is_empty=len(df) == 0,
        non_monotonic_timestamps=non_monotonic,
        duplicate_rows=duplicate_rows,
        invalid_entropy_rows=invalid_entropy_rows,
        unexpected_sentinel_rows=unexpected_sentinel_rows,
        unexpected_page_type_rows=unexpected_page_type_rows,
        negative_value_rows=negative_value_rows,
        min_timestamp_s=min_ts,
        max_timestamp_s=max_ts,
        errors=errors,
        warnings=warnings,
    )
    return df, report


class TimestampInfo(Tuple[int, int]):
    storage_t0_ns: int
    memory_t0_ns: int
    global_t0_ns: int

    def __new__(cls, storage_t0_ns: int, memory_t0_ns: int):
        instance = super().__new__(cls, (storage_t0_ns, memory_t0_ns))
        instance.storage_t0_ns = storage_t0_ns
        instance.memory_t0_ns = memory_t0_ns
        instance.global_t0_ns = min(storage_t0_ns, memory_t0_ns)
        return instance

    def __repr__(self) -> str:
        return (
            "TimestampInfo("
            f"storage_t0_ns={self.storage_t0_ns}, "
            f"memory_t0_ns={self.memory_t0_ns}, "
            f"global_t0_ns={self.global_t0_ns})"
        )


def extract_23_features(
    dfs: Dict[str, pd.DataFrame],
    t_d_sec: float = 30.0,
    t_window_sec: float = 0.1,
    step_epsilon_sec: float = None,
    exclude_mmio_from_entropy: bool = False,
) -> Tuple[np.ndarray, TimestampInfo, float]:
    """Extract the 23 RanSMAP features from the six input event streams."""
    if step_epsilon_sec is None:
        step_epsilon_sec = (
            1.0 if np.isclose(t_window_sec, 10.0) else t_window_sec
        )

    storage_times = [
        int(df["time_ns"].min())
        for filename, df in dfs.items()
        if filename in STORAGE_FILES and not df.empty and "time_ns" in df.columns
    ]
    memory_times = [
        int(df["time_ns"].min())
        for filename, df in dfs.items()
        if filename in MEMORY_FILES and not df.empty and "time_ns" in df.columns
    ]

    if not storage_times and not memory_times:
        raise ValueError("All 6 CSV files are empty; cannot determine execution start time.")

    t0_storage_ns = min(storage_times or memory_times)
    t0_memory_ns = min(memory_times or storage_times)
    t0_info = TimestampInfo(t0_storage_ns, t0_memory_ns)

    rel_data = {}
    for filename, df in dfs.items():
        base_t0 = t0_storage_ns if filename in STORAGE_FILES else t0_memory_ns

        if df.empty:
            rel_data[filename] = {
                "rel_ns": np.empty(0, dtype=np.int64),
                "addr": np.empty(0, dtype=np.int64),
                "size": np.empty(0, dtype=np.float64),
                "entropy": np.empty(0, dtype=np.float64),
                "type": np.empty(0, dtype=np.int64),
            }
            continue

        rel_ns = df["time_ns"].to_numpy(dtype=np.int64) - base_t0
        order = np.argsort(rel_ns, kind="mergesort")
        rel_data[filename] = {
            "rel_ns": rel_ns[order],
            "addr": df["addr"].to_numpy(dtype=np.int64)[order],
            "size": df["size"].to_numpy(dtype=np.float64)[order],
            "entropy": df["entropy"].to_numpy(dtype=np.float64)[order],
            "type": (
                df["type"].to_numpy(dtype=np.int64)[order]
                if "type" in df.columns
                else np.empty(0, dtype=np.int64)
            ),
        }

    t_d_ns = int(round(t_d_sec * 1e9))
    t_win_ns = int(round(t_window_sec * 1e9))
    eps_ns = int(round(step_epsilon_sec * 1e9))

    def slice_idx(rel_ns_arr: np.ndarray, start_ns: int, end_ns: int) -> slice:
        left = int(np.searchsorted(rel_ns_arr, start_ns, side="left"))
        right = int(np.searchsorted(rel_ns_arr, end_ns, side="left"))
        return slice(left, right)

    def calc_throughput(sizes: np.ndarray) -> float:
        if sizes.size == 0:
            return 0.0
        return float(np.sum(sizes) / t_window_sec)

    def calc_sample_variance(addrs: np.ndarray) -> float:
        # Center integer addresses before float conversion to preserve precision.
        if addrs.size < 2:
            return 0.0
        centered = addrs - addrs.min()
        return float(np.var(centered.astype(np.float64), ddof=1))

    def calc_mean_entropy(
        entropies: np.ndarray,
        types: np.ndarray = None,
    ) -> float:
        if entropies.size == 0:
            return 0.0
        if (
            exclude_mmio_from_entropy
            and types is not None
            and types.size == entropies.size
        ):
            valid = entropies[types != PAGE_TYPE_MMIO]
            return float(np.mean(valid)) if valid.size else 0.0
        return float(np.mean(entropies))

    windows_list = []
    t_ns = 0

    while t_ns <= (t_d_ns - t_win_ns):
        end_ns = t_ns + t_win_ns

        sr = rel_data.get(
            "ata_read.csv",
            {"rel_ns": np.empty(0), "size": np.empty(0), "addr": np.empty(0)},
        )
        sw = rel_data.get(
            "ata_write.csv",
            {
                "rel_ns": np.empty(0),
                "size": np.empty(0),
                "addr": np.empty(0),
                "entropy": np.empty(0),
            },
        )

        idx_sr = slice_idx(sr["rel_ns"], t_ns, end_ns)
        idx_sw = slice_idx(sw["rel_ns"], t_ns, end_ns)

        T_s_r = calc_throughput(sr["size"][idx_sr])
        T_s_w = calc_throughput(sw["size"][idx_sw])
        V_s_r = calc_sample_variance(sr["addr"][idx_sr])
        V_s_w = calc_sample_variance(sw["addr"][idx_sw])
        H_s_w = calc_mean_entropy(sw["entropy"][idx_sw])

        mr = rel_data.get(
            "mem_read.csv",
            {"rel_ns": np.empty(0), "addr": np.empty(0), "type": np.empty(0)},
        )
        mw = rel_data.get(
            "mem_write.csv",
            {
                "rel_ns": np.empty(0),
                "addr": np.empty(0),
                "entropy": np.empty(0),
                "type": np.empty(0),
            },
        )
        mrw = rel_data.get(
            "mem_readwrite.csv",
            {
                "rel_ns": np.empty(0),
                "addr": np.empty(0),
                "entropy": np.empty(0),
                "type": np.empty(0),
            },
        )
        mx = rel_data.get(
            "mem_exec.csv",
            {"rel_ns": np.empty(0), "addr": np.empty(0), "type": np.empty(0)},
        )

        idx_mr = slice_idx(mr["rel_ns"], t_ns, end_ns)
        idx_mw = slice_idx(mw["rel_ns"], t_ns, end_ns)
        idx_mrw = slice_idx(mrw["rel_ns"], t_ns, end_ns)
        idx_mx = slice_idx(mx["rel_ns"], t_ns, end_ns)

        H_m_w = calc_mean_entropy(
            mw["entropy"][idx_mw],
            mw["type"][idx_mw],
        )
        H_m_rw = calc_mean_entropy(
            mrw["entropy"][idx_mrw],
            mrw["type"][idx_mrw],
        )

        types_r = mr["type"][idx_mr]
        types_w = mw["type"][idx_mw]
        types_rw = mrw["type"][idx_mrw]
        types_x = mx["type"][idx_mx]

        C_4KB_r = float(np.sum(types_r == PAGE_TYPE_4KB))
        C_4KB_w = float(np.sum(types_w == PAGE_TYPE_4KB))
        C_4KB_rw = float(np.sum(types_rw == PAGE_TYPE_4KB))
        C_4KB_x = float(np.sum(types_x == PAGE_TYPE_4KB))

        C_2MB_r = float(np.sum(types_r == PAGE_TYPE_2MB))
        C_2MB_w = float(np.sum(types_w == PAGE_TYPE_2MB))
        C_2MB_rw = float(np.sum(types_rw == PAGE_TYPE_2MB))
        C_2MB_x = float(np.sum(types_x == PAGE_TYPE_2MB))

        C_MMIO_r = float(np.sum(types_r == PAGE_TYPE_MMIO))
        C_MMIO_w = float(np.sum(types_w == PAGE_TYPE_MMIO))
        C_MMIO_rw = float(np.sum(types_rw == PAGE_TYPE_MMIO))
        C_MMIO_x = float(np.sum(types_x == PAGE_TYPE_MMIO))

        V_m_r = calc_sample_variance(mr["addr"][idx_mr])
        V_m_w = calc_sample_variance(mw["addr"][idx_mw])
        V_m_rw = calc_sample_variance(mrw["addr"][idx_mrw])
        V_m_x = calc_sample_variance(mx["addr"][idx_mx])

        x_t = [
            T_s_r, T_s_w, V_s_r, V_s_w, H_s_w,
            H_m_w, H_m_rw,
            C_4KB_r, C_4KB_w, C_4KB_rw, C_4KB_x,
            C_2MB_r, C_2MB_w, C_2MB_rw, C_2MB_x,
            C_MMIO_r, C_MMIO_w, C_MMIO_rw, C_MMIO_x,
            V_m_r, V_m_w, V_m_rw, V_m_x,
        ]
        windows_list.append(x_t)
        t_ns += eps_ns

    return np.asarray(windows_list, dtype=np.float64), t0_info, step_epsilon_sec


def _read_input_bytes(
    raw_input: Union[bytes, str, os.PathLike, BinaryIO],
) -> bytes:
    """Convert supported input formats into raw bytes."""
    if isinstance(raw_input, (bytes, bytearray)):
        return bytes(raw_input)
    if isinstance(raw_input, (str, os.PathLike)):
        with open(raw_input, "rb") as file_obj:
            return file_obj.read()
    if hasattr(raw_input, "read"):
        content = raw_input.read()
        return content.encode("utf-8") if isinstance(content, str) else content
    raise TypeError(f"Unsupported file input type: {type(raw_input).__name__}")


def validate_dataset(
    file_inputs: Dict[str, Union[bytes, str, os.PathLike, BinaryIO]],
) -> Tuple[bool, List[str], Dict[str, FileQualityReport]]:
    """Validate that all required CSV streams are present and well-formed."""
    quality_reports: Dict[str, FileQualityReport] = {}
    rejection_reasons: List[str] = []

    missing = [
        filename
        for filename in REQUIRED_FILES
        if filename not in file_inputs or file_inputs[filename] is None
    ]
    for filename in missing:
        message = f"Missing required CSV: '{filename}'"
        rejection_reasons.append(message)
        quality_reports[filename] = FileQualityReport(
            filename=filename,
            is_valid=False,
            row_count=0,
            column_count=0,
            is_empty=True,
            non_monotonic_timestamps=0,
            duplicate_rows=0,
            invalid_entropy_rows=0,
            unexpected_sentinel_rows=0,
            unexpected_page_type_rows=0,
            negative_value_rows=0,
            errors=[message],
        )

    total_rows = 0
    for filename in REQUIRED_FILES:
        if filename in missing:
            continue

        try:
            raw_bytes = _read_input_bytes(file_inputs[filename])
        except Exception as exc:
            rejection_reasons.append(f"Failed to read '{filename}': {exc}")
            quality_reports[filename] = FileQualityReport(
                filename=filename,
                is_valid=False,
                row_count=0,
                column_count=0,
                is_empty=True,
                non_monotonic_timestamps=0,
                duplicate_rows=0,
                invalid_entropy_rows=0,
                unexpected_sentinel_rows=0,
                negative_value_rows=0,
                unexpected_page_type_rows=0,
                errors=[str(exc)],
            )
            continue

        df, report = parse_and_validate_csv(filename, raw_bytes)
        quality_reports[filename] = report
        total_rows += len(df)
        if not report.is_valid:
            rejection_reasons.append(
                f"{filename} failed validation: {'; '.join(report.errors)}"
            )

    if not missing and total_rows == 0:
        rejection_reasons.append(
            "All 6 CSV files are empty (0 total logged events)."
        )

    return len(rejection_reasons) == 0, rejection_reasons, quality_reports


def discover_files_in_dir(
    dir_path: Union[str, os.PathLike],
) -> Dict[str, str]:
    """Find the required RanSMAP CSV files in a directory."""
    dir_path = os.path.abspath(dir_path)
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Directory not found: {dir_path}")

    return {
        filename: os.path.join(dir_path, filename)
        for filename in REQUIRED_FILES
        if os.path.isfile(os.path.join(dir_path, filename))
    }


def process_files(
    file_inputs: Dict[str, Union[bytes, str, os.PathLike, BinaryIO]],
    t_d_sec: float = 30.0,
    t_window_sec: float = 0.1,
    step_epsilon_sec: Optional[float] = None,
    exclude_mmio_from_entropy: bool = False,
    sample_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    label: Optional[str] = "benign",
    label_mapper: Optional[LabelMapper] = None,
    raise_on_rejection: bool = True,
) -> PreprocessResponse:
    """Validate and process all six RanSMAP event streams."""
    mapper = label_mapper or LabelMapper()
    label_id = mapper.map_label(label) if label is not None else None

    dfs: Dict[str, pd.DataFrame] = {}
    quality_reports: Dict[str, FileQualityReport] = {}
    rejection_reasons: List[str] = []

    missing_files = [
        filename
        for filename in REQUIRED_FILES
        if filename not in file_inputs or file_inputs[filename] is None
    ]
    for filename in missing_files:
        message = f"Required file '{filename}' was not provided."
        rejection_reasons.append(f"Missing required CSV: '{filename}'")
        quality_reports[filename] = FileQualityReport(
            filename=filename,
            is_valid=False,
            row_count=0,
            column_count=0,
            is_empty=True,
            non_monotonic_timestamps=0,
            duplicate_rows=0,
            invalid_entropy_rows=0,
            unexpected_sentinel_rows=0,
            negative_value_rows=0,
            unexpected_page_type_rows=0,
            errors=[message],
        )

    for filename in REQUIRED_FILES:
        if filename in missing_files:
            continue

        try:
            raw_bytes = _read_input_bytes(file_inputs[filename])
        except Exception as exc:
            rejection_reasons.append(f"Failed to read '{filename}': {exc}")
            quality_reports[filename] = FileQualityReport(
                filename=filename,
                is_valid=False,
                row_count=0,
                column_count=0,
                is_empty=True,
                non_monotonic_timestamps=0,
                duplicate_rows=0,
                invalid_entropy_rows=0,
                unexpected_sentinel_rows=0,
                negative_value_rows=0,
                unexpected_page_type_rows=0,
                errors=[f"Read failure: {exc}"],
            )
            continue

        df, report = parse_and_validate_csv(filename, raw_bytes)
        quality_reports[filename] = report
        dfs[filename] = df

        if not report.is_valid:
            rejection_reasons.append(
                f"{filename} failed validation: {'; '.join(report.errors)}"
            )

    total_rows = sum(len(df) for df in dfs.values())
    if len(dfs) == len(REQUIRED_FILES) and total_rows == 0:
        rejection_reasons.append(
            "All 6 CSV files are empty (0 total logged events)."
        )

    if rejection_reasons:
        if raise_on_rejection:
            raise PipelineRejectionError(
                message=f"Pipeline rejected CSVs: {'; '.join(rejection_reasons)}",
                rejection_reasons=rejection_reasons,
                quality_reports=quality_reports,
            )

        return PreprocessResponse(
            is_valid=False,
            rejected=True,
            rejection_reasons=rejection_reasons,
            execution_metadata=None,
            t_d_sec=t_d_sec,
            t_window_sec=t_window_sec,
            step_epsilon_sec=step_epsilon_sec or t_window_sec,
            global_t0_ns=0,
            num_windows=0,
            num_features=len(FEATURE_NAMES_23),
            feature_names=FEATURE_NAMES_23,
            quality_reports=quality_reports,
            features=[],
        )

    features_matrix, timestamp_info, actual_epsilon = extract_23_features(
        dfs=dfs,
        t_d_sec=t_d_sec,
        t_window_sec=t_window_sec,
        step_epsilon_sec=step_epsilon_sec,
        exclude_mmio_from_entropy=exclude_mmio_from_entropy,
    )

    t0_storage_ns = timestamp_info.storage_t0_ns
    t0_memory_ns = timestamp_info.memory_t0_ns
    global_t0_ns = min(t0_storage_ns, t0_memory_ns)

    event_counts = {filename: len(dfs[filename]) for filename in REQUIRED_FILES}
    metadata = ExecutionMetadata(
        sample_id=sample_id,
        experiment_id=experiment_id,
        label=label,
        label_id=label_id,
        duration_sec=t_d_sec,
        window_size_sec=t_window_sec,
        step_size_sec=actual_epsilon,
        global_t0_ns=global_t0_ns,
        t0_storage_ns=t0_storage_ns,
        t0_memory_ns=t0_memory_ns,
        total_events=total_rows,
        event_counts_per_file=event_counts,
    )

    return PreprocessResponse(
        is_valid=True,
        rejected=False,
        rejection_reasons=[],
        execution_metadata=metadata,
        t_d_sec=t_d_sec,
        t_window_sec=t_window_sec,
        step_epsilon_sec=actual_epsilon,
        global_t0_ns=global_t0_ns,
        t0_storage_ns=t0_storage_ns,
        t0_memory_ns=t0_memory_ns,
        num_windows=int(features_matrix.shape[0]),
        num_features=int(features_matrix.shape[1]),
        feature_names=FEATURE_NAMES_23,
        quality_reports=quality_reports,
        features=features_matrix.tolist(),
    )


# Alias for backward compatibility.
process_and_extract_features = process_files


def process_directory(
    dir_path: Union[str, os.PathLike],
    t_d_sec: float = 30.0,
    t_window_sec: float = 0.1,
    step_epsilon_sec: Optional[float] = None,
    exclude_mmio_from_entropy: bool = False,
    sample_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    label: Optional[str] = "benign",
    label_mapper: Optional[LabelMapper] = None,
    raise_on_rejection: bool = True,
) -> PreprocessResponse:
    """Process all six RanSMAP CSV files from a directory."""
    discovered = discover_files_in_dir(dir_path)
    file_inputs: Dict[str, Union[bytes, str, os.PathLike, BinaryIO]] = {
        filename: discovered.get(filename)
        for filename in REQUIRED_FILES
    }
    return process_files(
        file_inputs=file_inputs,
        t_d_sec=t_d_sec,
        t_window_sec=t_window_sec,
        step_epsilon_sec=step_epsilon_sec,
        exclude_mmio_from_entropy=exclude_mmio_from_entropy,
        sample_id=sample_id,
        experiment_id=experiment_id,
        label=label,
        label_mapper=label_mapper,
        raise_on_rejection=raise_on_rejection,
    )