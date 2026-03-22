"""
Data Preprocessing Pipeline for UAV Forensic Anomaly Detection.

Handles:
- Kaggle Tampering Dataset v2 (time-series, labeled by case_id)
- Kaggle Supplemental Operations Log (real flight reference data)
- DJI DatCon CSV parsing
- NIST .TXT telemetry parsing
- Auto-detection of file format
- Two-Stage Normalization (Mean Instance Normalization + Min-Max)
- Sliding window segmentation
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)


# ============================================================================
# KAGGLE DATASET 1: TAMPERING DATASET (PRIMARY TRAINING DATA)
# ============================================================================

def load_tampering_dataset(
    csv_path: str = None,
    severity: str = None,
    max_cases: int = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the Kaggle Drone Telemetry Tampering Dataset v2.

    This is time-series data grouped by case_id with row_idx ordering.
    Each row has a label: 0 (normal) or 1 (tampered).

    Args:
        csv_path: Path to the main CSV file. If None, uses config default.
        severity: Load from a specific severity folder ('balanced', 'strong', 'subtle').
                  If None, loads the main pack CSV.
        max_cases: Limit number of case_ids to load (for memory/testing).

    Returns:
        Tuple of (DataFrame with telemetry columns, Series of labels).
    """
    if csv_path is None:
        csv_path = config.TAMPER_CSV

    # If severity specified, look for CSVs in that subfolder
    if severity and severity in config.TAMPER_SEVERITY_LEVELS:
        severity_dir = os.path.join(config.TAMPER_DATASET_DIR, severity)
        if os.path.isdir(severity_dir):
            csv_files = glob.glob(os.path.join(severity_dir, "**", "*.csv"), recursive=True)
            if csv_files:
                logger.info(f"Loading {len(csv_files)} CSVs from {severity}/ folder")
                dfs = [pd.read_csv(f) for f in csv_files]
                df = pd.concat(dfs, ignore_index=True)
                logger.info(f"  Loaded {len(df)} rows from {severity}/ severity")
                labels = df[config.TAMPER_LABEL_COL] if config.TAMPER_LABEL_COL in df.columns else None
                return df, labels

    # Load main pack CSV
    if not os.path.exists(csv_path):
        logger.error(f"Tampering dataset not found at: {csv_path}")
        logger.info(f"  Download from: kaggle.com/datasets/rasikaekanayakadevlk/"
                     f"drone-telemetry-tampering-dataset-v2")
        logger.info(f"  Place files in: {config.TAMPER_DATASET_DIR}")
        raise FileNotFoundError(f"Tampering dataset CSV not found: {csv_path}")

    logger.info(f"Loading tampering dataset: {csv_path}")

    # Read CSV — it may be large (1.48 GB), load in chunks if needed
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except MemoryError:
        logger.warning("CSV too large for memory, loading in chunks...")
        chunks = pd.read_csv(csv_path, chunksize=500_000, low_memory=False)
        df = pd.concat(chunks, ignore_index=True)

    logger.info(f"  Loaded {len(df)} rows, {df.shape[1]} columns")
    logger.info(f"  Columns: {list(df.columns)}")

    # Limit cases if requested
    if max_cases and config.TAMPER_CASE_COL in df.columns:
        unique_cases = df[config.TAMPER_CASE_COL].unique()[:max_cases]
        df = df[df[config.TAMPER_CASE_COL].isin(unique_cases)]
        logger.info(f"  Limited to {max_cases} cases → {len(df)} rows")

    # Extract labels
    labels = None
    if config.TAMPER_LABEL_COL in df.columns:
        labels = df[config.TAMPER_LABEL_COL].copy()
        n_normal = (labels == 0).sum()
        n_tampered = (labels == 1).sum()
        logger.info(f"  Labels: {n_normal} normal, {n_tampered} tampered")

    return df, labels


def preprocess_tampering_dataset(
    csv_path: str = None,
    severity: str = None,
    max_cases: int = None,
    window_size: int = config.WINDOW_SIZE,
    stride: int = config.STRIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for the tampering dataset.

    Steps:
        1. Load CSV
        2. Identify feature columns (auto-detect lat/lon/alt/speed/motion)
        3. Group by case_id
        4. Create sliding windows per case
        5. Assign per-window labels (majority vote within window)

    Args:
        csv_path: Path to tamper CSV.
        severity: Optional severity level folder.
        max_cases: Max cases to load.
        window_size: Sliding window size.
        stride: Sliding window stride.

    Returns:
        Tuple of (windows, labels) arrays.
    """
    df, raw_labels = load_tampering_dataset(csv_path, severity, max_cases)

    # Auto-detect which columns exist for features
    feature_cols = _detect_tamper_feature_columns(df)
    logger.info(f"  Using features: {feature_cols}")

    all_windows = []
    all_labels = []

    # Group by case_id if available
    if config.TAMPER_CASE_COL in df.columns:
        groups = df.groupby(config.TAMPER_CASE_COL)
        logger.info(f"  Processing {len(groups)} flight cases...")

        for case_id, case_df in groups:
            # Sort by row_idx
            if config.TAMPER_ROW_COL in case_df.columns:
                case_df = case_df.sort_values(config.TAMPER_ROW_COL)

            # Extract features
            features = case_df[feature_cols].values.astype(np.float32)
            features = np.nan_to_num(features, nan=0.0)

            # Get labels for this case
            if config.TAMPER_LABEL_COL in case_df.columns:
                case_labels = case_df[config.TAMPER_LABEL_COL].values
            else:
                case_labels = np.zeros(len(case_df), dtype=np.int32)

            # Skip very short cases
            if len(features) < 10:
                continue

            # Create sliding windows
            windows = create_sliding_windows(features, window_size, stride)

            # Per-window label: majority vote of rows in each window
            window_labels = []
            for start in range(0, len(case_labels) - window_size + 1, stride):
                window_lab = case_labels[start:start + window_size]
                # If ANY row in the window is tampered → label=1
                label = int(np.any(window_lab == 1))
                window_labels.append(label)

            # Handle padding edge case
            while len(window_labels) < len(windows):
                window_labels.append(int(np.any(case_labels == 1)))

            all_windows.append(windows)
            all_labels.append(np.array(window_labels[:len(windows)], dtype=np.int32))
    else:
        # No case_id — treat entire file as one sequence
        features = df[feature_cols].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)

        windows = create_sliding_windows(features, window_size, stride)

        if raw_labels is not None:
            labels_arr = raw_labels.values
            window_labels = []
            for start in range(0, len(labels_arr) - window_size + 1, stride):
                label = int(np.any(labels_arr[start:start + window_size] == 1))
                window_labels.append(label)
            while len(window_labels) < len(windows):
                window_labels.append(0)
            all_labels.append(np.array(window_labels[:len(windows)], dtype=np.int32))
        else:
            all_labels.append(np.zeros(len(windows), dtype=np.int32))

        all_windows.append(windows)

    if not all_windows:
        return np.empty((0, window_size, len(feature_cols))), np.empty((0,), dtype=np.int32)

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)

    logger.info(f"  Total windows: {len(X)} (normal={np.sum(y==0)}, tampered={np.sum(y==1)})")
    return X, y


def _detect_tamper_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect feature columns from the tampering dataset.

    Tries to match columns to our standard features:
    latitude, longitude, altitude, speed/velocity, motor_rpm.

    Falls back to all numeric columns excluding metadata (case_id, row_idx, label).

    Args:
        df: Tampering DataFrame.

    Returns:
        List of column names to use as features.
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}
    metadata_cols = {config.TAMPER_CASE_COL, config.TAMPER_ROW_COL, config.TAMPER_LABEL_COL}

    # Try exact match to our standard feature names first
    matched = []
    for feature in config.FEATURE_COLUMNS:
        if feature in cols_lower:
            matched.append(cols_lower[feature])
        else:
            # Fuzzy match
            for cl, orig in cols_lower.items():
                if feature[:3] in cl and orig not in matched and orig not in metadata_cols:
                    matched.append(orig)
                    break

    # If we got at least 3, use them
    if len(matched) >= 3:
        return matched

    # Fallback: use all numeric columns except metadata
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in metadata_cols]

    if not feature_cols:
        raise ValueError(f"No usable numeric feature columns found. Columns: {list(df.columns)}")

    # Limit to reasonable number
    if len(feature_cols) > 20:
        feature_cols = feature_cols[:20]
        logger.warning(f"  Truncated to 20 feature columns")

    return feature_cols


# ============================================================================
# KAGGLE DATASET 2: SUPPLEMENTAL OPERATIONS LOG (REFERENCE DATA)
# ============================================================================

def load_reference_dataset(csv_path: str = None) -> pd.DataFrame:
    """
    Load the Kaggle Supplemental Drone Telemetry Data & Operations Log.

    This is the KNOWN-GOOD reference data (1 row per real flight, 22 columns).
    Used to validate that the system correctly recognizes legitimate flights.

    Args:
        csv_path: Path to the operations log CSV.

    Returns:
        DataFrame with all 22 columns.
    """
    if csv_path is None:
        csv_path = config.REFERENCE_CSV

    if not os.path.exists(csv_path):
        logger.error(f"Reference dataset not found at: {csv_path}")
        logger.info(f"  Download from: kaggle.com/datasets/samsudeenashad/"
                     f"supplemental-drone-telemetry-data-and-operations-log")
        logger.info(f"  Place file in: {config.REFERENCE_DATASET_DIR}")
        raise FileNotFoundError(f"Reference dataset CSV not found: {csv_path}")

    logger.info(f"Loading reference dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  Loaded {len(df)} flights, {df.shape[1]} columns")
    logger.info(f"  Columns: {list(df.columns)}")

    # Standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    return df


def build_reference_profile(df: pd.DataFrame = None) -> Dict:
    """
    Build a statistical profile from the supplemental (real/known-good) data.

    This profile defines what "normal" real-world flights look like:
    - Valid altitude ranges
    - Typical flight durations
    - Expected battery levels
    - GPS accuracy norms
    - Wind speed ranges

    When you later provide new evidence data, the system checks if its
    properties fall within these known-good ranges.

    Args:
        df: Reference DataFrame. If None, loads from config path.

    Returns:
        Dictionary with reference statistics per numeric feature.
    """
    if df is None:
        df = load_reference_dataset()

    profile = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}

    for feature in config.REF_NUMERIC_FEATURES:
        # Find matching column
        match = None
        for cl, orig in cols_lower.items():
            if feature.replace("_", "") in cl.replace("_", "").replace(" ", ""):
                match = orig
                break
            if feature[:4] in cl:
                match = orig
                break

        if match and match in df.columns:
            values = pd.to_numeric(df[match], errors="coerce").dropna()
            if len(values) > 0:
                profile[feature] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "count": int(len(values)),
                    # Acceptable range: mean ± 3σ (or min/max, whichever wider)
                    "accept_min": float(max(values.min(), values.mean() - 3 * values.std())),
                    "accept_max": float(min(values.max(), values.mean() + 3 * values.std())),
                }
                logger.info(f"  {feature}: μ={profile[feature]['mean']:.2f}, "
                           f"σ={profile[feature]['std']:.2f}, "
                           f"range=[{profile[feature]['min']:.2f}, {profile[feature]['max']:.2f}]")
        else:
            logger.warning(f"  Reference column for '{feature}' not found")

    # Capture categorical distributions
    for cat_col in ["application", "flight_status", "manufacturer"]:
        for cl, orig in cols_lower.items():
            if cat_col in cl:
                profile[f"{cat_col}_distribution"] = df[orig].value_counts().to_dict()
                break

    logger.info(f"  Reference profile built with {len(profile)} features")
    return profile


def validate_against_reference(
    evidence_data: Dict,
    reference_profile: Dict,
) -> Dict:
    """
    Validate new evidence data against the known-good reference profile.

    Checks if the flight parameters of new evidence fall within the
    acceptable ranges derived from real supplemental data.

    Args:
        evidence_data: Dict of feature_name → value(s) from new evidence.
        reference_profile: Profile from build_reference_profile().

    Returns:
        Validation report: {feature: {value, in_range, deviation, flag}}.
    """
    report = {"checks": [], "passed": 0, "failed": 0, "warnings": 0}

    for feature, value in evidence_data.items():
        if feature not in reference_profile:
            continue

        ref = reference_profile[feature]
        if not isinstance(value, (int, float)):
            continue

        in_range = ref["accept_min"] <= value <= ref["accept_max"]
        deviation = abs(value - ref["mean"]) / (ref["std"] + 1e-8)

        check = {
            "feature": feature,
            "value": value,
            "ref_mean": ref["mean"],
            "ref_std": ref["std"],
            "ref_range": [ref["accept_min"], ref["accept_max"]],
            "in_range": in_range,
            "deviation_sigma": round(deviation, 2),
            "flag": "PASS" if in_range else ("WARNING" if deviation < 4 else "FAIL"),
        }
        report["checks"].append(check)

        if check["flag"] == "PASS":
            report["passed"] += 1
        elif check["flag"] == "WARNING":
            report["warnings"] += 1
        else:
            report["failed"] += 1

    total = report["passed"] + report["failed"] + report["warnings"]
    report["total_checks"] = total
    report["integrity_score"] = report["passed"] / max(total, 1)
    report["verdict"] = "LEGITIMATE" if report["failed"] == 0 else "SUSPICIOUS"

    return report


# ============================================================================
# LEGACY FILE PARSERS (DJI, NIST, Generic CSVs)
# ============================================================================

def detect_file_format(filepath: str) -> str:
    """
    Auto-detect file format: tampering dataset, DJI DatCon, NIST, or generic.

    Args:
        filepath: Path to telemetry file.

    Returns:
        'tamper', 'dji', 'nist', or 'generic'.
    """
    try:
        df_sample = pd.read_csv(filepath, nrows=5, sep=None, engine="python")
        columns_lower = [c.lower().strip() for c in df_sample.columns]

        # Check for tampering dataset format (has case_id + label)
        if any("case_id" in c for c in columns_lower) and any("label" in c for c in columns_lower):
            return "tamper"
        # Check for DJI-specific column patterns
        if any("gps:lat" in c or "gps:long" in c for c in columns_lower):
            return "dji"
        # Check for NIST-specific patterns
        if any("lat" == c or "lon" == c or "groundspeed" in c for c in columns_lower):
            return "nist"
        # Fallback: check for generic
        if any("latitude" in c for c in columns_lower):
            return "generic"
        # If has drone_id + altitude → supplemental operations log
        if any("drone_id" in c or "drone id" in c for c in columns_lower):
            return "reference"
        return "generic"
    except Exception as e:
        logger.error(f"Format detection failed for {filepath}: {e}")
        return "generic"


def parse_dji_csv(filepath: str) -> pd.DataFrame:
    """Parse a DJI DatCon CSV into standardized DataFrame."""
    logger.info(f"Parsing DJI CSV: {filepath}")
    df = pd.read_csv(filepath)
    result = pd.DataFrame()

    for dji_col, std_col in config.DJI_COLUMN_MAP.items():
        matching = [c for c in df.columns if dji_col.lower() in c.lower()]
        if matching:
            result[std_col] = df[matching[0]].astype(float)
        else:
            result[std_col] = np.nan

    # Compute speed from velocity components if needed
    vel_n_cols = [c for c in df.columns if "veln" in c.lower()]
    vel_e_cols = [c for c in df.columns if "vele" in c.lower()]
    if vel_n_cols and vel_e_cols and "speed" in result.columns:
        if result["speed"].isna().all():
            result["speed"] = np.sqrt(
                df[vel_n_cols[0]].astype(float)**2 + df[vel_e_cols[0]].astype(float)**2
            )

    motor_cols = [c for c in df.columns if "motor" in c.lower() and "speed" in c.lower()]
    if motor_cols and len(motor_cols) > 1:
        result["motor_rpm"] = df[motor_cols].astype(float).mean(axis=1)

    return result.ffill().bfill()


def parse_nist_txt(filepath: str) -> pd.DataFrame:
    """
    Parse a NIST drone forensic .TXT telemetry file.

    Args:
        filepath: Path to the NIST telemetry file.

    Returns:
        DataFrame with columns: [latitude, longitude, altitude, speed, motor_rpm]
    """
    logger.info(f"Parsing NIST TXT: {filepath}")
    df = pd.read_csv(filepath, sep=None, engine="python")

    result = pd.DataFrame()

    for nist_col, std_col in config.NIST_COLUMN_MAP.items():
        matching = [c for c in df.columns if nist_col.lower() == c.lower().strip()]
        if matching:
            result[std_col] = pd.to_numeric(df[matching[0]], errors="coerce")
        else:
            logger.warning(f"NIST column '{nist_col}' not found, filling with NaN")
            result[std_col] = np.nan

    result = result.ffill().bfill()
    return result


def parse_generic_csv(filepath: str) -> pd.DataFrame:
    """
    Parse a generic CSV with standard column names.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with standardized feature columns.
    """
    logger.info(f"Parsing generic CSV: {filepath}")
    df = pd.read_csv(filepath, sep=None, engine="python")

    result = pd.DataFrame()
    columns_lower = {c.lower().strip(): c for c in df.columns}

    for feature in config.FEATURE_COLUMNS:
        if feature in columns_lower:
            result[feature] = pd.to_numeric(df[columns_lower[feature]], errors="coerce")
        else:
            # Try fuzzy matching
            matched = [c for c in columns_lower if feature[:3] in c]
            if matched:
                result[feature] = pd.to_numeric(df[columns_lower[matched[0]]], errors="coerce")
            else:
                logger.warning(f"Column '{feature}' not found, filling with NaN")
                result[feature] = np.nan

    result = result.ffill().bfill()
    return result


def load_telemetry(filepath: str) -> pd.DataFrame:
    """
    Load a telemetry file with automatic format detection.

    Supports: Tampering dataset, DJI DatCon, NIST, generic CSV.

    Args:
        filepath: Path to telemetry file (.csv or .txt).

    Returns:
        Standardized DataFrame.
    """
    fmt = detect_file_format(filepath)
    if fmt == "tamper":
        # For tamper format, use the dedicated loader
        df = pd.read_csv(filepath, low_memory=False)
        return df
    elif fmt == "reference":
        return load_reference_dataset(filepath)
    elif fmt == "dji":
        return parse_dji_csv(filepath)
    elif fmt == "nist":
        return parse_nist_txt(filepath)
    else:
        return parse_generic_csv(filepath)


# ============================================================================
# NORMALIZATION
# ============================================================================

def mean_instance_normalize(data: np.ndarray) -> np.ndarray:
    """
    Stage 1: Mean Instance Normalization (MIN).
    Removes per-instance bias by normalizing each flight segment independently.

    For each instance x with feature f:
        x_norm(f) = (x(f) - mean_instance(f)) / (std_instance(f) + eps)

    Args:
        data: Array of shape (num_samples, seq_len, num_features)
              or (seq_len, num_features) for a single instance.

    Returns:
        Instance-normalized data of the same shape.
    """
    eps = 1e-8

    if data.ndim == 2:
        # Single instance: (seq_len, features)
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        return (data - mean) / (std + eps)
    elif data.ndim == 3:
        # Batch: (num_samples, seq_len, features)
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        return (data - mean) / (std + eps)
    else:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")


def min_max_scale(
    data: np.ndarray,
    feature_min: Optional[np.ndarray] = None,
    feature_max: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage 2: Global Min-Max Scaling to [0, 1].

    Args:
        data: Array of shape (num_samples, seq_len, num_features).
        feature_min: Pre-computed min per feature (for test-time scaling).
        feature_max: Pre-computed max per feature (for test-time scaling).

    Returns:
        Tuple of (scaled_data, feature_min, feature_max).
    """
    if data.ndim == 3:
        # Compute global min/max across all samples and timesteps
        if feature_min is None:
            feature_min = data.min(axis=(0, 1))
        if feature_max is None:
            feature_max = data.max(axis=(0, 1))
    elif data.ndim == 2:
        if feature_min is None:
            feature_min = data.min(axis=0)
        if feature_max is None:
            feature_max = data.max(axis=0)

    denom = feature_max - feature_min
    denom[denom == 0] = 1.0  # Prevent division by zero

    scaled = (data - feature_min) / denom
    return scaled, feature_min, feature_max


def two_stage_normalize(
    data: np.ndarray,
    feature_min: Optional[np.ndarray] = None,
    feature_max: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Two-Stage Normalization:
        Stage 1: Mean Instance Normalization
        Stage 2: Global Min-Max Scaling

    Args:
        data: Array of shape (num_samples, seq_len, num_features).
        feature_min: Optional pre-computed min (for test set).
        feature_max: Optional pre-computed max (for test set).

    Returns:
        Tuple of (normalized_data, feature_min, feature_max).
    """
    # Stage 1: Instance normalization
    data_min = mean_instance_normalize(data)
    # Stage 2: Global min-max scaling
    data_scaled, f_min, f_max = min_max_scale(data_min, feature_min, feature_max)
    return data_scaled, f_min, f_max


# ============================================================================
# SLIDING WINDOW SEGMENTATION
# ============================================================================

def create_sliding_windows(
    data: np.ndarray,
    window_size: int = config.WINDOW_SIZE,
    stride: int = config.STRIDE,
) -> np.ndarray:
    """
    Segment continuous telemetry into overlapping sliding windows.

    Args:
        data: 2D array of shape (total_timesteps, num_features).
        window_size: Number of timesteps per window.
        stride: Step size between consecutive windows.

    Returns:
        3D array of shape (num_windows, window_size, num_features).
    """
    if len(data) < window_size:
        # Pad short sequences with zeros
        padding = np.zeros((window_size - len(data), data.shape[1]))
        data = np.vstack([data, padding])
        logger.warning(f"Sequence padded from {len(data) - len(padding)} to {window_size}")

    windows = []
    for start in range(0, len(data) - window_size + 1, stride):
        window = data[start : start + window_size]
        windows.append(window)

    if not windows:
        # Edge case: return the padded full sequence
        windows.append(data[:window_size])

    return np.array(windows, dtype=np.float32)


# ============================================================================
# FULL PREPROCESSING PIPELINE
# ============================================================================

def preprocess_file(
    filepath: str,
    label: int,
    window_size: int = config.WINDOW_SIZE,
    stride: int = config.STRIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for a single telemetry file.

    Steps:
        1. Parse telemetry file (auto-detect format)
        2. Extract feature columns
        3. Create sliding windows
        4. Return windows and labels

    Args:
        filepath: Path to telemetry file.
        label: 0 (Normal) or 1 (Tampered).
        window_size: Sliding window size.
        stride: Sliding window stride.

    Returns:
        Tuple of (windows, labels) where:
            windows: shape (num_windows, window_size, num_features)
            labels: shape (num_windows,)
    """
    # Parse file
    df = load_telemetry(filepath)

    # Extract features
    features = df[config.FEATURE_COLUMNS].values.astype(np.float32)

    # Replace any remaining NaNs with 0
    features = np.nan_to_num(features, nan=0.0)

    # Create sliding windows
    windows = create_sliding_windows(features, window_size, stride)
    labels = np.full(len(windows), label, dtype=np.int32)

    logger.info(f"  → {len(windows)} windows from {os.path.basename(filepath)}")
    return windows, labels


def preprocess_directory(
    normal_dir: str,
    tampered_dir: Optional[str] = None,
    window_size: int = config.WINDOW_SIZE,
    stride: int = config.STRIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess all telemetry files in directories.

    Args:
        normal_dir: Directory containing normal flight logs.
        tampered_dir: Directory containing tampered/anomalous logs (optional).
        window_size: Sliding window size.
        stride: Sliding window stride.

    Returns:
        Tuple of (all_windows, all_labels).
    """
    all_windows = []
    all_labels = []

    # Process normal files
    if os.path.isdir(normal_dir):
        for ext in ["*.csv", "*.txt"]:
            for fp in glob.glob(os.path.join(normal_dir, ext)):
                try:
                    w, l = preprocess_file(fp, config.LABEL_NORMAL, window_size, stride)
                    all_windows.append(w)
                    all_labels.append(l)
                except Exception as e:
                    logger.error(f"Failed to process {fp}: {e}")

    # Process tampered files
    if tampered_dir and os.path.isdir(tampered_dir):
        for ext in ["*.csv", "*.txt"]:
            for fp in glob.glob(os.path.join(tampered_dir, ext)):
                try:
                    w, l = preprocess_file(fp, config.LABEL_TAMPERED, window_size, stride)
                    all_windows.append(w)
                    all_labels.append(l)
                except Exception as e:
                    logger.error(f"Failed to process {fp}: {e}")

    if not all_windows:
        logger.warning("No data files found. Returning empty arrays.")
        return np.empty((0, window_size, config.NUM_FEATURES)), np.empty((0,))

    windows = np.concatenate(all_windows, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    logger.info(f"Total: {len(windows)} windows ({np.sum(labels == 0)} normal, {np.sum(labels == 1)} tampered)")
    return windows, labels


def generate_synthetic_data(
    num_normal: int = 500,
    num_tampered: int = 100,
    seq_len: int = config.WINDOW_SIZE,
    num_features: int = config.NUM_FEATURES,
    seed: int = config.RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic flight telemetry data for testing and development.

    Normal flights: smooth sinusoidal patterns with small Gaussian noise.
    Tampered flights: normal patterns + injected anomalies (GPS spoof, altitude jump).

    Args:
        num_normal: Number of normal flight windows.
        num_tampered: Number of tampered flight windows.
        seq_len: Sequence length per window.
        num_features: Number of features.
        seed: Random seed.

    Returns:
        Tuple of (windows, labels).
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, seq_len)

    windows = []
    labels = []

    # Generate normal flights
    for _ in range(num_normal):
        flight = np.zeros((seq_len, num_features), dtype=np.float32)
        flight[:, 0] = 37.0 + 0.001 * np.sin(t) + rng.normal(0, 0.0001, seq_len)  # lat
        flight[:, 1] = -122.0 + 0.001 * np.cos(t) + rng.normal(0, 0.0001, seq_len)  # lon
        flight[:, 2] = 50.0 + 10 * np.sin(t * 0.5) + rng.normal(0, 0.5, seq_len)  # alt
        flight[:, 3] = 5.0 + 2 * np.abs(np.sin(t)) + rng.normal(0, 0.2, seq_len)  # speed
        flight[:, 4] = 5000 + 500 * np.sin(t * 2) + rng.normal(0, 50, seq_len)  # rpm
        windows.append(flight)
        labels.append(config.LABEL_NORMAL)

    # Generate tampered flights (with injected anomalies)
    for _ in range(num_tampered):
        flight = np.zeros((seq_len, num_features), dtype=np.float32)
        flight[:, 0] = 37.0 + 0.001 * np.sin(t) + rng.normal(0, 0.0001, seq_len)
        flight[:, 1] = -122.0 + 0.001 * np.cos(t) + rng.normal(0, 0.0001, seq_len)
        flight[:, 2] = 50.0 + 10 * np.sin(t * 0.5) + rng.normal(0, 0.5, seq_len)
        flight[:, 3] = 5.0 + 2 * np.abs(np.sin(t)) + rng.normal(0, 0.2, seq_len)
        flight[:, 4] = 5000 + 500 * np.sin(t * 2) + rng.normal(0, 50, seq_len)

        # Inject anomalies
        anomaly_type = rng.choice(["gps_spoof", "altitude_jump", "rpm_drop", "combined"])
        inject_start = rng.randint(20, 60)
        inject_end = min(inject_start + rng.randint(10, 30), seq_len)

        if anomaly_type == "gps_spoof":
            flight[inject_start:inject_end, 0] += rng.uniform(0.01, 0.05)  # lat shift
            flight[inject_start:inject_end, 1] += rng.uniform(0.01, 0.05)  # lon shift
        elif anomaly_type == "altitude_jump":
            flight[inject_start:inject_end, 2] += rng.uniform(50, 200)  # sudden altitude
        elif anomaly_type == "rpm_drop":
            flight[inject_start:inject_end, 4] *= rng.uniform(0.1, 0.3)  # motor failure
        elif anomaly_type == "combined":
            flight[inject_start:inject_end, 0] += rng.uniform(0.005, 0.02)
            flight[inject_start:inject_end, 2] += rng.uniform(20, 80)
            flight[inject_start:inject_end, 4] *= rng.uniform(0.2, 0.5)

        windows.append(flight)
        labels.append(config.LABEL_TAMPERED)

    windows = np.array(windows, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # Shuffle
    indices = rng.permutation(len(windows))
    return windows[indices], labels[indices]
