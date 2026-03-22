"""
Configuration for Transformer-Based UAV Forensic Anomaly Detection System.
All hyperparameters, paths, and thresholds are defined here.
"""

import os

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models_saved")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
HASH_LOG_PATH = os.path.join(LOGS_DIR, "hash_log.json")

# ── Kaggle Dataset Paths ─────────────────────────────────────────────────────
# Dataset 1: Tampering dataset (time-series, labeled) — for TRAINING
# Download from: kaggle.com/datasets/rasikaekanayakadevlk/drone-telemetry-tampering-dataset-v2
TAMPER_DATASET_DIR = os.path.join(DATA_RAW_DIR, "tampering")
TAMPER_CSV = os.path.join(TAMPER_DATASET_DIR, "tampering_research_dataset_pack.csv")

# Dataset 2: Supplemental operations log (real flights) — for REFERENCE VALIDATION
# Download from: kaggle.com/datasets/samsudeenashad/supplemental-drone-telemetry-data-and-operations-log
REFERENCE_DATASET_DIR = os.path.join(DATA_RAW_DIR, "reference")
REFERENCE_CSV = os.path.join(REFERENCE_DATASET_DIR,
                             "Supplemental Drone Telemetry Data - Drone Operations Log.csv")

# User evidence for checking (put new data here)
EVIDENCE_DIR = os.path.join(DATA_RAW_DIR, "evidence")

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
# Feature columns to extract from telemetry
FEATURE_COLUMNS = ["latitude", "longitude", "altitude", "speed", "motor_rpm"]
NUM_FEATURES = len(FEATURE_COLUMNS)

# Sliding window parameters
SAMPLING_RATE_HZ = 10              # DJI consumer drones ≈ 10 Hz
WINDOW_SIZE_SECONDS = 10           # 10-second analysis windows
WINDOW_SIZE = SAMPLING_RATE_HZ * WINDOW_SIZE_SECONDS  # 100 timesteps
STRIDE = WINDOW_SIZE // 2          # 50% overlap = stride of 50

# Labels
LABEL_NORMAL = 0
LABEL_TAMPERED = 1

# ============================================================================
# TRANSFORMER MODEL HYPERPARAMETERS
# ============================================================================
D_MODEL = 64                       # Embedding / latent dimension
NUM_HEADS = 4                      # Multi-head attention heads
NUM_ENCODER_LAYERS = 4             # Number of Transformer encoder blocks
D_FF = 128                         # Feed-forward inner dimension (2 * d_model)
DROPOUT_RATE = 0.1                 # Dropout in encoder blocks
CLASSIFIER_DROPOUT = 0.3           # Dropout before final dense layer
MAX_SEQ_LEN = WINDOW_SIZE          # Maximum sequence length (= window size)

# ── LoRA (Low-Rank Adaptation) ───────────────────────────────────────────────
USE_LORA = True                    # Enable LoRA for parameter-efficient training
LORA_RANK = 8                      # Rank of low-rank matrices (lower = fewer params)
LORA_ALPHA = 8.0                   # Scaling factor (α). Effective scale = α/r
LORA_DROPOUT = 0.05                # Dropout on LoRA path for regularization
LORA_TARGET_MODULES = [            # Which layers get LoRA adaptation
    "query", "key", "value",       # Attention projections
    "output",                      # Attention output projection
    "ffn",                         # Feed-forward network layers
]

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
LEARNING_RATE = 1e-4               # Adam optimizer learning rate
WARMUP_STEPS = 500                 # LR warmup steps for cosine schedule
BATCH_SIZE = 32                    # Training batch size
EPOCHS = 50                        # Maximum training epochs
EARLY_STOPPING_PATIENCE = 10       # Stop if val_loss doesn't improve
VALIDATION_SPLIT = 0.15            # Fraction for validation
TEST_SPLIT = 0.15                  # Fraction for testing
RANDOM_SEED = 42                   # For reproducibility

# ============================================================================
# ANOMALY DETECTION THRESHOLDS
# ============================================================================
# Dynamic threshold: threshold = mean + k * std
ANOMALY_THRESHOLD_K = 2.0          # Number of std deviations

# One-Class SVM parameters
OCSVM_KERNEL = "rbf"
OCSVM_NU = 0.05                    # Expected fraction of outliers
OCSVM_GAMMA = "scale"

# ============================================================================
# BASELINE MODEL (Random Forest)
# ============================================================================
RF_N_ESTIMATORS = 200              # Number of trees
RF_MAX_DEPTH = 20                  # Maximum tree depth
RF_RANDOM_STATE = RANDOM_SEED

# ============================================================================
# EVALUATION
# ============================================================================
INFERENCE_WARMUP_RUNS = 5          # Discard first N runs for timing
INFERENCE_BENCHMARK_RUNS = 100     # Number of timed inference runs

# ============================================================================
# DATASET 1: TAMPERING DATASET COLUMN MAPPING
# ============================================================================
# Kaggle: rasikaekanayakadevlk/drone-telemetry-tampering-dataset-v2
# Time-series telemetry: many rows per flight, grouped by case_id
# Columns: case_id, row_idx, latitude, longitude, altitude, ..., label
TAMPER_CASE_COL = "case_id"           # Flight case identifier
TAMPER_ROW_COL = "row_idx"            # Sequential record index
TAMPER_LABEL_COL = "label"            # 0=normal, 1=tampered

# Tampering severity folders inside the dataset
TAMPER_SEVERITY_LEVELS = ["balanced", "strong", "subtle"]

# ============================================================================
# DATASET 2: SUPPLEMENTAL OPERATIONS LOG COLUMN MAPPING
# ============================================================================
# Kaggle: samsudeenashad/supplemental-drone-telemetry-data-and-operations-log
# Operations log: ONE row per flight (22 columns)
REF_COLUMN_MAP = {
    # Identification
    "drone_id": "drone_id",
    "application": "application",
    "size": "drone_size",
    "model": "drone_model",
    "manufacturer": "manufacturer",
    # Flight specs
    "number_of_propellers": "propellers",
    "max_carry_weight": "max_carry_weight",
    "actual_carry_weight": "actual_carry_weight",
    "payload_type": "payload_type",
    "payload_description": "payload_description",
    # Telemetry (key numeric fields for validation)
    "altitude": "altitude",
    "flight_duration": "flight_duration",
    "distance_covered": "distance_covered",
    "battery_remaining": "battery_remaining",
    "gps_accuracy": "gps_accuracy",
    "wind_speed": "wind_speed",
    # Operations
    "operator": "operator",
    "date": "flight_date",
    "obstacles_encountered": "obstacles",
    "flight_status": "flight_status",
    "regulatory_approval_id": "regulatory_id",
    "notes": "notes",
}

# Numeric columns from the supplemental dataset used for reference profiling
REF_NUMERIC_FEATURES = [
    "altitude", "flight_duration", "distance_covered",
    "battery_remaining", "gps_accuracy", "wind_speed",
]

# ============================================================================
# LEGACY: DJI / NIST COLUMN MAPPINGS (for additional data sources)
# ============================================================================
DJI_COLUMN_MAP = {
    "GPS:Lat": "latitude",
    "GPS:Long": "longitude",
    "GPS:heightMSL": "altitude",
    "GPS:velN": "speed",
    "Motor:RFSpeed": "motor_rpm",
}

NIST_COLUMN_MAP = {
    "lat": "latitude",
    "lon": "longitude",
    "alt": "altitude",
    "groundspeed": "speed",
    "rpm": "motor_rpm",
}

# ============================================================================
# REFERENCE VALIDATION THRESHOLDS
# ============================================================================
# Acceptable ranges derived from supplemental (real) data statistics
# These are auto-computed when reference data is loaded; defaults below
REF_ALTITUDE_RANGE = (0, 500)          # meters — will be overridden
REF_DURATION_RANGE = (0, 7200)         # seconds
REF_BATTERY_MIN = 5.0                  # minimum expected battery %
REF_GPS_ACCURACY_MAX = 50.0            # maximum acceptable GPS error
REF_WIND_SPEED_MAX = 50.0              # m/s

# ============================================================================
# ENSURE DIRECTORIES EXIST
# ============================================================================
for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODEL_SAVE_DIR, LOGS_DIR,
             TAMPER_DATASET_DIR, REFERENCE_DATASET_DIR, EVIDENCE_DIR]:
    os.makedirs(_dir, exist_ok=True)
