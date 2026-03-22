"""
╔══════════════════════════════════════════════════════════════╗
║   Transformer-Based UAV Forensic Anomaly Detection Engine   ║
║   ─────────────────────────────────────────────────────────  ║
║   Dissertation Project — March 2026                         ║
╚══════════════════════════════════════════════════════════════╝

WORKFLOW:
  1. TRAIN on Kaggle Tampering Dataset v2 (normal + tampered, labeled)
  2. REFERENCE from Kaggle Supplemental Operations Log (real/good flights)
  3. CHECK: User provides new evidence → system verifies if tampered or not

Usage:
    python main.py                          # Full pipeline with synthetic data
    python main.py --kaggle                 # Use Kaggle datasets
    python main.py --check evidence.csv     # Check a specific evidence file
    python main.py --check-dir ./evidence/  # Check all files in a directory
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import tensorflow as tf

# Project imports
import config
from src.data.preprocessing import (
    preprocess_tampering_dataset,
    preprocess_directory,
    two_stage_normalize,
    generate_synthetic_data,
    load_reference_dataset,
    build_reference_profile,
)
from src.data.dataset import split_data, build_datasets
from src.models.transformer_model import build_transformer_model
from src.models.anomaly_detector import (
    DynamicThresholdDetector,
    OneClassSVMDetector,
    EnsembleAnomalyDetector,
)
from src.training.train import train_transformer
from src.training.baseline import train_random_forest
from src.evaluation.metrics import (
    compute_all_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_training_history,
    compare_models,
    save_metrics,
)
from src.evaluation.inference import benchmark_inference
from src.utils.hash_verify import verify_all_files, check_hash_log
from src.utils.reference_validator import (
    ForensicEvidenceChecker,
    check_evidence_directory,
)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure logging for the forensic pipeline."""
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(config.LOGS_DIR, "pipeline.log"),
                mode="a",
            ),
        ],
    )


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_full_pipeline(args):
    """
    Execute the full forensic analysis pipeline.

    Steps:
    1. Verify evidence integrity (SHA-256)
    2. Load data: Kaggle tamper dataset OR synthetic
    3. Build reference profile from supplemental (real) data
    4. Normalize and split
    5. Train Transformer model
    6. Train Random Forest baseline
    7. Anomaly detection
    8. Evaluate and compare
    9. Benchmark inference speed
    10. Check user evidence (if provided)
    """
    logger = logging.getLogger("main")

    print("\n" + "═" * 60)
    print("  TRANSFORMER-BASED UAV FORENSIC ANOMALY DETECTION")
    print("  ─────────────────────────────────────────────────")
    print(f"  Mode:       {args.mode}")
    data_source = "Kaggle Datasets" if args.kaggle else (
        "Synthetic" if args.synthetic else args.data_dir
    )
    print(f"  Data:       {data_source}")
    print(f"  Seed:       {config.RANDOM_SEED}")
    print(f"  TF Version: {tf.__version__}")
    print("═" * 60)

    # Set random seeds for reproducibility
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)

    # ── STEP 1: Evidence Integrity Verification ───────────────────
    print("\n▶ STEP 1: SHA-256 Evidence Integrity Verification")
    if args.kaggle and os.path.isdir(config.TAMPER_DATASET_DIR):
        hash_results = verify_all_files(config.TAMPER_DATASET_DIR)
        logger.info(f"Verified {len(hash_results)} files")
    elif not args.synthetic and os.path.isdir(args.data_dir):
        hash_results = verify_all_files(args.data_dir)
        logger.info(f"Verified {len(hash_results)} files")
    else:
        print("  Using synthetic data — no files to verify.")

    # ── STEP 2: Data Loading & Preprocessing ──────────────────────
    print("\n▶ STEP 2: Data Loading & Preprocessing")

    if args.kaggle:
        # Use Kaggle Tampering Dataset v2
        print("  Loading Kaggle Tampering Dataset v2...")
        try:
            X_raw, y = preprocess_tampering_dataset(
                severity=args.severity,
                max_cases=args.max_cases,
            )
            print(f"  Loaded {len(X_raw)} windows "
                  f"({np.sum(y == 0)} normal, {np.sum(y == 1)} tampered)")
        except FileNotFoundError as e:
            print(f"\n  ERROR: {e}")
            print(f"\n  ┌─────────────────────────────────────────────────────────┐")
            print(f"  │  HOW TO SET UP KAGGLE DATASETS:                        │")
            print(f"  │                                                         │")
            print(f"  │  1. TAMPERED data (for training):                       │")
            print(f"  │     Download from:                                      │")
            print(f"  │     kaggle.com/datasets/rasikaekanayakadevlk/           │")
            print(f"  │       drone-telemetry-tampering-dataset-v2              │")
            print(f"  │     Extract to: data/raw/tampering/                     │")
            print(f"  │                                                         │")
            print(f"  │  2. REFERENCE data (real flights):                      │")
            print(f"  │     Download from:                                      │")
            print(f"  │     kaggle.com/datasets/samsudeenashad/                 │")
            print(f"  │       supplemental-drone-telemetry-data-and-operations  │")
            print(f"  │     Extract to: data/raw/reference/                     │")
            print(f"  │                                                         │")
            print(f"  │  3. YOUR EVIDENCE (data to check):                      │")
            print(f"  │     Place in: data/raw/evidence/                        │")
            print(f"  └─────────────────────────────────────────────────────────┘")
            print(f"\n  Falling back to synthetic data...\n")
            args.synthetic = True

    if args.synthetic and not args.kaggle or (args.kaggle and args.synthetic):
        if args.kaggle and not args.synthetic:
            pass  # Already loaded above
        else:
            print("  Generating synthetic telemetry data...")
            X_raw, y = generate_synthetic_data(
                num_normal=args.num_normal,
                num_tampered=args.num_tampered,
            )
            print(f"  Generated {len(X_raw)} samples "
                  f"({np.sum(y == 0)} normal, {np.sum(y == 1)} tampered)")

    elif not args.kaggle and not args.synthetic:
        normal_dir = os.path.join(args.data_dir, "normal")
        tampered_dir = os.path.join(args.data_dir, "tampered")

        if not os.path.isdir(normal_dir):
            normal_dir = args.data_dir
            tampered_dir = None

        X_raw, y = preprocess_directory(normal_dir, tampered_dir)

        if len(X_raw) == 0:
            print("  ERROR: No data loaded. Check your data directory.")
            sys.exit(1)

    # ── STEP 2b: Load Reference Profile ───────────────────────────
    print("\n▶ STEP 2b: Building Reference Profile (from real flight data)")
    reference_profile = {}
    try:
        if os.path.exists(config.REFERENCE_CSV):
            ref_df = load_reference_dataset()
            reference_profile = build_reference_profile(ref_df)
            print(f"  Reference profile built: {len(reference_profile)} features")

            # Save reference profile
            ref_profile_path = os.path.join(config.LOGS_DIR, "reference_profile.json")
            with open(ref_profile_path, "w") as f:
                json.dump(reference_profile, f, indent=2, default=str)
            print(f"  Saved: {ref_profile_path}")
        else:
            print(f"  Reference dataset not found at: {config.REFERENCE_CSV}")
            print(f"  Skipping reference profile — system will still work "
                  f"with Transformer-only detection.")
    except Exception as e:
        logger.warning(f"Could not build reference profile: {e}")
        print(f"  WARNING: {e}")

    # ── STEP 3: Two-Stage Normalization ───────────────────────────
    print("\n▶ STEP 3: Two-Stage Normalization")
    X_norm, feat_min, feat_max = two_stage_normalize(X_raw)
    print(f"  Shape: {X_norm.shape}")
    print(f"  Features per window: {X_norm.shape[2]}")
    print(f"  Value range: [{X_norm.min():.4f}, {X_norm.max():.4f}]")

    # Save normalization parameters for inference
    norm_params = {
        "feature_min": feat_min.tolist(),
        "feature_max": feat_max.tolist(),
    }
    norm_path = os.path.join(config.LOGS_DIR, "normalization_params.json")
    with open(norm_path, "w") as f:
        json.dump(norm_params, f, indent=2)
    print(f"  Normalization params saved: {norm_path}")

    # Update model config if feature count differs from default
    num_features = X_norm.shape[2]

    # ── STEP 4: Train/Val/Test Split ──────────────────────────────
    print("\n▶ STEP 4: Data Splitting")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X_norm, y)
    train_ds, val_ds, test_ds = build_datasets(X_norm, y)

    # ── STEP 5: Train Transformer ─────────────────────────────────
    if args.mode in ("train", "full"):
        print("\n▶ STEP 5: Transformer Training")
        if args.resume:
            print("  (Auto-resume enabled — will continue from checkpoint if available)")
        train_result = train_transformer(
            train_ds=train_ds,
            val_ds=val_ds,
            num_features=num_features,
            seq_len=X_norm.shape[1],
            epochs=args.epochs,
            resume=args.resume,
        )
        model = train_result["model"]
        history = train_result["history"]

        # Plot training curves
        plot_training_history(history)
    else:
        print("\n▶ STEP 5: Loading pre-trained model...")
        model_path = os.path.join(config.MODEL_SAVE_DIR, "transformer_best.keras")
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"  Loaded: {model_path}")
        else:
            print("  No saved model found. Building fresh model...")
            model = build_transformer_model(
                num_features=num_features,
                seq_len=X_norm.shape[1],
            )
        history = None

    # ── STEP 6: Transformer Evaluation ────────────────────────────
    print("\n▶ STEP 6: Transformer Evaluation")
    y_scores = model.predict(X_test, verbose=0).flatten()
    y_pred_transformer = (y_scores > 0.5).astype(np.int32)

    transformer_metrics = compute_all_metrics(
        y_test, y_pred_transformer, y_scores, model_name="Transformer"
    )
    save_metrics(transformer_metrics, "transformer_metrics.json")
    plot_confusion_matrix(y_test, y_pred_transformer, "Transformer")

    if y_scores is not None and len(np.unique(y_test)) > 1:
        plot_roc_curve(y_test, y_scores, "Transformer")
        plot_precision_recall_curve(y_test, y_scores, "Transformer")

    # ── STEP 7: Anomaly Detection Layer ───────────────────────────
    print("\n▶ STEP 7: Anomaly Detection (Threshold + One-Class SVM)")

    # Get embeddings for anomaly detection
    embeddings_test = model.get_embeddings(
        tf.cast(X_test, tf.float32)
    ).numpy()

    # Use normal validation samples to calibrate
    normal_val_mask = y_val == config.LABEL_NORMAL
    if np.any(normal_val_mask):
        normal_val_scores = model.predict(X_val[normal_val_mask], verbose=0).flatten()
        normal_val_embeddings = model.get_embeddings(
            tf.cast(X_val[normal_val_mask], tf.float32)
        ).numpy()

        # Ensemble detector
        ensemble = EnsembleAnomalyDetector()
        ensemble.fit(normal_val_scores, normal_val_embeddings)

        ensemble_pred, dyn_pred, ocsvm_pred = ensemble.predict(y_scores, embeddings_test)

        print("\n  Dynamic Threshold Results:")
        dyn_metrics = compute_all_metrics(y_test, dyn_pred, model_name="Dynamic Threshold")

        print("\n  One-Class SVM Results:")
        svm_metrics = compute_all_metrics(y_test, ocsvm_pred, model_name="OC-SVM")

        print("\n  Ensemble (OR) Results:")
        ens_metrics = compute_all_metrics(y_test, ensemble_pred, model_name="Ensemble")
    else:
        print("  WARNING: No normal validation samples for anomaly calibration.")

    # ── STEP 8: Random Forest Baseline ────────────────────────────
    if args.mode in ("train", "full"):
        print("\n▶ STEP 8: Random Forest Baseline")
        rf_result = train_random_forest(X_train, y_train, X_test, y_test)
        rf_metrics = rf_result["metrics"]

        # Plot RF confusion matrix
        plot_confusion_matrix(y_test, rf_result["predictions"], "Random Forest")
    else:
        rf_metrics = None
        print("\n▶ STEP 8: Skipping RF baseline (evaluate mode)")

    # ── STEP 9: Model Comparison ──────────────────────────────────
    print("\n▶ STEP 9: Model Comparison")
    comparison = {"Transformer": transformer_metrics}
    if rf_metrics:
        comparison["Random Forest"] = rf_metrics

    compare_models(comparison)

    # Save comparison
    comparison_path = os.path.join(config.LOGS_DIR, "model_comparison.json")
    with open(comparison_path, "w") as f:
        serializable = {}
        for k, v in comparison.items():
            serializable[k] = {
                mk: mv for mk, mv in v.items()
                if isinstance(mv, (int, float, str, type(None)))
            }
        json.dump(serializable, f, indent=2)

    # ── STEP 10: Inference Speed Benchmark ────────────────────────
    print("\n▶ STEP 10: Inference Speed Benchmark")
    speed_results = benchmark_inference(
        model,
        seq_len=X_norm.shape[1],
        num_features=num_features,
    )
    save_metrics(speed_results, "inference_benchmark.json")

    # ── STEP 11: Check User Evidence (if requested) ───────────────
    if args.check or args.check_dir:
        print("\n▶ STEP 11: FORENSIC EVIDENCE CHECK")
        checker = ForensicEvidenceChecker(
            model=model,
            reference_profile=reference_profile,
            norm_params=norm_params,
        )

        if args.check:
            # Check a single file
            checker.check_file(args.check)
        elif args.check_dir:
            # Check all files in a directory
            check_evidence_directory(checker, args.check_dir)
    else:
        # Always check the evidence directory if it has files
        evidence_files = []
        if os.path.isdir(config.EVIDENCE_DIR):
            evidence_files = [f for f in os.listdir(config.EVIDENCE_DIR)
                            if f.endswith((".csv", ".txt", ".dat"))]

        if evidence_files:
            print("\n▶ STEP 11: Auto-checking evidence files found in data/raw/evidence/")
            checker = ForensicEvidenceChecker(
                model=model,
                reference_profile=reference_profile,
                norm_params=norm_params,
            )
            check_evidence_directory(checker)
        else:
            print(f"\n▶ STEP 11: No evidence files to check.")
            print(f"  To check new data, place files in: {config.EVIDENCE_DIR}")
            print(f"  Or run: python main.py --check <filepath>")

    # ── FINAL SUMMARY ─────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  PIPELINE COMPLETE — SUMMARY")
    print("═" * 60)
    print(f"  Data source:     {data_source}")
    print(f"  Training samples: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")
    print(f"  Transformer F1:  {transformer_metrics['f1_score']:.4f}")
    if rf_metrics:
        print(f"  Random Forest F1: {rf_metrics['f1_score']:.4f}")
    print(f"  Inference Speed: {speed_results['mean_ms']:.2f} ms/seq")
    print(f"  10-min Flight:   {speed_results['full_flight_10min_seconds']:.2f} sec")
    print(f"  NFR-01 (< 5s):   {'PASS' if speed_results['nfr01_met'] else 'FAIL'}")
    if reference_profile:
        print(f"  Reference:       {len(reference_profile)} features profiled")
    print(f"\n  All outputs saved to: {config.LOGS_DIR}")
    print(f"\n  TO CHECK NEW EVIDENCE:")
    print(f"    python main.py --check <your_file.csv>")
    print(f"    OR place files in: {config.EVIDENCE_DIR}")
    print("═" * 60 + "\n")

    return {
        "transformer_metrics": transformer_metrics,
        "rf_metrics": rf_metrics,
        "speed_results": speed_results,
        "reference_profile": reference_profile,
    }


# ============================================================================
# EVIDENCE-ONLY MODE (quick check without retraining)
# ============================================================================

def run_evidence_check(args):
    """
    Quick mode: Load saved model + reference, check new evidence.
    No training — just inference on user-provided data.
    """
    print("\n" + "═" * 60)
    print("  FORENSIC EVIDENCE CHECK MODE")
    print("═" * 60)

    # Load saved model
    model_path = os.path.join(config.MODEL_SAVE_DIR, "transformer_best.keras")
    if not os.path.exists(model_path):
        print(f"  ERROR: No trained model found at {model_path}")
        print(f"  Run 'python main.py' first to train the model.")
        sys.exit(1)

    print(f"  Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Load normalization params
    norm_path = os.path.join(config.LOGS_DIR, "normalization_params.json")
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            norm_params = json.load(f)
    else:
        norm_params = {}

    # Load reference profile
    ref_path = os.path.join(config.LOGS_DIR, "reference_profile.json")
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            reference_profile = json.load(f)
    else:
        reference_profile = {}

    # Initialize checker
    checker = ForensicEvidenceChecker(
        model=model,
        reference_profile=reference_profile,
        norm_params=norm_params,
    )

    if args.check:
        checker.check_file(args.check)
    elif args.check_dir:
        check_evidence_directory(checker, args.check_dir)
    else:
        check_evidence_directory(checker)  # Default: check evidence dir


# ============================================================================
# CLI ARGUMENT PARSER
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Transformer-Based UAV Forensic Anomaly Detection Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Full pipeline, synthetic data
  python main.py --kaggle                 # Use Kaggle datasets
  python main.py --kaggle --severity balanced  # Use specific severity
  python main.py --check evidence.csv     # Check a specific evidence file
  python main.py --check-dir ./evidence/  # Check all files in directory
  python main.py --mode evaluate --check evidence.csv  # Quick check mode

Dataset Setup:
  1. Tampered data → data/raw/tampering/
     (kaggle.com/datasets/rasikaekanayakadevlk/drone-telemetry-tampering-dataset-v2)
  2. Reference data → data/raw/reference/
     (kaggle.com/datasets/samsudeenashad/supplemental-drone-telemetry-data-and-operations-log)
  3. Your evidence → data/raw/evidence/
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["full", "train", "evaluate", "check"],
        default="full",
        help="Pipeline mode. 'check' = evidence-only mode (no training).",
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        default=False,
        help="Use Kaggle datasets (tampering + supplemental).",
    )
    parser.add_argument(
        "--severity",
        choices=["balanced", "strong", "subtle"],
        default=None,
        help="Tampering severity level to load (default: main pack CSV).",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit number of flight cases to load (for testing on large data).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=config.DATA_RAW_DIR,
        help="Directory containing telemetry files.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=True,
        help="Use synthetic data (default: True).",
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_false",
        dest="synthetic",
        help="Use real data instead of synthetic.",
    )
    parser.add_argument(
        "--num-normal",
        type=int,
        default=500,
        help="Number of synthetic normal samples.",
    )
    parser.add_argument(
        "--num-tampered",
        type=int,
        default=100,
        help="Number of synthetic tampered samples.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume training from the latest checkpoint (default: True).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Force training from scratch (ignore existing checkpoints).",
    )
    parser.add_argument(
        "--check",
        type=str,
        default=None,
        help="Path to a single evidence file to check.",
    )
    parser.add_argument(
        "--check-dir",
        type=str,
        default=None,
        help="Directory of evidence files to check.",
    )

    return parser.parse_args()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    setup_logging()
    args = parse_args()

    # If --kaggle specified, disable synthetic
    if args.kaggle:
        args.synthetic = False

    # Evidence-only check mode
    if args.mode == "check" or (args.check and args.mode == "evaluate"):
        run_evidence_check(args)
    else:
        results = run_full_pipeline(args)
