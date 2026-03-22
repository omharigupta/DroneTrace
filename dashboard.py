"""
UAV Forensic Anomaly Detection — Web Dashboard

A Flask-based dashboard for:
  - Uploading evidence files for tampering analysis
  - Viewing forensic verdicts (CLEAN / TAMPERED / SUSPICIOUS)
  - Inspecting per-window Transformer scores
  - Reference profile validation results

Run:
    python dashboard.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import sys
import json
import logging
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from werkzeug.utils import secure_filename

# Flask
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = logging.getLogger(__name__)

# ============================================================================
# APP CONFIG
# ============================================================================
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max upload
app.config["UPLOAD_FOLDER"] = config.EVIDENCE_DIR
app.secret_key = "uav-forensic-dashboard-key"

ALLOWED_EXTENSIONS = {"csv", "txt", "dat"}

# Global state — loaded once on startup
_model = None
_reference_profile = None
_norm_params = None
_checker = None
_model_loaded = False
_load_error = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model_and_checker():
    """Load saved model, reference profile, and normalization params."""
    global _model, _reference_profile, _norm_params, _checker, _model_loaded, _load_error

    import tensorflow as tf
    from src.utils.reference_validator import ForensicEvidenceChecker
    from src.data.preprocessing import load_reference_dataset, build_reference_profile

    # ── Load model ────────────────────────────────────────────
    model_path = os.path.join(config.MODEL_SAVE_DIR, "transformer_best.keras")
    if not os.path.exists(model_path):
        # Try .h5 fallback
        model_path = os.path.join(config.MODEL_SAVE_DIR, "transformer_best.h5")

    if os.path.exists(model_path):
        try:
            _model = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            _load_error = f"Failed to load model: {e}"
            logger.error(_load_error)
            return
    else:
        _load_error = (
            f"No trained model found at {config.MODEL_SAVE_DIR}. "
            "Please train the model first: python main.py"
        )
        logger.warning(_load_error)
        return

    # ── Load norm params ──────────────────────────────────────
    norm_path = os.path.join(config.MODEL_SAVE_DIR, "norm_params.json")
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            _norm_params = json.load(f)
        logger.info("Loaded normalization parameters")
    else:
        _norm_params = {"feature_min": [0] * config.NUM_FEATURES,
                        "feature_max": [1] * config.NUM_FEATURES}
        logger.warning("No norm_params.json found — using defaults")

    # ── Load reference profile ────────────────────────────────
    _reference_profile = {}
    if os.path.exists(config.REFERENCE_CSV):
        try:
            ref_df = load_reference_dataset()
            _reference_profile = build_reference_profile(ref_df)
            logger.info("Loaded reference profile from supplemental dataset")
        except Exception as e:
            logger.warning(f"Could not load reference profile: {e}")

    # ── Build checker ─────────────────────────────────────────
    _checker = ForensicEvidenceChecker(
        model=_model,
        reference_profile=_reference_profile,
        norm_params=_norm_params,
    )
    _model_loaded = True
    logger.info("ForensicEvidenceChecker ready")


# ============================================================================
# ROUTES
# ============================================================================

@app.route("/")
def index():
    """Dashboard home page."""
    # List recent reports
    reports = _get_recent_reports()
    return render_template(
        "index.html",
        model_loaded=_model_loaded,
        load_error=_load_error,
        reports=reports,
    )


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload and run forensic analysis."""
    if not _model_loaded:
        return jsonify({"error": "Model not loaded. Train the model first: python main.py"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file selected"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_name = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], saved_name)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file.save(filepath)

    # Run forensic check
    try:
        report = _checker.check_file(filepath)
        report["uploaded_filename"] = filename
        return jsonify(report)
    except Exception as e:
        logger.error(f"Analysis failed: {traceback.format_exc()}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/report/<report_name>")
def view_report(report_name):
    """View a saved forensic report."""
    report_path = os.path.join(config.LOGS_DIR, report_name)
    if not os.path.exists(report_path):
        return "Report not found", 404
    with open(report_path) as f:
        report = json.load(f)
    return jsonify(report)


@app.route("/api/status")
def api_status():
    """API health check."""
    return jsonify({
        "status": "ok",
        "model_loaded": _model_loaded,
        "model_error": _load_error,
        "supported_formats": list(ALLOWED_EXTENSIONS),
    })


# ============================================================================
# HELPERS
# ============================================================================
def _get_recent_reports():
    """Get list of recent forensic reports from logs dir."""
    reports = []
    if os.path.isdir(config.LOGS_DIR):
        for f in sorted(os.listdir(config.LOGS_DIR), reverse=True):
            if f.startswith("forensic_report_") and f.endswith(".json"):
                fpath = os.path.join(config.LOGS_DIR, f)
                try:
                    with open(fpath) as fp:
                        data = json.load(fp)
                    reports.append({
                        "name": f,
                        "filename": data.get("filename", f),
                        "verdict": data.get("verdict", "UNKNOWN"),
                        "confidence": data.get("confidence", 0),
                        "timestamp": data.get("timestamp", ""),
                    })
                except Exception:
                    pass
    return reports[:20]  # Last 20


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 60)
    print("  UAV Forensic Anomaly Detection — Dashboard")
    print("=" * 60)
    print()

    # Ensure directories exist
    os.makedirs(config.EVIDENCE_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    # Load model
    print("  Loading trained model...")
    load_model_and_checker()
    if _model_loaded:
        print("  ✓ Model loaded successfully")
    else:
        print(f"  ✗ {_load_error}")
        print("  Dashboard will start in LIMITED mode (no analysis).")
        print("  Train the model first: python main.py")
    print()

    print("  Starting dashboard at: http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop.\n")

    app.run(host="127.0.0.1", port=5000, debug=False)
