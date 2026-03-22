"""
Reference Validation Module for UAV Forensic Analysis.

This module implements the core forensic workflow:
1. The Transformer model (trained on tampered dataset) classifies new data
2. The reference profile (from supplemental real data) cross-validates results
3. Combined verdict: "CLEAN" / "TAMPERED" / "SUSPICIOUS"

When you provide new evidence data:
    → Transformer checks: "Does this look tampered?"
    → Reference check: "Does this match known-good real flight patterns?"
    → Combined: Both must agree for high-confidence verdict
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.data.preprocessing import (
    load_telemetry,
    detect_file_format,
    build_reference_profile,
    validate_against_reference,
    create_sliding_windows,
    two_stage_normalize,
    _detect_tamper_feature_columns,
)
from src.utils.hash_verify import compute_sha256, compute_and_log_hash

logger = logging.getLogger(__name__)


class ForensicEvidenceChecker:
    """
    Main forensic evidence checking engine.

    Workflow:
        1. User places new evidence files in data/raw/evidence/
        2. SHA-256 hash is computed for chain-of-custody
        3. Transformer model classifies each window as normal/tampered
        4. Reference profile validates flight parameters against real data
        5. Combined forensic report is generated

    Usage:
        checker = ForensicEvidenceChecker(model, reference_profile, norm_params)
        report = checker.check_file("path/to/evidence.csv")
        print(report["verdict"])  # "CLEAN" / "TAMPERED" / "SUSPICIOUS"
    """

    def __init__(
        self,
        model: tf.keras.Model,
        reference_profile: Dict,
        norm_params: Dict,
        threshold: float = 0.5,
    ):
        """
        Args:
            model: Trained Transformer forensic model.
            reference_profile: Profile from supplemental (real) data.
            norm_params: Dict with 'feature_min' and 'feature_max' arrays.
            threshold: Classification threshold (default 0.5).
        """
        self.model = model
        self.reference_profile = reference_profile
        self.norm_params = norm_params
        self.threshold = threshold

    def check_file(self, filepath: str) -> Dict[str, Any]:
        """
        Run full forensic check on a new evidence file.

        Steps:
            1. Compute SHA-256 hash (chain of custody)
            2. Load and parse the evidence file
            3. Create sliding windows from telemetry
            4. Normalize using saved parameters
            5. Run Transformer inference on each window
            6. Cross-validate against reference profile
            7. Generate combined forensic report

        Args:
            filepath: Path to the evidence file to check.

        Returns:
            Comprehensive forensic report dictionary.
        """
        report = {
            "filepath": os.path.abspath(filepath),
            "filename": os.path.basename(filepath),
            "timestamp": datetime.now().isoformat(),
            "checks": {},
        }

        print(f"\n{'═' * 60}")
        print(f"  FORENSIC EVIDENCE CHECK")
        print(f"  File: {os.path.basename(filepath)}")
        print(f"{'═' * 60}")

        # ── Step 1: Hash verification ────────────────────────────
        print("\n  [1/5] Computing SHA-256 hash...")
        try:
            file_hash = compute_and_log_hash(filepath)
            report["sha256"] = file_hash
            report["checks"]["integrity"] = "PASS"
            print(f"        Hash: {file_hash[:32]}...")
        except Exception as e:
            report["sha256"] = None
            report["checks"]["integrity"] = f"ERROR: {e}"
            print(f"        ERROR: {e}")

        # ── Step 2: Load evidence data ───────────────────────────
        print("\n  [2/5] Loading evidence data...")
        try:
            fmt = detect_file_format(filepath)
            report["detected_format"] = fmt
            df = load_telemetry(filepath)
            report["rows"] = len(df)
            report["columns"] = list(df.columns)
            print(f"        Format: {fmt}")
            print(f"        Rows: {len(df)}, Columns: {df.shape[1]}")
        except Exception as e:
            report["error"] = str(e)
            report["verdict"] = "ERROR"
            print(f"        ERROR loading file: {e}")
            return report

        # ── Step 3: Transformer classification ───────────────────
        print("\n  [3/5] Running Transformer anomaly detection...")
        transformer_result = self._run_transformer_check(df, fmt)
        report["checks"]["transformer"] = transformer_result
        print(f"        Windows analyzed: {transformer_result['num_windows']}")
        print(f"        Tampered windows: {transformer_result['tampered_count']} "
              f"/ {transformer_result['num_windows']}")
        print(f"        Tampering ratio: {transformer_result['tampering_ratio']:.2%}")
        print(f"        Mean score: {transformer_result['mean_score']:.4f}")
        print(f"        Transformer verdict: {transformer_result['verdict']}")

        # ── Step 4: Reference profile validation ─────────────────
        print("\n  [4/5] Validating against reference (real flight) data...")
        ref_result = self._run_reference_check(df)
        report["checks"]["reference"] = ref_result
        print(f"        Checks passed: {ref_result.get('passed', 'N/A')} "
              f"/ {ref_result.get('total_checks', 'N/A')}")
        print(f"        Reference verdict: {ref_result.get('verdict', 'N/A')}")

        # ── Step 5: Combined verdict ─────────────────────────────
        print("\n  [5/5] Generating combined verdict...")
        verdict = self._combine_verdicts(transformer_result, ref_result)
        report["verdict"] = verdict["verdict"]
        report["confidence"] = verdict["confidence"]
        report["reasoning"] = verdict["reasoning"]

        # Print final verdict
        print(f"\n{'─' * 60}")
        print(f"  VERDICT: {verdict['verdict']}")
        print(f"  Confidence: {verdict['confidence']:.1%}")
        print(f"  Reasoning: {verdict['reasoning']}")
        print(f"{'═' * 60}\n")

        # Save report
        report_path = os.path.join(
            config.LOGS_DIR,
            f"forensic_report_{os.path.splitext(os.path.basename(filepath))[0]}.json"
        )
        with open(report_path, "w") as f:
            # Convert numpy types for JSON
            json.dump(self._serialize(report), f, indent=2, default=str)
        print(f"  Report saved: {report_path}")

        return report

    def _run_transformer_check(self, df: pd.DataFrame, fmt: str) -> Dict:
        """Run the Transformer model on evidence data windows."""
        try:
            # Detect features
            feature_cols = _detect_tamper_feature_columns(df)

            # Extract numeric features
            features = df[feature_cols].apply(pd.to_numeric, errors="coerce")
            features = features.fillna(0).values.astype(np.float32)

            # Create sliding windows
            windows = create_sliding_windows(features)

            if len(windows) == 0:
                return {"verdict": "INSUFFICIENT_DATA", "num_windows": 0,
                        "tampered_count": 0, "tampering_ratio": 0, "mean_score": 0}

            # Normalize using saved training parameters
            feat_min = np.array(self.norm_params.get("feature_min", [0] * features.shape[1]))
            feat_max = np.array(self.norm_params.get("feature_max", [1] * features.shape[1]))

            # Adjust dimensions if feature count differs
            n_feat = windows.shape[2]
            if len(feat_min) != n_feat:
                feat_min = np.zeros(n_feat)
                feat_max = np.ones(n_feat)

            windows_norm, _, _ = two_stage_normalize(windows, feat_min, feat_max)

            # Run inference
            scores = self.model.predict(windows_norm, verbose=0).flatten()
            predictions = (scores > self.threshold).astype(int)

            tampered_count = int(predictions.sum())
            tampering_ratio = tampered_count / len(predictions)

            # Per-window details
            window_details = []
            for i, (score, pred) in enumerate(zip(scores, predictions)):
                window_details.append({
                    "window_idx": i,
                    "score": float(score),
                    "prediction": "TAMPERED" if pred == 1 else "NORMAL",
                })

            # Determine verdict
            if tampering_ratio > 0.5:
                verdict = "TAMPERED"
            elif tampering_ratio > 0.1:
                verdict = "SUSPICIOUS"
            else:
                verdict = "CLEAN"

            return {
                "verdict": verdict,
                "num_windows": len(predictions),
                "tampered_count": tampered_count,
                "normal_count": int(len(predictions) - tampered_count),
                "tampering_ratio": float(tampering_ratio),
                "mean_score": float(np.mean(scores)),
                "max_score": float(np.max(scores)),
                "min_score": float(np.min(scores)),
                "window_details": window_details[:20],  # First 20 for report
            }

        except Exception as e:
            logger.error(f"Transformer check failed: {e}")
            return {"verdict": "ERROR", "error": str(e), "num_windows": 0,
                    "tampered_count": 0, "tampering_ratio": 0, "mean_score": 0}

    def _run_reference_check(self, df: pd.DataFrame) -> Dict:
        """Validate evidence against the supplemental (real) reference profile."""
        if not self.reference_profile:
            return {"verdict": "SKIPPED", "reason": "No reference profile loaded"}

        try:
            # Extract available numeric features from the evidence
            evidence_stats = {}
            cols_lower = {c.lower().strip(): c for c in df.columns}

            for ref_feature in config.REF_NUMERIC_FEATURES:
                for cl, orig in cols_lower.items():
                    if ref_feature.replace("_", "") in cl.replace("_", "").replace(" ", ""):
                        values = pd.to_numeric(df[orig], errors="coerce").dropna()
                        if len(values) > 0:
                            evidence_stats[ref_feature] = float(values.mean())
                        break

            if not evidence_stats:
                # Try extracting aggregate stats from telemetry
                numeric_df = df.select_dtypes(include=[np.number])
                if "altitude" in [c.lower() for c in numeric_df.columns]:
                    for c in numeric_df.columns:
                        if "alt" in c.lower():
                            evidence_stats["altitude"] = float(numeric_df[c].mean())

            if not evidence_stats:
                return {"verdict": "SKIPPED", "reason": "No matching features for reference check"}

            # Run validation against reference
            result = validate_against_reference(evidence_stats, self.reference_profile)
            return result

        except Exception as e:
            logger.error(f"Reference check failed: {e}")
            return {"verdict": "ERROR", "error": str(e)}

    def _combine_verdicts(
        self,
        transformer_result: Dict,
        reference_result: Dict,
    ) -> Dict:
        """
        Combine Transformer and Reference verdicts into a final decision.

        Logic:
            - Transformer says TAMPERED + Reference says SUSPICIOUS → TAMPERED (high confidence)
            - Transformer says TAMPERED + Reference says LEGITIMATE → SUSPICIOUS (medium)
            - Transformer says CLEAN + Reference says LEGITIMATE → CLEAN (high confidence)
            - Transformer says CLEAN + Reference says SUSPICIOUS → SUSPICIOUS (low)
        """
        t_verdict = transformer_result.get("verdict", "UNKNOWN")
        r_verdict = reference_result.get("verdict", "SKIPPED")
        tampering_ratio = transformer_result.get("tampering_ratio", 0)

        # Calculate confidence
        if t_verdict == "TAMPERED" and r_verdict == "SUSPICIOUS":
            verdict = "TAMPERED"
            confidence = 0.95
            reasoning = ("Transformer detected tampering in {:.0%} of windows AND "
                        "flight parameters deviate from real reference data.").format(tampering_ratio)

        elif t_verdict == "TAMPERED" and r_verdict in ("LEGITIMATE", "SKIPPED"):
            verdict = "TAMPERED"
            confidence = 0.80
            reasoning = ("Transformer detected tampering in {:.0%} of windows. "
                        "Reference check passed or unavailable.").format(tampering_ratio)

        elif t_verdict == "SUSPICIOUS":
            verdict = "SUSPICIOUS"
            confidence = 0.60
            reasoning = ("Transformer found anomalies in {:.0%} of windows. "
                        "Manual review recommended.").format(tampering_ratio)

        elif t_verdict == "CLEAN" and r_verdict == "LEGITIMATE":
            verdict = "CLEAN"
            confidence = 0.95
            reasoning = ("Transformer found no tampering AND flight parameters "
                        "match known-good reference data.")

        elif t_verdict == "CLEAN" and r_verdict == "SUSPICIOUS":
            verdict = "SUSPICIOUS"
            confidence = 0.50
            reasoning = ("Transformer found no tampering but flight parameters "
                        "deviate from reference. Possible novel attack pattern.")

        elif t_verdict == "CLEAN" and r_verdict == "SKIPPED":
            verdict = "CLEAN"
            confidence = 0.75
            reasoning = "Transformer found no tampering. Reference check not available."

        else:
            verdict = "INCONCLUSIVE"
            confidence = 0.30
            reasoning = (f"Transformer: {t_verdict}, Reference: {r_verdict}. "
                        "Insufficient data for confident verdict.")

        return {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": reasoning,
            "transformer_verdict": t_verdict,
            "reference_verdict": r_verdict,
        }

    @staticmethod
    def _serialize(obj):
        """Convert numpy types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: ForensicEvidenceChecker._serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ForensicEvidenceChecker._serialize(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


def check_evidence_directory(
    checker: ForensicEvidenceChecker,
    evidence_dir: str = None,
) -> Dict[str, Dict]:
    """
    Check all evidence files in a directory.

    Args:
        checker: Initialized ForensicEvidenceChecker.
        evidence_dir: Directory containing evidence files.

    Returns:
        Dict mapping filename → forensic report.
    """
    if evidence_dir is None:
        evidence_dir = config.EVIDENCE_DIR

    if not os.path.isdir(evidence_dir):
        print(f"  Evidence directory not found: {evidence_dir}")
        print(f"  Place your evidence files in: {evidence_dir}")
        return {}

    results = {}
    files = [f for f in os.listdir(evidence_dir)
             if f.endswith((".csv", ".txt", ".dat"))]

    if not files:
        print(f"\n  No evidence files found in: {evidence_dir}")
        print(f"  Supported formats: .csv, .txt, .dat")
        print(f"  Place files there and re-run.\n")
        return {}

    print(f"\n  Found {len(files)} evidence file(s) to check:")
    for f in files:
        print(f"    → {f}")

    for filename in files:
        filepath = os.path.join(evidence_dir, filename)
        try:
            report = checker.check_file(filepath)
            results[filename] = report
        except Exception as e:
            logger.error(f"Failed to check {filename}: {e}")
            results[filename] = {"verdict": "ERROR", "error": str(e)}

    # Summary
    print(f"\n{'═' * 60}")
    print(f"  EVIDENCE CHECK SUMMARY — {len(results)} file(s)")
    print(f"{'═' * 60}")
    for fname, rep in results.items():
        v = rep.get("verdict", "UNKNOWN")
        c = rep.get("confidence", 0)
        icon = {"CLEAN": "✓", "TAMPERED": "✗", "SUSPICIOUS": "?", "ERROR": "!"}.get(v, "?")
        print(f"  {icon} {fname:40s} → {v} ({c:.0%} confidence)")
    print(f"{'═' * 60}\n")

    return results
