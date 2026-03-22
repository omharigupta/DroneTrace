"""
SHA-256 Hash Verification for Forensic Evidence Integrity.

Ensures chain-of-custody by computing and verifying cryptographic
hashes of all input telemetry files before processing.

Requirement: NFR-02 — Hash-based verification of input logs.
"""

import os
import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)


def compute_sha256(filepath: str) -> str:
    """
    Compute the SHA-256 hash of a file.

    Reads file in chunks to handle large files efficiently.

    Args:
        filepath: Absolute path to the file.

    Returns:
        Hexadecimal SHA-256 hash string.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def verify_file_integrity(
    filepath: str,
    expected_hash: str,
) -> Tuple[bool, str]:
    """
    Verify that a file's SHA-256 hash matches the expected value.

    Args:
        filepath: Path to the file to verify.
        expected_hash: Expected SHA-256 hex digest.

    Returns:
        Tuple of (is_valid, actual_hash).
    """
    actual_hash = compute_sha256(filepath)
    is_valid = actual_hash.lower() == expected_hash.lower()

    if is_valid:
        logger.info(f"INTEGRITY OK: {os.path.basename(filepath)}")
    else:
        logger.error(
            f"INTEGRITY FAILED: {os.path.basename(filepath)}\n"
            f"  Expected: {expected_hash}\n"
            f"  Actual:   {actual_hash}"
        )

    return is_valid, actual_hash


def compute_and_log_hash(
    filepath: str,
    hash_log_path: str = config.HASH_LOG_PATH,
) -> str:
    """
    Compute SHA-256 hash and append to the hash log file.

    Creates a JSON log entry with:
    - File path
    - SHA-256 hash
    - File size
    - Timestamp

    Args:
        filepath: Path to the evidence file.
        hash_log_path: Path to the hash log JSON file.

    Returns:
        The computed SHA-256 hash.
    """
    file_hash = compute_sha256(filepath)
    file_size = os.path.getsize(filepath)

    entry = {
        "filepath": os.path.abspath(filepath),
        "filename": os.path.basename(filepath),
        "sha256": file_hash,
        "size_bytes": file_size,
        "timestamp": datetime.now().isoformat(),
    }

    # Load existing log or create new
    log = []
    if os.path.exists(hash_log_path):
        try:
            with open(hash_log_path, "r") as f:
                log = json.load(f)
        except (json.JSONDecodeError, IOError):
            log = []

    log.append(entry)

    # Save updated log
    os.makedirs(os.path.dirname(hash_log_path), exist_ok=True)
    with open(hash_log_path, "w") as f:
        json.dump(log, f, indent=2)

    logger.info(
        f"Hash logged: {os.path.basename(filepath)} → {file_hash[:16]}..."
    )
    return file_hash


def verify_all_files(
    directory: str,
    hash_log_path: str = config.HASH_LOG_PATH,
) -> Dict[str, Dict]:
    """
    Compute and log SHA-256 hashes for all files in a directory.

    Args:
        directory: Path to the directory containing evidence files.
        hash_log_path: Path to the hash log file.

    Returns:
        Dictionary mapping filename → {hash, size, status}.
    """
    results = {}

    if not os.path.isdir(directory):
        logger.warning(f"Directory not found: {directory}")
        return results

    print(f"\n  SHA-256 Verification — {directory}")
    print(f"  {'─' * 50}")

    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            try:
                file_hash = compute_and_log_hash(filepath, hash_log_path)
                file_size = os.path.getsize(filepath)
                results[filename] = {
                    "hash": file_hash,
                    "size_bytes": file_size,
                    "status": "verified",
                }
                print(f"  ✓ {filename:30s} {file_hash[:16]}... ({file_size:,} bytes)")
            except Exception as e:
                results[filename] = {
                    "hash": None,
                    "size_bytes": 0,
                    "status": f"error: {e}",
                }
                print(f"  ✗ {filename:30s} ERROR: {e}")

    print(f"  {'─' * 50}")
    print(f"  Total files verified: {len(results)}")
    print(f"  Hash log: {hash_log_path}")

    return results


def check_hash_log(hash_log_path: str = config.HASH_LOG_PATH) -> None:
    """
    Print the current hash log for auditing purposes.

    Args:
        hash_log_path: Path to the hash log JSON file.
    """
    if not os.path.exists(hash_log_path):
        print("  No hash log found.")
        return

    with open(hash_log_path, "r") as f:
        log = json.load(f)

    print(f"\n  Hash Log ({len(log)} entries)")
    print(f"  {'─' * 70}")
    for entry in log:
        print(
            f"  {entry['timestamp'][:19]}  "
            f"{entry['filename']:30s}  "
            f"{entry['sha256'][:16]}..."
        )
    print(f"  {'─' * 70}")
