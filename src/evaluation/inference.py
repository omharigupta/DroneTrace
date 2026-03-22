"""
Inference Speed Benchmarking for UAV Forensic Model.

Measures latency (ms per sequence) to verify real-time forensic capability.
Target: < 50ms per 10-second window on Apple M1.
"""

import os
import time
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)


def benchmark_inference(
    model: tf.keras.Model,
    seq_len: int = config.MAX_SEQ_LEN,
    num_features: int = config.NUM_FEATURES,
    warmup_runs: int = config.INFERENCE_WARMUP_RUNS,
    benchmark_runs: int = config.INFERENCE_BENCHMARK_RUNS,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Benchmark model inference speed.

    Args:
        model: Trained TensorFlow model.
        seq_len: Sequence length for dummy input.
        num_features: Number of features.
        warmup_runs: Discard first N runs (JIT warmup).
        benchmark_runs: Number of timed runs.
        batch_size: Batch size for inference.

    Returns:
        Dictionary with latency statistics.
    """
    print("\n" + "=" * 50)
    print("  INFERENCE SPEED BENCHMARK")
    print("=" * 50)

    # Create dummy input
    dummy_input = tf.random.normal((batch_size, seq_len, num_features))

    # Warmup runs (discard)
    print(f"  Warmup: {warmup_runs} runs...")
    for _ in range(warmup_runs):
        _ = model(dummy_input, training=False)

    # Timed runs
    print(f"  Benchmarking: {benchmark_runs} runs...")
    latencies = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        _ = model(dummy_input, training=False)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)

    results = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_features": num_features,
        "num_runs": benchmark_runs,
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }

    print(f"\n  Results (batch_size={batch_size}):")
    print(f"    Mean:   {results['mean_ms']:.2f} ms")
    print(f"    Median: {results['median_ms']:.2f} ms")
    print(f"    Std:    {results['std_ms']:.2f} ms")
    print(f"    Min:    {results['min_ms']:.2f} ms")
    print(f"    Max:    {results['max_ms']:.2f} ms")
    print(f"    P95:    {results['p95_ms']:.2f} ms")
    print(f"    P99:    {results['p99_ms']:.2f} ms")

    # Check against target
    target_ms = 50.0
    status = "PASS" if results["mean_ms"] < target_ms else "FAIL"
    print(f"\n  Target: < {target_ms} ms → {status}")
    results["target_met"] = results["mean_ms"] < target_ms

    # Estimate full 10-minute flight processing time
    windows_in_10min = (10 * 60 * config.SAMPLING_RATE_HZ - config.WINDOW_SIZE) // config.STRIDE + 1
    total_time_s = (results["mean_ms"] * windows_in_10min) / 1000
    results["full_flight_10min_seconds"] = float(total_time_s)
    results["full_flight_windows"] = int(windows_in_10min)

    print(f"\n  Full 10-min flight ({windows_in_10min} windows):")
    print(f"    Estimated time: {total_time_s:.2f} seconds")
    nfr01_status = "PASS" if total_time_s < 5.0 else "FAIL"
    print(f"    NFR-01 (< 5 sec): {nfr01_status}")
    results["nfr01_met"] = total_time_s < 5.0

    print("=" * 50)

    return results
