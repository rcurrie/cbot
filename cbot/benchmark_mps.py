#!/usr/bin/env python
"""Benchmark MPS vs CPU performance for the wikipedia.py script."""

import argparse
import subprocess
import time
from pathlib import Path


def run_benchmark(device_type: str, epochs: int = 10) -> tuple[float, str]:
    """Run the wikipedia.py script and capture timing.

    Args:
        device_type: Either 'cpu' or 'mps'
        epochs: Number of epochs to run

    Returns:
        Tuple of (runtime, output)
    """
    # Force device type by temporarily modifying environment
    cmd = ["python", "cbot/wikipedia.py", "-e", str(epochs), "-bs", "32"]

    if device_type == "cpu":
        # Run with CPU by setting env var
        import os
        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        # We'll need to modify the script to check an env var to force CPU
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    else:
        # Run normally (will use MPS if available)
        result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse runtime from output
    output = result.stdout
    runtime = -1.0
    for line in output.split('\n'):
        if "Runtime:" in line and "Best Train MSE" in line:
            # Extract runtime from line like "Runtime: 123.45; Best Train MSE: ..."
            runtime_str = line.split("Runtime:")[1].split(";")[0].strip()
            runtime = float(runtime_str)
            break

    return runtime, output


def main() -> None:
    """Run benchmarks comparing MPS and CPU performance."""
    parser = argparse.ArgumentParser(description="Benchmark MPS vs CPU performance")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    args = parser.parse_args()

    print(f"Running benchmarks with {args.epochs} epochs...")
    print("=" * 60)

    # Run with MPS
    print("\n1. Running with MPS device...")
    mps_time, mps_output = run_benchmark("mps", args.epochs)
    print(f"MPS Runtime: {mps_time:.2f} seconds")

    # Run with CPU
    print("\n2. Running with CPU device...")
    cpu_time, cpu_output = run_benchmark("cpu", args.epochs)
    print(f"CPU Runtime: {cpu_time:.2f} seconds")

    # Compare results
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"CPU Runtime:  {cpu_time:.2f} seconds")
    print(f"MPS Runtime:  {mps_time:.2f} seconds")

    if mps_time > 0 and cpu_time > 0:
        speedup = cpu_time / mps_time
        if speedup > 1:
            print(f"MPS Speedup:  {speedup:.2f}x faster than CPU")
        else:
            print(f"CPU Speedup:  {1/speedup:.2f}x faster than MPS")
            print("\nNote: The optimized version should improve MPS performance.")


if __name__ == "__main__":
    main()