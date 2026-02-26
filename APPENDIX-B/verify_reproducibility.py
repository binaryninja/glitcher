#!/usr/bin/env python3
"""Verify that the current environment matches a previously saved snapshot.

Compares library versions, hardware, quantization, and seed settings against
a snapshot produced by collect_environment.py.  Exits with code 0 if all
critical fields match, 1 otherwise.

Usage:
    python APPENDIX-B/verify_reproducibility.py APPENDIX-B/snapshot.json
"""

import argparse
import json
import sys
from pathlib import Path


def get_current_versions() -> dict:
    import platform
    versions = {"python": platform.python_version()}
    for lib in ["torch", "transformers", "accelerate", "bitsandbytes", "numpy"]:
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not installed"
    return versions


def get_current_gpu() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda or "unknown",
                "gpu_count": torch.cuda.device_count(),
            }
    except ImportError:
        pass
    return {"gpu_name": "N/A", "cuda_version": "N/A", "gpu_count": 0}


def compare(snapshot: dict) -> list:
    """Return a list of (field, expected, actual, severity) mismatches."""
    mismatches = []
    current_versions = get_current_versions()
    current_gpu = get_current_gpu()

    # -- Library versions (critical) --
    saved_versions = snapshot.get("library_versions", {})
    for lib in ["torch", "transformers", "accelerate", "bitsandbytes"]:
        expected = saved_versions.get(lib, "")
        actual = current_versions.get(lib, "not installed")
        if expected and actual != expected:
            mismatches.append((f"library_versions.{lib}", expected, actual, "CRITICAL"))

    # Python major.minor
    expected_py = saved_versions.get("python", "")
    actual_py = current_versions.get("python", "")
    if expected_py and actual_py:
        if expected_py.rsplit(".", 1)[0] != actual_py.rsplit(".", 1)[0]:
            mismatches.append(("library_versions.python", expected_py, actual_py, "CRITICAL"))
        elif expected_py != actual_py:
            mismatches.append(("library_versions.python", expected_py, actual_py, "WARN"))

    # -- Hardware (warn) --
    saved_hw = snapshot.get("hardware", {})
    if saved_hw.get("gpu_name") and saved_hw["gpu_name"] != "N/A":
        if current_gpu["gpu_name"] != saved_hw["gpu_name"]:
            mismatches.append(("hardware.gpu_name", saved_hw["gpu_name"], current_gpu["gpu_name"], "WARN"))
    if saved_hw.get("cuda_version") and saved_hw["cuda_version"] != "N/A":
        if current_gpu["cuda_version"] != saved_hw["cuda_version"]:
            mismatches.append(("hardware.cuda_version", saved_hw["cuda_version"], current_gpu["cuda_version"], "WARN"))

    # -- Quantization (critical) --
    expected_quant = snapshot.get("quantization", "")
    if expected_quant:
        mismatches.append(("quantization", expected_quant, "(check CLI --quant-type)", "INFO"))

    # -- Seeds (info) --
    seeds = snapshot.get("seeds", {})
    if seeds.get("torch_manual_seed") is None:
        mismatches.append(("seeds", "not fixed", "non-deterministic run", "INFO"))

    return mismatches


def main():
    parser = argparse.ArgumentParser(description="Verify environment against snapshot")
    parser.add_argument("snapshot", help="Path to snapshot JSON file")
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        print(f"ERROR: Snapshot file not found: {snapshot_path}")
        sys.exit(1)

    with open(snapshot_path) as f:
        snapshot = json.load(f)

    print(f"Verifying against snapshot from {snapshot.get('timestamp', 'unknown')}")
    print(f"Model: {snapshot.get('model', {}).get('model_path', 'unknown')}")
    print()

    mismatches = compare(snapshot)

    has_critical = False
    for field, expected, actual, severity in mismatches:
        tag = f"[{severity}]"
        if severity == "CRITICAL":
            has_critical = True
            print(f"  {tag:10s} {field}: expected={expected}, actual={actual}")
        elif severity == "WARN":
            print(f"  {tag:10s} {field}: expected={expected}, actual={actual}")
        else:
            print(f"  {tag:10s} {field}: {expected} -> {actual}")

    print()
    if has_critical:
        print("FAIL: Critical mismatches detected. Results may not be reproducible.")
        sys.exit(1)
    else:
        print("PASS: No critical mismatches. Check WARN/INFO items above.")
        sys.exit(0)


if __name__ == "__main__":
    main()
