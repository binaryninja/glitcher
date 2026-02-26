#!/usr/bin/env python3
"""Collect and record environment details for reproducibility.

Captures model identifiers, library versions, hardware info, quantization
settings, and random seed state.  Writes a JSON snapshot that can later be
verified with verify_reproducibility.py.

Usage:
    python APPENDIX-B/collect_environment.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --quant-type bfloat16 \
        --output APPENDIX-B/snapshot.json

    # With seed fixing (sets torch + python + numpy seeds)
    python APPENDIX-B/collect_environment.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --seed 42 \
        --output APPENDIX-B/snapshot.json
"""

import argparse
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_python_version() -> str:
    return platform.python_version()


def get_library_versions() -> dict:
    versions = {"python": get_python_version()}
    for lib in ["torch", "transformers", "accelerate", "bitsandbytes", "numpy"]:
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not installed"
    # glitcher itself
    try:
        import glitcher
        versions["glitcher"] = getattr(glitcher, "__version__", "1.0.0")
    except ImportError:
        versions["glitcher"] = "1.0.0"
    return versions


def get_hardware_info(device: str = "cuda") -> dict:
    info = {
        "device": device,
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            )
            info["cuda_version"] = torch.version.cuda or "unknown"
            info["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
            info["driver_version"] = "see nvidia-smi"
            info["cudnn_deterministic"] = torch.backends.cudnn.deterministic
            info["cudnn_benchmark"] = torch.backends.cudnn.benchmark
        else:
            info["gpu_count"] = 0
            info["gpu_name"] = "N/A (CPU only)"
            info["gpu_memory_mb"] = 0
            info["cuda_version"] = "N/A"
    except ImportError:
        info["gpu_count"] = 0
        info["gpu_name"] = "torch not installed"
    return info


def get_model_info(model_path: str) -> dict:
    info = {
        "model_path": model_path,
        "tokenizer_path": model_path,
        "model_class": "AutoModelForCausalLM",
        "tokenizer_class": "AutoTokenizer",
    }
    # Detect special model types
    lower = model_path.lower()
    if "gpt-oss" in lower:
        info["is_harmony"] = True
        info["reasoning_level"] = os.environ.get("GLITCHER_REASONING_LEVEL", "medium")
    else:
        info["is_harmony"] = False

    # Detect likely chat template
    if "llama-3.2" in lower or "llama32" in lower:
        info["chat_template"] = "llama32"
    elif "llama-3.3" in lower:
        info["chat_template"] = "llama3.3"
    elif "llama-3" in lower or "llama3" in lower:
        info["chat_template"] = "llama3"
    elif "qwen" in lower:
        info["chat_template"] = "builtin"
    else:
        info["chat_template"] = "auto"
    return info


def fix_seeds(seed: int) -> dict:
    """Fix random seeds for reproducibility where supported."""
    import torch
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: full determinism also requires:
    #   torch.backends.cudnn.deterministic = True
    #   torch.backends.cudnn.benchmark = False
    #   CUBLAS_WORKSPACE_CONFIG=:4096:8 (env var)
    # These can degrade performance significantly.
    return {
        "torch_manual_seed": seed,
        "torch_cuda_manual_seed_all": seed,
        "python_random_seed": seed,
        "numpy_random_seed": seed,
        "torch_backends_cudnn_deterministic": torch.backends.cudnn.deterministic,
        "torch_backends_cudnn_benchmark": torch.backends.cudnn.benchmark,
        "non_deterministic_components": [
            "CUDA kernel scheduling (unless CUBLAS_WORKSPACE_CONFIG set)",
            "cuBLAS workspace selection",
            "torch.nn.functional operations with non-deterministic algorithms",
        ],
    }


def build_snapshot(args) -> dict:
    seed_info = fix_seeds(args.seed) if args.seed is not None else {
        "torch_manual_seed": None,
        "torch_cuda_manual_seed_all": None,
        "python_random_seed": None,
        "numpy_random_seed": None,
        "non_deterministic_components": [
            "No seeds fixed -- results will vary between runs",
            "CUDA kernel scheduling",
            "cuBLAS workspace selection",
            "random.sample() in population initialization",
            "random.randint() in mutation operators",
        ],
    }

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": get_model_info(args.model),
        "library_versions": get_library_versions(),
        "hardware": get_hardware_info(args.device),
        "quantization": args.quant_type,
        "seeds": seed_info,
        "decoding": {
            "do_sample": True,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": args.max_tokens,
            "stop_sequences": [],
        },
        "validation": {
            "enhanced_validation": True,
            "validation_tokens": args.max_tokens,
            "num_attempts": args.num_attempts,
            "asr_threshold": args.asr_threshold,
            "probe_template_count": 3,
        },
        "control_sample": {
            "tokens": ["the", "computer", "science", "model"],
        },
    }
    return snapshot


def main():
    parser = argparse.ArgumentParser(
        description="Collect environment snapshot for reproducibility"
    )
    parser.add_argument("--model", required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument(
        "--quant-type", default="bfloat16",
        choices=["auto", "bfloat16", "float16", "int8", "int4"],
        help="Quantization type (default: bfloat16)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed to fix (optional)")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max generation tokens (default: 50)")
    parser.add_argument("--num-attempts", type=int, default=1, help="Validation attempts (default: 1)")
    parser.add_argument("--asr-threshold", type=float, default=0.5, help="ASR threshold (default: 0.5)")
    parser.add_argument(
        "--output", default="APPENDIX-B/snapshot.json",
        help="Output file (default: APPENDIX-B/snapshot.json)",
    )

    args = parser.parse_args()
    snapshot = build_snapshot(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"Environment snapshot written to {output_path}")
    print(f"  Model: {args.model}")
    print(f"  Quantization: {args.quant_type}")
    print(f"  Seed: {args.seed if args.seed is not None else 'not fixed'}")
    print(f"  GPU: {snapshot['hardware'].get('gpu_name', 'N/A')}")
    print(f"  PyTorch: {snapshot['library_versions'].get('torch', 'N/A')}")
    print(f"  Transformers: {snapshot['library_versions'].get('transformers', 'N/A')}")


if __name__ == "__main__":
    main()
