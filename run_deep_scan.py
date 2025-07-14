#!/usr/bin/env python3
"""
Deep Scan Script with Enhanced Validation

This script performs a comprehensive deep scan for glitch tokens using the enhanced
validation method that generates multiple tokens and searches for the target token
in the generated sequence.

Usage:
    python run_deep_scan.py meta-llama/Llama-3.2-1B-Instruct
    python run_deep_scan.py meta-llama/Llama-3.2-1B-Instruct --iterations 100 --batch-size 16
    python run_deep_scan.py meta-llama/Llama-3.2-1B-Instruct --validation-only --input glitch_tokens.json
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any

from glitcher.model import initialize_model_and_tokenizer, mine_glitch_tokens
from glitcher.enhanced_validation import enhanced_glitch_verify, batch_enhanced_verify


def deep_scan_mining(
    model_path: str,
    num_iterations: int = 200,
    batch_size: int = 8,
    k: int = 32,
    device: str = "cuda",
    quant_type: str = "float16"
) -> Dict[str, Any]:
    """
    Perform deep scan mining to discover potential glitch tokens.

    Args:
        model_path: Path to the model
        num_iterations: Number of mining iterations
        batch_size: Batch size for mining
        k: Number of nearest tokens to consider
        device: Device to use
        quant_type: Quantization type

    Returns:
        Dictionary containing discovered tokens and metadata
    """
    print(f"🔍 Starting Deep Scan Mining")
    print(f"Model: {model_path}")
    print(f"Iterations: {num_iterations}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"Quantization: {quant_type}")
    print("=" * 60)

    # Initialize model
    print("Loading model...")
    model, tokenizer = initialize_model_and_tokenizer(model_path, device, quant_type)

    # Run mining
    log_file = f"deep_scan_mining_{int(time.time())}.jsonl"
    print(f"Mining logs: {log_file}")

    glitch_tokens, glitch_token_ids = mine_glitch_tokens(
        model=model,
        tokenizer=tokenizer,
        num_iterations=num_iterations,
        batch_size=batch_size,
        k=k,
        verbose=True,
        language="ENG",
        log_file=log_file
    )

    print(f"\n✅ Mining completed!")
    print(f"Found {len(glitch_tokens)} potential glitch tokens")

    return {
        "model_path": model_path,
        "mining_config": {
            "num_iterations": num_iterations,
            "batch_size": batch_size,
            "k": k,
            "device": device,
            "quant_type": quant_type
        },
        "discovered_tokens": glitch_tokens,
        "discovered_token_ids": glitch_token_ids,
        "mining_log": log_file,
        "timestamp": time.time()
    }


def enhanced_validation_scan(
    model_path: str,
    token_data: Dict[str, Any],
    max_tokens: int = 50,
    device: str = "cuda",
    quant_type: str = "float16"
) -> Dict[str, Any]:
    """
    Perform enhanced validation on discovered or provided tokens.

    Args:
        model_path: Path to the model
        token_data: Dictionary containing tokens to validate
        max_tokens: Maximum tokens to generate during validation
        device: Device to use
        quant_type: Quantization type

    Returns:
        Dictionary containing validation results
    """
    print(f"\n🧪 Starting Enhanced Validation")
    print(f"Validating {len(token_data.get('discovered_token_ids', []))} tokens")
    print(f"Max tokens per test: {max_tokens}")
    print("=" * 60)

    # Initialize model
    print("Loading model...")
    model, tokenizer = initialize_model_and_tokenizer(model_path, device, quant_type)

    # Prepare tokens for validation
    token_ids = token_data.get('discovered_token_ids', [])
    if not token_ids:
        # Try alternative keys
        token_ids = token_data.get('glitch_token_ids', [])

    if not token_ids:
        print("❌ No token IDs found in input data")
        return {}

    # Run enhanced validation
    log_file = f"deep_scan_validation_{int(time.time())}.jsonl"
    print(f"Validation logs: {log_file}")

    # Add header to log file
    with open(log_file, 'w') as f:
        f.write("# Enhanced Validation Deep Scan Results\n")
        f.write(json.dumps({
            "event": "start_enhanced_validation",
            "model_path": model_path,
            "num_tokens": len(token_ids),
            "max_tokens": max_tokens,
            "timestamp": time.time()
        }) + "\n")

    # Validate tokens
    validated_results = batch_enhanced_verify(
        model=model,
        tokenizer=tokenizer,
        token_ids=token_ids,
        log_file=log_file,
        max_tokens=max_tokens,
        quiet=True
    )

    # Analyze results
    confirmed_glitch_tokens = []
    confirmed_glitch_token_ids = []
    false_positives = []
    false_positive_ids = []

    for result in validated_results:
        if result['is_glitch']:
            confirmed_glitch_tokens.append(result['token'])
            confirmed_glitch_token_ids.append(result['token_id'])
        else:
            false_positives.append(result['token'])
            false_positive_ids.append(result['token_id'])

    # Summary
    print(f"\n✅ Enhanced Validation completed!")
    print(f"Total tokens tested: {len(validated_results)}")
    print(f"Confirmed glitch tokens: {len(confirmed_glitch_tokens)}")
    print(f"False positives: {len(false_positives)}")
    print(f"Accuracy improvement: {len(confirmed_glitch_tokens)}/{len(token_ids)} = {len(confirmed_glitch_tokens)/len(token_ids)*100:.1f}%")

    return {
        "model_path": model_path,
        "validation_config": {
            "max_tokens": max_tokens,
            "device": device,
            "quant_type": quant_type
        },
        "original_mining_data": token_data,
        "validation_results": validated_results,
        "confirmed_glitch_tokens": confirmed_glitch_tokens,
        "confirmed_glitch_token_ids": confirmed_glitch_token_ids,
        "false_positives": false_positives,
        "false_positive_ids": false_positive_ids,
        "validation_log": log_file,
        "timestamp": time.time(),
        "summary": {
            "total_tested": len(validated_results),
            "confirmed_glitch": len(confirmed_glitch_tokens),
            "false_positives": len(false_positives),
            "accuracy_rate": len(confirmed_glitch_tokens) / len(token_ids) if token_ids else 0
        }
    }


def save_results(data: Dict[str, Any], filename: str):
    """Save results to JSON file with pretty formatting"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Results saved to: {filename}")


def load_token_data(filepath: str) -> Dict[str, Any]:
    """Load token data from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Deep scan for glitch tokens with enhanced validation")
    parser.add_argument("model_path", help="Path or name of the model to scan")

    # Mining options
    parser.add_argument("--iterations", type=int, default=200,
                       help="Number of mining iterations (default: 200)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for mining (default: 8)")
    parser.add_argument("--k", type=int, default=32,
                       help="Number of nearest tokens to consider (default: 32)")

    # Validation options
    parser.add_argument("--validation-tokens", type=int, default=50,
                       help="Maximum tokens to generate in validation (default: 50)")
    parser.add_argument("--validation-only", action="store_true",
                       help="Skip mining, only run enhanced validation")
    parser.add_argument("--input", type=str,
                       help="Input file with tokens to validate (for --validation-only)")

    # System options
    parser.add_argument("--device", default="cuda", help="Device to use (default: cuda)")
    parser.add_argument("--quant-type", default="float16",
                       choices=["bfloat16", "float16", "int8", "int4"],
                       help="Quantization type (default: float16)")
    parser.add_argument("--output", help="Output file for results")

    args = parser.parse_args()

    print("🚀 Glitcher Deep Scan with Enhanced Validation")
    print("=" * 60)

    try:
        if args.validation_only:
            # Validation-only mode
            if not args.input:
                print("❌ --input required for --validation-only mode")
                sys.exit(1)

            print(f"📂 Loading tokens from: {args.input}")
            token_data = load_token_data(args.input)

            # Run enhanced validation
            results = enhanced_validation_scan(
                model_path=args.model_path,
                token_data=token_data,
                max_tokens=args.validation_tokens,
                device=args.device,
                quant_type=args.quant_type
            )

            # Save results
            output_file = args.output or f"enhanced_validation_results_{int(time.time())}.json"
            save_results(results, output_file)

        else:
            # Full deep scan mode
            # Step 1: Mining
            mining_results = deep_scan_mining(
                model_path=args.model_path,
                num_iterations=args.iterations,
                batch_size=args.batch_size,
                k=args.k,
                device=args.device,
                quant_type=args.quant_type
            )

            # Step 2: Enhanced validation
            validation_results = enhanced_validation_scan(
                model_path=args.model_path,
                token_data=mining_results,
                max_tokens=args.validation_tokens,
                device=args.device,
                quant_type=args.quant_type
            )

            # Combine results
            final_results = {
                "deep_scan_type": "full",
                "mining_phase": mining_results,
                "validation_phase": validation_results,
                "final_summary": {
                    "discovered_tokens": len(mining_results.get('discovered_tokens', [])),
                    "confirmed_glitch_tokens": len(validation_results.get('confirmed_glitch_tokens', [])),
                    "false_positive_rate": len(validation_results.get('false_positives', [])) / len(mining_results.get('discovered_tokens', [])) * 100 if mining_results.get('discovered_tokens') else 0,
                    "recommended_tokens": validation_results.get('confirmed_glitch_tokens', [])
                },
                "timestamp": time.time()
            }

            # Save results
            output_file = args.output or f"deep_scan_results_{int(time.time())}.json"
            save_results(final_results, output_file)

        print("\n🎉 Deep scan completed successfully!")

    except KeyboardInterrupt:
        print("\n⏹️  Deep scan interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during deep scan: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
