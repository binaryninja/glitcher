#!/usr/bin/env python3
"""
Quick Enhanced Validation Script

This script validates existing glitch tokens using the new enhanced validation method.
It's designed to quickly test tokens from your existing glitch_tokens.json file.

Usage:
    python validate_existing_tokens.py meta-llama/Llama-3.2-1B-Instruct
    python validate_existing_tokens.py meta-llama/Llama-3.2-1B-Instruct --input custom_tokens.json
    python validate_existing_tokens.py meta-llama/Llama-3.2-1B-Instruct --max-tokens 100 --sample 20
"""

import argparse
import json
import time
import sys
import random
from pathlib import Path
from typing import List, Dict, Any

from glitcher.model import initialize_model_and_tokenizer
from glitcher.enhanced_validation import enhanced_glitch_verify


def load_tokens(filepath: str) -> Dict[str, Any]:
    """Load tokens from JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Handle different file formats
        if isinstance(data, list):
            return {"glitch_token_ids": data}
        elif "glitch_token_ids" in data:
            return data
        elif "discovered_token_ids" in data:
            return {"glitch_token_ids": data["discovered_token_ids"]}
        else:
            print(f"❌ Unrecognized token file format in {filepath}")
            sys.exit(1)

    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in {filepath}")
        sys.exit(1)


def validate_tokens(
    model_path: str,
    token_ids: List[int],
    max_tokens: int = 50,
    device: str = "cuda",
    quant_type: str = "float16",
    sample_size: int = None
) -> Dict[str, Any]:
    """
    Validate tokens using enhanced validation method.

    Args:
        model_path: Path to the model
        token_ids: List of token IDs to validate
        max_tokens: Maximum tokens to generate during validation
        device: Device to use
        quant_type: Quantization type
        sample_size: If provided, randomly sample this many tokens

    Returns:
        Dictionary containing validation results
    """
    print(f"🔍 Enhanced Validation of Existing Tokens")
    print(f"Model: {model_path}")
    print(f"Total tokens: {len(token_ids)}")

    # Sample tokens if requested
    if sample_size and sample_size < len(token_ids):
        print(f"📊 Sampling {sample_size} random tokens")
        token_ids = random.sample(token_ids, sample_size)

    print(f"Testing: {len(token_ids)} tokens")
    print(f"Max tokens per test: {max_tokens}")
    print("=" * 60)

    # Initialize model
    print("Loading model...")
    model, tokenizer = initialize_model_and_tokenizer(model_path, device, quant_type)

    # Create log file
    log_file = f"token_validation_{int(time.time())}.jsonl"
    print(f"Detailed logs: {log_file}")

    # Add header to log file
    with open(log_file, 'w') as f:
        f.write("# Enhanced Token Validation Results\n")
        f.write(json.dumps({
            "event": "start_validation",
            "model_path": model_path,
            "num_tokens": len(token_ids),
            "max_tokens": max_tokens,
            "device": device,
            "quant_type": quant_type,
            "timestamp": time.time()
        }) + "\n")

    # Validate each token
    results = []
    confirmed_glitch = []
    false_positives = []

    print("\nValidating tokens:")
    for i, token_id in enumerate(token_ids):
        try:
            token_text = tokenizer.decode([token_id])
            print(f"[{i+1}/{len(token_ids)}] Testing '{token_text}' (ID: {token_id})")

            # Log token being tested
            with open(log_file, 'a') as f:
                f.write(json.dumps({
                    "event": "start_token_validation",
                    "token": token_text,
                    "token_id": token_id,
                    "index": i + 1,
                    "total": len(token_ids)
                }) + "\n")

            # Run enhanced validation
            is_glitch = enhanced_glitch_verify(
                model=model,
                tokenizer=tokenizer,
                token_id=token_id,
                log_file=log_file,
                max_tokens=max_tokens,
                quiet=True
            )

            result = {
                "token_id": token_id,
                "token": token_text,
                "is_glitch": is_glitch,
                "validation_method": "enhanced"
            }
            results.append(result)

            if is_glitch:
                confirmed_glitch.append(result)
                status = "✅ CONFIRMED GLITCH"
            else:
                false_positives.append(result)
                status = "❌ FALSE POSITIVE"

            print(f"   {status}")

            # Log result
            with open(log_file, 'a') as f:
                f.write(json.dumps({
                    "event": "token_validation_result",
                    "token": token_text,
                    "token_id": token_id,
                    "is_glitch": is_glitch
                }) + "\n")

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            # Log error
            with open(log_file, 'a') as f:
                f.write(json.dumps({
                    "event": "validation_error",
                    "token_id": token_id,
                    "error": str(e)
                }) + "\n")

    # Calculate statistics
    total_tested = len(results)
    confirmed_count = len(confirmed_glitch)
    false_positive_count = len(false_positives)
    confirmation_rate = confirmed_count / total_tested * 100 if total_tested > 0 else 0

    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total tokens tested: {total_tested}")
    print(f"Confirmed glitch tokens: {confirmed_count}")
    print(f"False positives: {false_positive_count}")
    print(f"Confirmation rate: {confirmation_rate:.1f}%")

    if confirmed_count > 0:
        print(f"\n✅ Top 5 confirmed glitch tokens:")
        for i, result in enumerate(confirmed_glitch[:5]):
            print(f"   {i+1}. '{result['token']}' (ID: {result['token_id']})")

    if false_positive_count > 0:
        print(f"\n❌ Top 5 false positives:")
        for i, result in enumerate(false_positives[:5]):
            print(f"   {i+1}. '{result['token']}' (ID: {result['token_id']})")

    # Add summary to log file
    with open(log_file, 'a') as f:
        f.write(json.dumps({
            "event": "validation_summary",
            "total_tested": total_tested,
            "confirmed_glitch": confirmed_count,
            "false_positives": false_positive_count,
            "confirmation_rate": confirmation_rate
        }) + "\n")

    return {
        "model_path": model_path,
        "validation_config": {
            "max_tokens": max_tokens,
            "device": device,
            "quant_type": quant_type,
            "sample_size": sample_size
        },
        "results": results,
        "confirmed_glitch_tokens": [r['token'] for r in confirmed_glitch],
        "confirmed_glitch_token_ids": [r['token_id'] for r in confirmed_glitch],
        "false_positives": [r['token'] for r in false_positives],
        "false_positive_ids": [r['token_id'] for r in false_positives],
        "summary": {
            "total_tested": total_tested,
            "confirmed_glitch": confirmed_count,
            "false_positives": false_positive_count,
            "confirmation_rate": confirmation_rate
        },
        "log_file": log_file,
        "timestamp": time.time()
    }


def save_results(data: Dict[str, Any], filename: str):
    """Save results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Results saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Validate existing glitch tokens with enhanced validation")
    parser.add_argument("model_path", help="Path or name of the model to use")
    parser.add_argument("--input", default="glitch_tokens.json",
                       help="Input file with tokens to validate (default: glitch_tokens.json)")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--max-tokens", type=int, default=50,
                       help="Maximum tokens to generate in validation (default: 50)")
    parser.add_argument("--sample", type=int, help="Randomly sample this many tokens for testing")
    parser.add_argument("--device", default="cuda", help="Device to use (default: cuda)")
    parser.add_argument("--quant-type", default="float16",
                       choices=["bfloat16", "float16", "int8", "int4"],
                       help="Quantization type (default: float16)")

    args = parser.parse_args()

    try:
        # Load tokens
        print(f"📂 Loading tokens from: {args.input}")
        token_data = load_tokens(args.input)
        token_ids = token_data.get("glitch_token_ids", [])

        if not token_ids:
            print("❌ No token IDs found in input file")
            sys.exit(1)

        # Run validation
        results = validate_tokens(
            model_path=args.model_path,
            token_ids=token_ids,
            max_tokens=args.max_tokens,
            device=args.device,
            quant_type=args.quant_type,
            sample_size=args.sample
        )

        # Save results
        output_file = args.output or f"validated_tokens_{int(time.time())}.json"
        save_results(results, output_file)

        print("\n🎉 Validation completed successfully!")

    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
