#!/usr/bin/env python3
"""
Test script for enhanced validation in mining process with ASR support

This script tests the updated mining functionality to ensure enhanced validation
with ASR (Attack Success Rate) is working correctly and compares it with standard validation methods.

Usage:
    python test_enhanced_mining.py meta-llama/Llama-3.2-1B-Instruct
    python test_enhanced_mining.py meta-llama/Llama-3.2-1B-Instruct --iterations 10 --compare
    python test_enhanced_mining.py meta-llama/Llama-3.2-1B-Instruct --asr-threshold 0.8 --num-attempts 5
"""

import argparse
import json
import time
from typing import Dict, List, Any

from glitcher.model import initialize_model_and_tokenizer, mine_glitch_tokens


def test_enhanced_mining(
    model_path: str,
    iterations: int = 5,
    batch_size: int = 4,
    k: int = 16,
    max_tokens: int = 50,
    num_attempts: int = 1,
    asr_threshold: float = 0.5,
    device: str = "cuda",
    quant_type: str = "bfloat16"
) -> Dict[str, Any]:
    """
    Test mining with enhanced validation enabled

    Args:
        model_path: Path to the model to test
        iterations: Number of mining iterations
        batch_size: Batch size for mining
        k: Number of nearest tokens to consider
        max_tokens: Maximum tokens for enhanced validation
        num_attempts: Number of validation attempts
        asr_threshold: ASR threshold for considering token a glitch
        device: Device to use
        quant_type: Quantization type

    Returns:
        Dictionary containing test results
    """
    print(f"🔍 Testing Enhanced Mining with model: {model_path}")
    print(f"Parameters: iterations={iterations}, batch_size={batch_size}, k={k}")
    print(f"Enhanced validation: max_tokens={max_tokens}, num_attempts={num_attempts}, ASR threshold={asr_threshold}")
    print("=" * 80)

    # Initialize model
    print("📦 Loading model and tokenizer...")
    model, tokenizer = initialize_model_and_tokenizer(
        model_path=model_path,
        device=device,
        quant_type=quant_type
    )
    print(f"✅ Model loaded on {model.device}")

    # Test enhanced mining
    print("\n🧪 Running mining with enhanced validation...")
    start_time = time.time()

    log_file = f"enhanced_mining_test_{int(time.time())}.jsonl"
    print(f"📝 Detailed logs: {log_file}")

    try:
        glitch_tokens, glitch_token_ids = mine_glitch_tokens(
            model=model,
            tokenizer=tokenizer,
            num_iterations=iterations,
            batch_size=batch_size,
            k=k,
            verbose=True,
            language="ENG",
            log_file=log_file,
            enhanced_validation=True,
            max_tokens=max_tokens,
            num_attempts=num_attempts,
            asr_threshold=asr_threshold
        )

        end_time = time.time()
        mining_time = end_time - start_time

        print(f"\n✅ Enhanced mining completed in {mining_time:.2f} seconds")
        print(f"📊 Results: Found {len(glitch_tokens)} validated glitch tokens")

        if glitch_tokens:
            print("\n🎯 Validated Glitch Tokens:")
            for i, (token, token_id) in enumerate(zip(glitch_tokens, glitch_token_ids)):
                print(f"  {i+1}. '{token}' (ID: {token_id})")
        else:
            print("\n❌ No glitch tokens found with enhanced validation")

        # Return results
        return {
            "method": "enhanced",
            "model_path": model_path,
            "parameters": {
                "iterations": iterations,
                "batch_size": batch_size,
                "k": k,
                "max_tokens": max_tokens,
                "num_attempts": num_attempts,
                "asr_threshold": asr_threshold
            },
            "results": {
                "glitch_tokens": glitch_tokens,
                "glitch_token_ids": glitch_token_ids,
                "count": len(glitch_tokens),
                "mining_time": mining_time
            },
            "log_file": log_file
        }

    except Exception as e:
        print(f"❌ Error during enhanced mining: {e}")
        raise


def test_standard_mining(
    model_path: str,
    iterations: int = 5,
    batch_size: int = 4,
    k: int = 16,
    device: str = "cuda",
    quant_type: str = "bfloat16"
) -> Dict[str, Any]:
    """
    Test mining with standard validation for comparison

    Args:
        model_path: Path to the model to test
        iterations: Number of mining iterations
        batch_size: Batch size for mining
        k: Number of nearest tokens to consider
        device: Device to use
        quant_type: Quantization type

    Returns:
        Dictionary containing test results
    """
    print(f"\n🔍 Testing Standard Mining with model: {model_path}")
    print("=" * 80)

    # Initialize model
    print("📦 Loading model and tokenizer...")
    model, tokenizer = initialize_model_and_tokenizer(
        model_path=model_path,
        device=device,
        quant_type=quant_type
    )

    # Test standard mining
    print("\n🧪 Running mining with standard validation...")
    start_time = time.time()

    log_file = f"standard_mining_test_{int(time.time())}.jsonl"
    print(f"📝 Detailed logs: {log_file}")

    try:
        glitch_tokens, glitch_token_ids = mine_glitch_tokens(
            model=model,
            tokenizer=tokenizer,
            num_iterations=iterations,
            batch_size=batch_size,
            k=k,
            verbose=True,
            language="ENG",
            log_file=log_file,
            enhanced_validation=False
        )

        end_time = time.time()
        mining_time = end_time - start_time

        print(f"\n✅ Standard mining completed in {mining_time:.2f} seconds")
        print(f"📊 Results: Found {len(glitch_tokens)} validated glitch tokens")

        if glitch_tokens:
            print("\n🎯 Validated Glitch Tokens:")
            for i, (token, token_id) in enumerate(zip(glitch_tokens, glitch_token_ids)):
                print(f"  {i+1}. '{token}' (ID: {token_id})")
        else:
            print("\n❌ No glitch tokens found with standard validation")

        # Return results
        return {
            "method": "standard",
            "model_path": model_path,
            "parameters": {
                "iterations": iterations,
                "batch_size": batch_size,
                "k": k
            },
            "results": {
                "glitch_tokens": glitch_tokens,
                "glitch_token_ids": glitch_token_ids,
                "count": len(glitch_tokens),
                "mining_time": mining_time
            },
            "log_file": log_file
        }

    except Exception as e:
        print(f"❌ Error during standard mining: {e}")
        raise


def compare_mining_methods(
    model_path: str,
    iterations: int = 5,
    batch_size: int = 4,
    k: int = 16,
    max_tokens: int = 50,
    num_attempts: int = 1,
    asr_threshold: float = 0.5,
    device: str = "cuda",
    quant_type: str = "bfloat16"
) -> Dict[str, Any]:
    """
    Compare enhanced vs standard mining methods

    Args:
        model_path: Path to the model to test
        iterations: Number of mining iterations
        batch_size: Batch size for mining
        k: Number of nearest tokens to consider
        max_tokens: Maximum tokens for enhanced validation
        num_attempts: Number of validation attempts
        asr_threshold: ASR threshold for enhanced validation
        device: Device to use
        quant_type: Quantization type

    Returns:
        Dictionary containing comparison results
    """
    print("\n🆚 MINING METHODS COMPARISON")
    print("=" * 80)

    # Test enhanced mining
    enhanced_results = test_enhanced_mining(
        model_path, iterations, batch_size, k, max_tokens, num_attempts, asr_threshold, device, quant_type
    )

    # Test standard mining
    standard_results = test_standard_mining(
        model_path, iterations, batch_size, k, device, quant_type
    )

    # Compare results
    print("\n📊 COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Enhanced validation: {enhanced_results['results']['count']} tokens "
          f"in {enhanced_results['results']['mining_time']:.2f}s")
    print(f"Standard validation:  {standard_results['results']['count']} tokens "
          f"in {standard_results['results']['mining_time']:.2f}s")

    # Find common tokens
    enhanced_ids = set(enhanced_results['results']['glitch_token_ids'])
    standard_ids = set(standard_results['results']['glitch_token_ids'])
    common_ids = enhanced_ids.intersection(standard_ids)

    print(f"\nCommon tokens found by both methods: {len(common_ids)}")
    print(f"Enhanced-only tokens: {len(enhanced_ids - standard_ids)}")
    print(f"Standard-only tokens: {len(standard_ids - enhanced_ids)}")

    if common_ids:
        print("\n🎯 Tokens found by both methods:")
        for token_id in common_ids:
            token = enhanced_results['results']['glitch_tokens'][
                enhanced_results['results']['glitch_token_ids'].index(token_id)
            ]
            print(f"  '{token}' (ID: {token_id})")

    return {
        "enhanced_results": enhanced_results,
        "standard_results": standard_results,
        "comparison": {
            "common_tokens": len(common_ids),
            "enhanced_only": len(enhanced_ids - standard_ids),
            "standard_only": len(standard_ids - enhanced_ids),
            "enhanced_count": enhanced_results['results']['count'],
            "standard_count": standard_results['results']['count'],
            "enhanced_time": enhanced_results['results']['mining_time'],
            "standard_time": standard_results['results']['mining_time']
        }
    }


def main():
    """Main function for testing enhanced mining"""
    parser = argparse.ArgumentParser(
        description="Test enhanced validation in mining process"
    )
    parser.add_argument(
        "model_path",
        help="Path or name of the model to test"
    )
    parser.add_argument(
        "--iterations", type=int, default=5,
        help="Number of mining iterations (default: 5)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for mining (default: 4)"
    )
    parser.add_argument(
        "--k", type=int, default=16,
        help="Number of nearest tokens to consider (default: 16)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50,
        help="Maximum tokens for enhanced validation (default: 50)"
    )
    parser.add_argument(
        "--num-attempts", type=int, default=1,
        help="Number of validation attempts (default: 1)"
    )
    parser.add_argument(
        "--asr-threshold", type=float, default=0.5,
        help="ASR threshold for considering token a glitch (default: 0.5)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--quant-type", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "int8", "int4"],
        help="Quantization type (default: bfloat16)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare enhanced vs standard mining methods"
    )
    parser.add_argument(
        "--output", type=str,
        help="Output file to save results (JSON format)"
    )

    args = parser.parse_args()

    try:
        if args.compare:
            results = compare_mining_methods(
                model_path=args.model_path,
                iterations=args.iterations,
                batch_size=args.batch_size,
                k=args.k,
                max_tokens=args.max_tokens,
                num_attempts=args.num_attempts,
                asr_threshold=getattr(args, 'asr_threshold', 0.5),
                device=args.device,
                quant_type=args.quant_type
            )
        else:
            results = test_enhanced_mining(
                model_path=args.model_path,
                iterations=args.iterations,
                batch_size=args.batch_size,
                k=args.k,
                max_tokens=args.max_tokens,
                num_attempts=args.num_attempts,
                asr_threshold=getattr(args, 'asr_threshold', 0.5),
                device=args.device,
                quant_type=args.quant_type
            )

        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")

        print("\n✅ Testing completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
