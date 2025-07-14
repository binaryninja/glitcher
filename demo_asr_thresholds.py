#!/usr/bin/env python3
"""
ASR (Attack Success Rate) Threshold Demonstration

This script demonstrates how different ASR thresholds affect glitch token detection.
It shows how the same token can be classified differently based on the threshold setting.

Usage:
    python demo_asr_thresholds.py meta-llama/Llama-3.2-1B-Instruct
    python demo_asr_thresholds.py meta-llama/Llama-3.2-1B-Instruct --token-ids 89472,127438,85069
    python demo_asr_thresholds.py meta-llama/Llama-3.2-1B-Instruct --num-attempts 5 --max-tokens 100
"""

import argparse
import json
import time
from typing import List, Dict, Any

from glitcher.model import initialize_model_and_tokenizer, get_template_for_model
from glitcher.enhanced_validation import enhanced_glitch_verify


def test_token_with_asr_thresholds(
    model,
    tokenizer,
    token_id: int,
    chat_template,
    thresholds: List[float] = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
    num_attempts: int = 5,
    max_tokens: int = 50
) -> Dict[str, Any]:
    """
    Test a single token with different ASR thresholds to show classification differences.

    Args:
        model: The language model
        tokenizer: The tokenizer
        token_id: Token ID to test
        chat_template: Chat template for the model
        thresholds: List of ASR thresholds to test
        num_attempts: Number of validation attempts
        max_tokens: Maximum tokens to generate

    Returns:
        Dictionary containing test results for all thresholds
    """
    token = tokenizer.decode([token_id])

    print(f"\n🎯 Testing Token: '{token}' (ID: {token_id})")
    print(f"   Attempts: {num_attempts}, Max tokens: {max_tokens}")
    print("-" * 60)

    results = {
        "token": token,
        "token_id": token_id,
        "num_attempts": num_attempts,
        "max_tokens": max_tokens,
        "threshold_results": {}
    }

    # Test with first threshold to get detailed attempt data
    log_file = f"asr_demo_{token_id}_{int(time.time())}.jsonl"
    first_result, actual_asr = enhanced_glitch_verify(
        model, tokenizer, token_id, chat_template, log_file,
        max_tokens, quiet=True, num_attempts=num_attempts, asr_threshold=thresholds[0]
    )

    # Calculate glitch attempts from ASR
    glitch_attempts = int(actual_asr * num_attempts)

    # Parse log to get additional details if needed
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                try:
                    data = json.loads(line)
                    if data.get("event") == "enhanced_token_verification":
                        glitch_attempts = data.get("glitch_attempts", glitch_attempts)
                        break
                except:
                    continue
    except FileNotFoundError:
        pass

    print(f"   Actual ASR: {actual_asr:.2%} ({glitch_attempts}/{num_attempts} attempts successful)")
    print("\n   Threshold Results:")
    print("   ASR Threshold | Classification | Reason")
    print("   --------------|---------------|------------------")

    # Test all thresholds
    for threshold in thresholds:
        is_glitch = actual_asr >= threshold
        status = "GLITCH" if is_glitch else "normal"
        reason = f"ASR {actual_asr:.2%} >= {threshold:.1%}" if is_glitch else f"ASR {actual_asr:.2%} < {threshold:.1%}"

        print(f"   {threshold:12.1%} | {status:>13} | {reason}")

        results["threshold_results"][threshold] = {
            "asr_threshold": threshold,
            "is_glitch": is_glitch,
            "classification": status,
            "reason": reason
        }

    # Add actual ASR data
    results["actual_asr"] = actual_asr
    results["glitch_attempts"] = glitch_attempts
    results["log_file"] = log_file

    return results


def demonstrate_asr_impact(
    model_path: str,
    token_ids: List[int] = None,
    num_attempts: int = 5,
    max_tokens: int = 50,
    device: str = "cuda",
    quant_type: str = "bfloat16"
):
    """
    Demonstrate the impact of ASR thresholds on token classification.

    Args:
        model_path: Path to the model to test
        token_ids: List of token IDs to test (will use defaults if None)
        num_attempts: Number of validation attempts per token
        max_tokens: Maximum tokens to generate during validation
        device: Device to use
        quant_type: Quantization type
    """
    print("🧪 ASR Threshold Demonstration")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Validation parameters: {num_attempts} attempts, {max_tokens} max tokens")
    print()

    # Load model
    print("📦 Loading model and tokenizer...")
    model, tokenizer = initialize_model_and_tokenizer(
        model_path=model_path,
        device=device,
        quant_type=quant_type
    )

    # Get chat template
    chat_template = get_template_for_model(model.config._name_or_path, tokenizer)

    # Use default token IDs if none provided
    if token_ids is None:
        # Some commonly known problematic tokens (adjust based on your model)
        token_ids = [89472, 127438, 85069, 12345, 67890]
        print("🎲 Using default test tokens (adjust with --token-ids for specific tokens)")

    print(f"✅ Model loaded on {model.device}")

    # ASR thresholds to test
    thresholds = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

    all_results = []

    # Test each token
    for i, token_id in enumerate(token_ids):
        try:
            result = test_token_with_asr_thresholds(
                model, tokenizer, token_id, chat_template,
                thresholds, num_attempts, max_tokens
            )
            all_results.append(result)

        except Exception as e:
            print(f"❌ Error testing token {token_id}: {e}")
            continue

    # Summary analysis
    print("\n" + "=" * 80)
    print("📊 SUMMARY ANALYSIS")
    print("=" * 80)

    if not all_results:
        print("❌ No tokens were successfully tested")
        return

    # ASR distribution
    asr_values = [r["actual_asr"] for r in all_results]
    print(f"\n📈 ASR Distribution across {len(all_results)} tokens:")
    print(f"   Average ASR: {sum(asr_values) / len(asr_values):.2%}")
    print(f"   Min ASR: {min(asr_values):.2%}")
    print(f"   Max ASR: {max(asr_values):.2%}")

    # Threshold impact analysis
    print(f"\n🎯 Classification Impact by Threshold:")
    print("   Threshold | Tokens Classified as Glitch | Percentage")
    print("   ----------|----------------------------|----------")

    for threshold in thresholds:
        glitch_count = sum(1 for r in all_results if r["actual_asr"] >= threshold)
        percentage = (glitch_count / len(all_results)) * 100
        print(f"   {threshold:8.1%} | {glitch_count:26} | {percentage:7.1f}%")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")

    high_asr_count = sum(1 for asr in asr_values if asr >= 0.8)
    medium_asr_count = sum(1 for asr in asr_values if 0.4 <= asr < 0.8)
    low_asr_count = sum(1 for asr in asr_values if asr < 0.4)

    print(f"   • High ASR tokens (≥80%): {high_asr_count} - Definitely problematic")
    print(f"   • Medium ASR tokens (40-79%): {medium_asr_count} - Potentially problematic")
    print(f"   • Low ASR tokens (<40%): {low_asr_count} - Likely not glitches")

    if high_asr_count > 0:
        print(f"   • Consider threshold ≥ 0.8 for high-confidence glitch detection")
    if medium_asr_count > 0:
        print(f"   • Consider threshold ≥ 0.5 for balanced detection")
    if low_asr_count == len(all_results):
        print(f"   • No clear glitch tokens found - try different tokens or increase attempts")

    # Save results
    output_file = f"asr_demo_results_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "model_path": model_path,
            "test_parameters": {
                "num_attempts": num_attempts,
                "max_tokens": max_tokens,
                "thresholds_tested": thresholds
            },
            "results": all_results,
            "summary": {
                "total_tokens": len(all_results),
                "average_asr": sum(asr_values) / len(asr_values) if asr_values else 0,
                "min_asr": min(asr_values) if asr_values else 0,
                "max_asr": max(asr_values) if asr_values else 0,
                "high_asr_count": high_asr_count,
                "medium_asr_count": medium_asr_count,
                "low_asr_count": low_asr_count
            }
        }, indent=2)

    print(f"\n💾 Detailed results saved to: {output_file}")
    print("\n✅ ASR demonstration completed!")


def main():
    """Main function for ASR threshold demonstration"""
    parser = argparse.ArgumentParser(
        description="Demonstrate ASR threshold impact on glitch token detection"
    )
    parser.add_argument(
        "model_path",
        help="Path or name of the model to test"
    )
    parser.add_argument(
        "--token-ids", type=str,
        help="Comma-separated list of token IDs to test"
    )
    parser.add_argument(
        "--num-attempts", type=int, default=5,
        help="Number of validation attempts per token (default: 5)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50,
        help="Maximum tokens to generate during validation (default: 50)"
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

    args = parser.parse_args()

    # Parse token IDs if provided
    token_ids = None
    if args.token_ids:
        try:
            token_ids = [int(tid.strip()) for tid in args.token_ids.split(",")]
        except ValueError:
            print("❌ Error: Token IDs must be comma-separated integers")
            return 1

    try:
        demonstrate_asr_impact(
            model_path=args.model_path,
            token_ids=token_ids,
            num_attempts=args.num_attempts,
            max_tokens=args.max_tokens,
            device=args.device,
            quant_type=args.quant_type
        )
        return 0

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
