#!/usr/bin/env python3
"""
Enhanced Validation Testing Script

This script demonstrates the enhanced glitch token validation that generates
multiple tokens and searches for the target token in the generated sequence.

Usage:
    python test_enhanced_validation.py meta-llama/Llama-3.2-1B-Instruct
"""

import argparse
import json
import sys
from glitcher.model import initialize_model_and_tokenizer, get_template_for_model
from glitcher.enhanced_validation import enhanced_glitch_verify, compare_validation_methods


def test_known_tokens():
    """Test some known tokens to demonstrate the enhanced validation"""
    return [
        # Known glitch tokens
        89472,   # useRalative
        127438,  # ▍▍▍▍▍▍▍▍▍▍▍▍▍▍▍▍
        85069,   # PostalCodesNL

        # Normal tokens that should not be glitch tokens
        1,       # "
        2,       # #
        300,     # as
        100,     # � (replacement character - interesting edge case)

        # Common English words
        262,     # the
        323,     # and
        374,     # is
        311,     # to
    ]


def run_enhanced_validation_demo(model_path, device="cuda", max_tokens=50, num_attempts=1):
    """Run enhanced validation demonstration"""
    print(f"Enhanced Glitch Token Validation Demo")
    print(f"Model: {model_path}")
    print(f"Max tokens to generate: {max_tokens}")
    if num_attempts > 1:
        print(f"Number of attempts per token: {num_attempts}")
    print("=" * 60)

    # Initialize model
    print("Loading model...")
    model, tokenizer = initialize_model_and_tokenizer(model_path, device, "bfloat16")
    chat_template = get_template_for_model(model.config._name_or_path, tokenizer)

    # Test tokens
    test_tokens = test_known_tokens()

    print(f"\nTesting {len(test_tokens)} tokens with enhanced validation:\n")

    results = []
    for i, token_id in enumerate(test_tokens):
        try:
            token_text = tokenizer.decode([token_id])
            print(f"[{i+1}/{len(test_tokens)}] Testing token {token_id}: '{token_text}'")

            # Run enhanced validation
            is_glitch = enhanced_glitch_verify(
                model, tokenizer, token_id, chat_template, max_tokens=max_tokens, num_attempts=num_attempts
            )

            result = {
                "token_id": token_id,
                "token": token_text,
                "is_glitch": is_glitch
            }
            results.append(result)

            status = "GLITCH" if is_glitch else "NORMAL"
            print(f"   Result: {status}")

        except Exception as e:
            print(f"   Error: {e}")

        print()

    # Summary
    glitch_count = sum(1 for r in results if r.get("is_glitch", False))
    print("=" * 60)
    print("SUMMARY:")
    print(f"Total tokens tested: {len(results)}")
    print(f"Glitch tokens found: {glitch_count}")
    print(f"Normal tokens: {len(results) - glitch_count}")
    print(f"Glitch rate: {glitch_count / len(results) * 100:.1f}%")

    return results


def run_comparison_demo(model_path, device="cuda", max_tokens=50, num_attempts=1):
    """Run comparison between standard and enhanced validation"""
    print(f"\nValidation Methods Comparison Demo")
    print(f"Model: {model_path}")
    if num_attempts > 1:
        print(f"Number of attempts per token: {num_attempts}")
    print("=" * 60)

    # Initialize model
    print("Loading model...")
    model, tokenizer = initialize_model_and_tokenizer(model_path, device, "bfloat16")
    chat_template = get_template_for_model(model.config._name_or_path, tokenizer)

    # Test tokens
    test_tokens = test_known_tokens()

    print(f"\nComparing standard vs enhanced validation for {len(test_tokens)} tokens:\n")

    comparisons = []
    agreements = 0
    disagreements = 0

    for i, token_id in enumerate(test_tokens):
        try:
            token_text = tokenizer.decode([token_id])
            print(f"[{i+1}/{len(test_tokens)}] Token {token_id}: '{token_text}'")

            # Run comparison
            result = compare_validation_methods(
                model, tokenizer, token_id, chat_template, max_tokens, num_attempts=num_attempts
            )

            comparisons.append(result)

            if result["methods_agree"]:
                agreements += 1
                print(f"   Standard: {result['original_method']}, Enhanced: {result['enhanced_method']} ✓ AGREE")
            else:
                disagreements += 1
                print(f"   Standard: {result['original_method']}, Enhanced: {result['enhanced_method']} ✗ DISAGREE")
                print(f"   Difference: {result['difference']}")

        except Exception as e:
            print(f"   Error: {e}")

        print()

    # Summary
    print("=" * 60)
    print("COMPARISON SUMMARY:")
    print(f"Total tokens tested: {len(comparisons)}")
    print(f"Methods agree: {agreements}")
    print(f"Methods disagree: {disagreements}")
    print(f"Agreement rate: {agreements / len(comparisons) * 100:.1f}%" if comparisons else "N/A")

    if disagreements > 0:
        print("\nDisagreements show cases where:")
        print("- Enhanced method is more lenient: Model can generate the token in context")
        print("- Enhanced method is more strict: Model cannot generate the token even with multiple attempts")

    return comparisons


def save_results(results, filename):
    """Save results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced glitch token validation demo")
    parser.add_argument("model_path", help="Path or name of the model to test")
    parser.add_argument("--device", default="cuda", help="Device to use (default: cuda)")
    parser.add_argument("--max-tokens", type=int, default=50,
                       help="Maximum tokens to generate (default: 50)")
    parser.add_argument("--num-attempts", type=int, default=1,
                       help="Number of times to test each token (default: 1)")
    parser.add_argument("--compare", action="store_true",
                       help="Run comparison between standard and enhanced methods")
    parser.add_argument("--output", help="Output file for results")

    args = parser.parse_args()

    try:
        if args.compare:
            results = run_comparison_demo(args.model_path, args.device, args.max_tokens, args.num_attempts)
            default_output = "enhanced_validation_comparison.json"
        else:
            results = run_enhanced_validation_demo(args.model_path, args.device, args.max_tokens, args.num_attempts)
            default_output = "enhanced_validation_results.json"

        if args.output:
            save_results(results, args.output)
        else:
            save_results(results, default_output)

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
