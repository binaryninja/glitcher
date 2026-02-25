#!/usr/bin/env python3
"""
Test script for comprehensive wanted token search functionality.

This script demonstrates how to use the comprehensive search feature to find
the most effective tokens for increasing the probability of a wanted token
before running the genetic algorithm.

Usage:
    python test_comprehensive_search.py meta-llama/Llama-3.2-1B-Instruct --wanted-token "fox"
    python test_comprehensive_search.py meta-llama/Llama-3.2-1B-Instruct --wanted-token "world" --ascii-only
    python test_comprehensive_search.py meta-llama/Llama-3.2-1B-Instruct --wanted-token "cat" --max-tokens 1000
"""

import argparse
import logging
import time
from pathlib import Path

from glitcher.genetic.reducer import GeneticProbabilityReducer


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('comprehensive_search_test.log')
        ]
    )


def test_comprehensive_search(model_name: str, wanted_token: str, base_text: str = "The quick brown",
                            ascii_only: bool = False, max_tokens: int = None,
                            include_normal_tokens: bool = True):
    """
    Test comprehensive wanted token search functionality.

    Args:
        model_name: HuggingFace model identifier
        wanted_token: Token to increase probability for
        base_text: Base text to test on
        ascii_only: Filter to ASCII-only tokens
        max_tokens: Maximum tokens to test (None = all available)
        include_normal_tokens: Include normal vocabulary tokens
    """
    print("=" * 80)
    print("🔍 COMPREHENSIVE WANTED TOKEN SEARCH TEST")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Wanted token: '{wanted_token}'")
    print(f"Base text: '{base_text}'")
    print(f"ASCII only: {ascii_only}")
    print(f"Max tokens: {max_tokens or 'All available'}")
    print(f"Include normal tokens: {include_normal_tokens}")
    print()

    # Create genetic algorithm instance
    print("📦 Initializing genetic algorithm...")
    ga = GeneticProbabilityReducer(
        model_name=model_name,
        base_text=base_text,
        wanted_token=wanted_token
    )

    # Load model
    print("🤖 Loading model and tokenizer...")
    start_time = time.time()
    ga.load_model()
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f} seconds")

    # Load tokens with comprehensive search enabled
    print("🔧 Loading tokens with comprehensive search enabled...")
    start_time = time.time()
    ga.load_glitch_tokens(
        ascii_only=ascii_only,
        include_normal_tokens=include_normal_tokens,
        comprehensive_search=True  # This is the key parameter
    )
    token_load_time = time.time() - start_time
    print(f"✅ Tokens loaded in {token_load_time:.2f} seconds")
    print(f"📊 Total available tokens: {len(ga.available_tokens):,}")

    # Get baseline probability
    print("\n📈 Getting baseline probabilities...")
    target_id, target_prob, wanted_id, wanted_prob = ga.get_baseline_probability()
    ga.target_token_id = target_id
    ga.baseline_target_probability = target_prob
    ga.wanted_token_id = wanted_id
    ga.baseline_wanted_probability = wanted_prob or 0.0

    print(f"🎯 Wanted token '{wanted_token}' baseline probability: {wanted_prob:.6f}")

    # Run comprehensive search
    print(f"\n🚀 Starting comprehensive search (max_tokens: {max_tokens or 'unlimited'})...")
    search_start_time = time.time()

    # Limit tokens for testing if specified
    if max_tokens:
        original_tokens = ga.available_tokens.copy()
        ga.available_tokens = ga.available_tokens[:max_tokens]
        print(f"🔧 Limited search to first {max_tokens:,} tokens for testing")

    comprehensive_results = ga.comprehensive_wanted_token_search(max_tokens=max_tokens)
    search_time = time.time() - search_start_time

    print(f"✅ Comprehensive search completed in {search_time:.2f} seconds")
    print(f"📊 Tested {len(comprehensive_results):,} tokens")

    # Analyze results
    print("\n📋 SEARCH RESULTS ANALYSIS")
    print("-" * 40)

    positive_impacts = [metrics for metrics in comprehensive_results.values() if metrics['wanted_impact'] > 0]
    negative_impacts = [metrics for metrics in comprehensive_results.values() if metrics['wanted_impact'] <= 0]

    print(f"📈 Positive impact tokens: {len(positive_impacts):,}")
    print(f"📉 Zero/negative impact tokens: {len(negative_impacts):,}")

    if positive_impacts:
        impacts = [m['wanted_impact'] for m in positive_impacts]
        avg_impact = sum(impacts) / len(impacts)
        max_impact = max(impacts)
        min_positive_impact = min(impacts)

        print(f"📊 Average positive impact: {avg_impact:.6f}")
        print(f"🏆 Maximum impact: {max_impact:.6f}")
        print(f"📉 Minimum positive impact: {min_positive_impact:.6f}")

        # Show excellent tokens (>90% normalized impact)
        excellent_tokens = [m for m in positive_impacts if m['wanted_normalized'] > 0.9]
        if excellent_tokens:
            print(f"🔥 Excellent tokens (>90% normalized impact): {len(excellent_tokens)}")

    # Show top results
    print(f"\n🏆 TOP 15 TOKENS FOR WANTED '{wanted_token}':")
    print("-" * 80)
    top_results = list(comprehensive_results.items())[:15]

    for i, (token_id, metrics) in enumerate(top_results, 1):
        impact_pct = metrics['wanted_normalized'] * 100
        impact_indicator = "🔥" if impact_pct > 90 else "⭐" if impact_pct > 50 else "📈" if metrics['wanted_impact'] > 0 else "📉"

        print(f"{i:2d}. Token {token_id:6d} '{metrics['token_text']:25s}' "
              f"Impact: {metrics['wanted_impact']:8.6f} ({impact_pct:5.1f}%) "
              f"Prob: {metrics['wanted_prob_before']:.4f} → {metrics['wanted_prob_after']:.4f} {impact_indicator}")

    # Demonstrate token reordering
    print(f"\n🔄 TOKEN PRIORITIZATION DEMONSTRATION")
    print("-" * 50)
    print("The comprehensive search automatically reorders available tokens")
    print("to prioritize high-impact tokens for genetic algorithm initialization.")
    print()
    print("Top 10 prioritized tokens (now at beginning of available_tokens):")
    for i, token_id in enumerate(ga.available_tokens[:10], 1):
        if token_id in comprehensive_results:
            metrics = comprehensive_results[token_id]
            impact_indicator = "🔥" if metrics['wanted_normalized'] > 0.9 else "⭐" if metrics['wanted_normalized'] > 0.5 else "📈"
            print(f"  {i:2d}. Token {token_id:6d} '{metrics['token_text']:20s}' Impact: {metrics['wanted_impact']:8.6f} {impact_indicator}")

    # Performance summary
    print(f"\n⚡ PERFORMANCE SUMMARY")
    print("-" * 30)
    total_time = load_time + token_load_time + search_time
    tokens_per_second = len(comprehensive_results) / search_time if search_time > 0 else 0

    print(f"Model loading time: {load_time:.2f}s")
    print(f"Token loading time: {token_load_time:.2f}s")
    print(f"Search time: {search_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Search speed: {tokens_per_second:.0f} tokens/second")

    # Save detailed results
    results_file = f"comprehensive_search_results_{wanted_token}_{int(time.time())}.json"
    ga.save_token_impact_results(results_file)
    print(f"\n💾 Detailed results saved to: {results_file}")

    print("\n" + "=" * 80)
    print("✅ COMPREHENSIVE SEARCH TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return comprehensive_results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Test comprehensive wanted token search functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_comprehensive_search.py meta-llama/Llama-3.2-1B-Instruct --wanted-token "fox"
  python test_comprehensive_search.py meta-llama/Llama-3.2-1B-Instruct --wanted-token "world" --ascii-only
  python test_comprehensive_search.py meta-llama/Llama-3.2-1B-Instruct --wanted-token "cat" --max-tokens 1000
  python test_comprehensive_search.py meta-llama/Llama-3.2-1B-Instruct --wanted-token "dog" --base-text "Hello"
        """
    )

    parser.add_argument("model_name", help="HuggingFace model identifier")
    parser.add_argument("--wanted-token", required=True, help="Token to increase probability for")
    parser.add_argument("--base-text", default="The quick brown", help="Base text to test on")
    parser.add_argument("--ascii-only", action="store_true", help="Filter to ASCII-only tokens")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens to test (for faster testing)")
    parser.add_argument("--no-normal-tokens", action="store_true", help="Don't include normal vocabulary tokens")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    setup_logging()

    try:
        # Run comprehensive search test
        results = test_comprehensive_search(
            model_name=args.model_name,
            wanted_token=args.wanted_token,
            base_text=args.base_text,
            ascii_only=args.ascii_only,
            max_tokens=args.max_tokens,
            include_normal_tokens=not args.no_normal_tokens
        )

        print(f"\n🎉 Test completed successfully! Found {len(results):,} token results.")

    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
