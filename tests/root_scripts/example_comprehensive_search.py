#!/usr/bin/env python3
"""
Example: Comprehensive Wanted Token Search

This example demonstrates how to use the comprehensive search feature to find
the most effective tokens for increasing the probability of a wanted token
before running the genetic algorithm.

The comprehensive search feature:
- Tests all available tokens in the model vocabulary
- Uses optimized batching for faster execution
- Tests multiple token positions (prefix, suffix, colon-separated)
- Automatically reorders tokens to prioritize high-impact discoveries
- Integrates seamlessly with genetic algorithm evolution

Usage:
    python example_comprehensive_search.py
"""

import sys
from pathlib import Path

# Add the glitcher package to path
sys.path.insert(0, str(Path(__file__).parent))

from glitcher.genetic.reducer import GeneticProbabilityReducer


def example_basic_comprehensive_search():
    """Basic example of comprehensive search for wanted token."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Comprehensive Search")
    print("=" * 60)

    # Create genetic algorithm instance with wanted token
    ga = GeneticProbabilityReducer(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        base_text="The quick brown",
        wanted_token="fox"  # We want to increase probability of "fox"
    )

    # Load model and enable comprehensive search
    print("Loading model...")
    ga.load_model()

    print("Loading tokens with comprehensive search enabled...")
    ga.load_glitch_tokens(
        ascii_only=True,  # Use ASCII-only for cleaner results
        include_normal_tokens=True,  # Include normal vocabulary
        comprehensive_search=True  # KEY: Enable comprehensive search
    )

    # The comprehensive search will automatically run when we start evolution
    print("\nRunning genetic algorithm with comprehensive search...")
    ga.population_size = 20
    ga.max_generations = 10  # Short run for example

    final_population = ga.run_evolution()

    # Show results
    if final_population:
        best = final_population[0]
        print(f"\nBest result: {best.tokens}")
        print(f"Wanted token 'fox' probability: {best.wanted_baseline_prob:.6f} → {best.wanted_modified_prob:.6f}")
        improvement = (best.wanted_modified_prob - best.wanted_baseline_prob) / (1.0 - best.wanted_baseline_prob) * 100
        print(f"Improvement: {improvement:.2f}%")


def example_comprehensive_search_with_gui():
    """Example with GUI monitoring of comprehensive search."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Comprehensive Search with GUI")
    print("=" * 60)

    try:
        from glitcher.genetic.gui_animator import EnhancedGeneticAnimator, GeneticAnimationCallback

        # Setup GUI
        animator = EnhancedGeneticAnimator(
            base_text="Hello",
            wanted_token="world",
            max_generations=15
        )
        gui_callback = GeneticAnimationCallback(animator)

        # Create GA instance
        ga = GeneticProbabilityReducer(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            base_text="Hello",
            wanted_token="world",
            gui_callback=gui_callback
        )

        # Configure for demo
        ga.population_size = 15
        ga.max_generations = 15

        # Load with comprehensive search
        ga.load_model()
        gui_callback.set_tokenizer(ga.tokenizer)

        ga.load_glitch_tokens(
            ascii_only=True,
            comprehensive_search=True
        )

        print("Running with GUI animation...")
        print("Watch the GUI window to see comprehensive search progress!")

        final_population = ga.run_evolution()

        # Keep GUI alive to view results
        print("Close the GUI window when done viewing results.")
        gui_callback.keep_alive()

    except ImportError:
        print("GUI not available. Install matplotlib for GUI support:")
        print("pip install matplotlib")


def example_comprehensive_search_comparison():
    """Example comparing with and without comprehensive search."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Comparison - With vs Without Comprehensive Search")
    print("=" * 60)

    # Test WITHOUT comprehensive search
    print("1. Testing WITHOUT comprehensive search...")
    ga_baseline = GeneticProbabilityReducer(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        base_text="The cat",
        wanted_token="dog"
    )
    ga_baseline.load_model()
    ga_baseline.load_glitch_tokens(
        ascii_only=True,
        include_normal_tokens=True,
        comprehensive_search=False  # Disabled
    )

    # Quick evolution
    ga_baseline.population_size = 10
    ga_baseline.max_generations = 5
    baseline_population = ga_baseline.run_evolution()
    baseline_best = baseline_population[0] if baseline_population else None

    # Test WITH comprehensive search
    print("\n2. Testing WITH comprehensive search...")
    ga_comprehensive = GeneticProbabilityReducer(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        base_text="The cat",
        wanted_token="dog"
    )
    ga_comprehensive.load_model()
    ga_comprehensive.load_glitch_tokens(
        ascii_only=True,
        include_normal_tokens=True,
        comprehensive_search=True  # Enabled
    )

    # Quick evolution
    ga_comprehensive.population_size = 10
    ga_comprehensive.max_generations = 5
    comprehensive_population = ga_comprehensive.run_evolution()
    comprehensive_best = comprehensive_population[0] if comprehensive_population else None

    # Compare results
    print("\n" + "-" * 40)
    print("COMPARISON RESULTS")
    print("-" * 40)

    if baseline_best:
        baseline_improvement = (baseline_best.wanted_modified_prob - baseline_best.wanted_baseline_prob) / (1.0 - baseline_best.wanted_baseline_prob) * 100
        print(f"Baseline (no comprehensive search):")
        print(f"  Best tokens: {baseline_best.tokens}")
        print(f"  Improvement: {baseline_improvement:.2f}%")

    if comprehensive_best:
        comprehensive_improvement = (comprehensive_best.wanted_modified_prob - comprehensive_best.wanted_baseline_prob) / (1.0 - comprehensive_best.wanted_baseline_prob) * 100
        print(f"Comprehensive search:")
        print(f"  Best tokens: {comprehensive_best.tokens}")
        print(f"  Improvement: {comprehensive_improvement:.2f}%")

        if baseline_best:
            if comprehensive_improvement > baseline_improvement:
                print(f"✅ Comprehensive search improved results by {comprehensive_improvement - baseline_improvement:.2f}%")
            else:
                print("📊 Results similar - comprehensive search provides token insights")


def example_advanced_comprehensive_search():
    """Advanced example with custom parameters."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Advanced Comprehensive Search Configuration")
    print("=" * 60)

    ga = GeneticProbabilityReducer(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        base_text="I love",
        wanted_token="pizza"
    )

    # Advanced configuration
    ga.population_size = 30
    ga.max_generations = 25
    ga.baseline_seeding_ratio = 0.9  # 90% baseline-guided seeding
    ga.early_stopping_threshold = 0.95  # Stop at 95% improvement

    # Load model and comprehensive search
    ga.load_model()
    ga.load_glitch_tokens(
        ascii_only=True,
        include_normal_tokens=True,
        comprehensive_search=True
    )

    print("Running advanced configuration...")
    final_population = ga.run_evolution()

    # Show detailed results
    print(f"\nTop 3 Results:")
    for i, individual in enumerate(final_population[:3], 1):
        if ga.tokenizer:
            token_texts = [ga.tokenizer.decode([tid]) for tid in individual.tokens]
            print(f"{i}. Tokens: {individual.tokens} ({token_texts})")
        else:
            print(f"{i}. Tokens: {individual.tokens}")

        improvement = (individual.wanted_modified_prob - individual.wanted_baseline_prob) / (1.0 - individual.wanted_baseline_prob) * 100
        print(f"   Fitness: {individual.fitness:.6f}")
        print(f"   Wanted token improvement: {improvement:.2f}%")
        print(f"   Probability: {individual.wanted_baseline_prob:.6f} → {individual.wanted_modified_prob:.6f}")
        print()


def main():
    """Run all comprehensive search examples."""
    print("🔍 COMPREHENSIVE WANTED TOKEN SEARCH EXAMPLES")
    print("=" * 80)
    print("This script demonstrates the new comprehensive search feature")
    print("for wanted token optimization before genetic algorithm evolution.")
    print()

    try:
        # Run examples
        example_basic_comprehensive_search()
        example_comprehensive_search_comparison()
        example_advanced_comprehensive_search()

        # GUI example (optional)
        response = input("\nRun GUI example? (y/n): ").lower().strip()
        if response == 'y':
            example_comprehensive_search_with_gui()

        print("\n" + "=" * 80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nKey takeaways:")
        print("• Use --comprehensive-search flag for thorough vocabulary exploration")
        print("• Combine with --ascii-only for cleaner, faster results")
        print("• Comprehensive search automatically prioritizes high-impact tokens")
        print("• Results feed directly into genetic algorithm for optimal evolution")
        print("• Use --gui flag to watch comprehensive search progress in real-time")

    except KeyboardInterrupt:
        print("\n⚠️ Examples interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
