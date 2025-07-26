#!/usr/bin/env python3
"""
Test script to compare improved genetic algorithm vs old behavior.

This script demonstrates how the improved crossover, diversity preservation,
and mutation strategies help avoid local optima with lower mutation rates.
"""

import json
import time
import argparse
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from glitcher.genetic.reducer import GeneticProbabilityReducer


def run_genetic_test(model_name: str, base_text: str, token_file: str,
                    mutation_rate: float, generations: int = 100,
                    population_size: int = 30, max_tokens: int = 4) -> Dict[str, Any]:
    """
    Run a genetic algorithm test with specified parameters.

    Args:
        model_name: Model to test
        base_text: Base text for probability reduction
        token_file: File containing glitch tokens
        mutation_rate: Mutation rate to test
        generations: Number of generations
        population_size: Population size
        max_tokens: Max tokens per individual

    Returns:
        Dictionary with test results
    """
    print(f"\n🧬 Testing with mutation rate: {mutation_rate}")

    # Initialize genetic algorithm
    reducer = GeneticProbabilityReducer(model_name, base_text)
    reducer.population_size = population_size
    reducer.max_generations = generations
    reducer.mutation_rate = mutation_rate
    reducer.max_tokens_per_individual = max_tokens
    reducer.early_stopping_threshold = 0.95  # Stop at 95% for faster testing

    # Load model and tokens
    reducer.load_model()
    reducer.load_glitch_tokens(token_file, ascii_only=True)

    start_time = time.time()

    try:
        # Run evolution
        final_population = reducer.run_evolution()

        # Get results
        best_individual = max(final_population, key=lambda x: x.fitness)
        best_fitness = best_individual.fitness
        reduction_pct = (best_fitness / reducer.baseline_probability) * 100 if reducer.baseline_probability > 0 else 0

        # Calculate final diversity
        final_diversity = reducer.calculate_population_diversity(final_population)

        elapsed_time = time.time() - start_time

        return {
            'mutation_rate': mutation_rate,
            'best_fitness': best_fitness,
            'reduction_percentage': reduction_pct,
            'best_tokens': best_individual.tokens,
            'best_token_texts': [reducer.tokenizer.decode([tid]).strip() for tid in best_individual.tokens],
            'final_diversity_ratio': final_diversity['diversity_ratio'],
            'unique_individuals': final_diversity['unique_individuals'],
            'elapsed_time': elapsed_time,
            'baseline_probability': reducer.baseline_probability,
            'converged': reduction_pct >= 95.0,
            'population_size': population_size
        }

    except Exception as e:
        print(f"❌ Test failed with mutation rate {mutation_rate}: {e}")
        return {
            'mutation_rate': mutation_rate,
            'error': str(e),
            'elapsed_time': time.time() - start_time
        }


def plot_comparison_results(results: List[Dict[str, Any]], output_file: str = "genetic_comparison.png"):
    """Plot comparison results."""

    # Filter successful results
    successful_results = [r for r in results if 'error' not in r]

    if not successful_results:
        print("❌ No successful results to plot")
        return

    mutation_rates = [r['mutation_rate'] for r in successful_results]
    reduction_percentages = [r['reduction_percentage'] for r in successful_results]
    diversity_ratios = [r['final_diversity_ratio'] for r in successful_results]
    elapsed_times = [r['elapsed_time'] for r in successful_results]
    converged = [r['converged'] for r in successful_results]

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Improved Genetic Algorithm Performance Comparison', fontsize=16)

    # Plot 1: Reduction Percentage vs Mutation Rate
    colors = ['green' if c else 'red' for c in converged]
    ax1.scatter(mutation_rates, reduction_percentages, c=colors, s=100, alpha=0.7)
    ax1.axhline(y=95, color='orange', linestyle='--', label='95% Target')
    ax1.set_xlabel('Mutation Rate')
    ax1.set_ylabel('Probability Reduction %')
    ax1.set_title('Reduction Performance vs Mutation Rate')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add text annotations for convergence
    for i, (mr, rp, conv) in enumerate(zip(mutation_rates, reduction_percentages, converged)):
        marker = "✓" if conv else "✗"
        ax1.annotate(f'{marker} {rp:.1f}%', (mr, rp), xytext=(5, 5),
                    textcoords='offset points', fontsize=9)

    # Plot 2: Diversity Ratio vs Mutation Rate
    ax2.scatter(mutation_rates, diversity_ratios, c=colors, s=100, alpha=0.7)
    ax2.set_xlabel('Mutation Rate')
    ax2.set_ylabel('Final Diversity Ratio')
    ax2.set_title('Population Diversity vs Mutation Rate')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Convergence Time vs Mutation Rate
    ax3.scatter(mutation_rates, elapsed_times, c=colors, s=100, alpha=0.7)
    ax3.set_xlabel('Mutation Rate')
    ax3.set_ylabel('Time to Completion (seconds)')
    ax3.set_title('Convergence Time vs Mutation Rate')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Success Rate Bar Chart
    mutation_rate_bins = [0.1, 0.3, 0.5, 0.8]
    success_rates = []

    for mr_bin in mutation_rate_bins:
        bin_results = [r for r in successful_results if abs(r['mutation_rate'] - mr_bin) < 0.05]
        if bin_results:
            success_rate = sum(1 for r in bin_results if r['converged']) / len(bin_results)
            success_rates.append(success_rate * 100)
        else:
            success_rates.append(0)

    bars = ax4.bar([str(mr) for mr in mutation_rate_bins], success_rates,
                   color=['green' if sr >= 50 else 'orange' if sr >= 25 else 'red' for sr in success_rates],
                   alpha=0.7)
    ax4.set_xlabel('Mutation Rate')
    ax4.set_ylabel('Success Rate %')
    ax4.set_title('Success Rate by Mutation Rate')
    ax4.set_ylim(0, 100)

    # Add value labels on bars
    for bar, sr in zip(bars, success_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{sr:.0f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plot saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Test improved genetic algorithm performance")
    parser.add_argument("model_name", help="HuggingFace model name")
    parser.add_argument("--token-file", default="email_extraction_all_glitches.json",
                       help="Glitch tokens file")
    parser.add_argument("--base-text", default="The quick brown",
                       help="Base text for testing")
    parser.add_argument("--generations", type=int, default=50,
                       help="Number of generations")
    parser.add_argument("--population-size", type=int, default=25,
                       help="Population size")
    parser.add_argument("--max-tokens", type=int, default=4,
                       help="Max tokens per individual")
    parser.add_argument("--output", default="genetic_test_results.json",
                       help="Output file for results")
    parser.add_argument("--plot", action="store_true",
                       help="Generate comparison plots")

    args = parser.parse_args()

    print("🧬 Testing Improved Genetic Algorithm Performance")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Base text: '{args.base_text}'")
    print(f"Generations: {args.generations}")
    print(f"Population size: {args.population_size}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Token file: {args.token_file}")

    # Test different mutation rates
    mutation_rates = [0.1, 0.2, 0.3, 0.5, 0.8]
    results = []

    for mutation_rate in mutation_rates:
        result = run_genetic_test(
            args.model_name,
            args.base_text,
            args.token_file,
            mutation_rate,
            args.generations,
            args.population_size,
            args.max_tokens
        )
        results.append(result)

        # Print immediate results
        if 'error' not in result:
            status = "✅ CONVERGED" if result['converged'] else "⚠️  PARTIAL"
            print(f"{status} | Mutation: {mutation_rate:.1f} | "
                  f"Reduction: {result['reduction_percentage']:.1f}% | "
                  f"Diversity: {result['final_diversity_ratio']:.3f} | "
                  f"Time: {result['elapsed_time']:.1f}s")
        else:
            print(f"❌ FAILED | Mutation: {mutation_rate:.1f} | Error: {result['error']}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {args.output}")

    # Generate summary
    successful_results = [r for r in results if 'error' not in r]
    converged_results = [r for r in successful_results if r['converged']]

    print("\n📊 SUMMARY")
    print("=" * 30)
    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Converged (≥95%): {len(converged_results)}")

    if successful_results:
        best_result = max(successful_results, key=lambda x: x['reduction_percentage'])
        print(f"\n🏆 BEST RESULT:")
        print(f"Mutation rate: {best_result['mutation_rate']}")
        print(f"Reduction: {best_result['reduction_percentage']:.2f}%")
        print(f"Tokens: {best_result['best_token_texts']}")
        print(f"Diversity: {best_result['final_diversity_ratio']:.3f}")
        print(f"Time: {best_result['elapsed_time']:.1f}s")

        # Show which low mutation rates worked
        low_mutation_successes = [r for r in converged_results if r['mutation_rate'] <= 0.3]
        if low_mutation_successes:
            print(f"\n🎯 LOW MUTATION SUCCESSES ({len(low_mutation_successes)}):")
            for result in low_mutation_successes:
                print(f"  Rate {result['mutation_rate']:.1f}: {result['reduction_percentage']:.1f}% reduction")
        else:
            print("\n⚠️  No low mutation rates (≤0.3) achieved convergence")
            print("Consider further algorithm improvements")

    # Generate plots if requested
    if args.plot and successful_results:
        plot_comparison_results(results, "genetic_algorithm_comparison.png")


if __name__ == "__main__":
    main()
