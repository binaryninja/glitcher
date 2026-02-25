#!/usr/bin/env python3
"""
Comprehensive Genetic Algorithm Performance Test

This script compares different genetic algorithm approaches to determine
the most effective strategy for breeding glitch token combinations.

Compares:
1. Standard mutation rates (0.1, 0.3, 0.5, 0.8)
2. Adaptive mutation (starts high, decreases over time)
3. Different population sizes
4. Impact of improved crossover and diversity preservation

Usage:
    python comprehensive_genetic_test.py meta-llama/Llama-3.2-1B-Instruct
"""

import json
import time
import argparse
import statistics
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add the parent directory to sys.path to import glitcher modules
sys.path.insert(0, str(Path(__file__).parent))

from glitcher.genetic.reducer import GeneticProbabilityReducer


class ComprehensiveGeneticTester:
    """Comprehensive tester for genetic algorithm performance."""

    def __init__(self, model_name: str, token_file: str = "email_extraction_all_glitches.json"):
        self.model_name = model_name
        self.token_file = token_file
        self.results = []

    def run_test_scenario(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single test scenario with given configuration.

        Args:
            name: Name of the test scenario
            config: Configuration parameters

        Returns:
            Test results dictionary
        """
        print(f"\n🧬 Running scenario: {name}")
        print("─" * 50)
        for key, value in config.items():
            print(f"  {key}: {value}")

        start_time = time.time()

        try:
            # Create genetic algorithm instance
            ga = GeneticProbabilityReducer(
                model_name=self.model_name,
                base_text=config.get('base_text', 'The quick brown'),
                target_token=config.get('target_token')
            )

            # Configure parameters
            ga.population_size = config.get('population_size', 50)
            ga.max_generations = config.get('generations', 1000)
            ga.mutation_rate = config.get('mutation_rate', 0.1)
            ga.crossover_rate = config.get('crossover_rate', 0.7)
            ga.elite_size = config.get('elite_size', 2)  # Use improved default
            ga.max_tokens_per_individual = config.get('max_tokens', 4)
            ga.early_stopping_threshold = config.get('early_stopping_threshold', 0.99)

            # Configure adaptive mutation if specified
            if config.get('adaptive_mutation', False):
                ga.adaptive_mutation = True
                ga.initial_mutation_rate = config.get('initial_mutation_rate', 0.8)
                ga.final_mutation_rate = config.get('final_mutation_rate', 0.1)
                ga.current_mutation_rate = ga.initial_mutation_rate

            # Load model and tokens
            ga.load_model()
            ga.load_glitch_tokens(self.token_file, ascii_only=True)

            # Track initial state
            initial_diversity = ga.calculate_population_diversity(ga.create_initial_population())
            baseline_prob = ga.baseline_probability

            # Run evolution
            final_population = ga.run_evolution()

            # Calculate results - ensure all individuals are properly evaluated
            valid_individuals = [ind for ind in final_population if hasattr(ind, 'fitness') and ind.fitness > -1.0]

            if not valid_individuals:
                print(f"⚠️ WARNING: No valid individuals found in final population!")
                best_fitness = 0.0
                reduction_pct = 0.0
                best_individual = final_population[0] if final_population else None
            else:
                best_individual = max(valid_individuals, key=lambda x: x.fitness)
                best_fitness = best_individual.fitness

                # Debug logging
                if hasattr(best_individual, 'modified_prob') and hasattr(best_individual, 'baseline_prob'):
                    modified_prob = best_individual.modified_prob
                    baseline_prob = best_individual.baseline_prob
                    actual_reduction_pct = ((baseline_prob - modified_prob) / baseline_prob) * 100
                    print(f"🔍 Debug: baseline={baseline_prob:.4f}, modified={modified_prob:.4f}, fitness={best_fitness:.4f}, reduction={actual_reduction_pct:.1f}%")
                    reduction_pct = actual_reduction_pct
                else:
                    # Fallback calculation if individual properties not available
                    reduction_pct = (best_fitness / baseline_prob) * 100 if baseline_prob > 0 else 0

            # Calculate final diversity
            final_diversity = ga.calculate_population_diversity(final_population)

            elapsed_time = time.time() - start_time

            # Success criteria: 95% probability reduction
            success = reduction_pct >= 95.0

            result = {
                'scenario_name': name,
                'config': config.copy(),
                'success': success,
                'best_fitness': best_fitness,
                'reduction_percentage': reduction_pct,
                'baseline_probability': baseline_prob,
                'best_tokens': best_individual.tokens if best_individual else [],
                'best_token_texts': [ga.tokenizer.decode([tid]).strip() for tid in best_individual.tokens] if best_individual else [],
                'final_diversity_ratio': final_diversity['diversity_ratio'],
                'final_unique_individuals': final_diversity['unique_individuals'],
                'initial_diversity_ratio': initial_diversity['diversity_ratio'],
                'elapsed_time': elapsed_time,
                'converged_early': reduction_pct >= (config.get('early_stopping_threshold', 0.99) * 100),
                'population_size': ga.population_size,
                'generations_run': ga.max_generations,
                'valid_individuals_count': len(valid_individuals),
                'total_individuals_count': len(final_population)
            }

            # Status indicator
            status = "✅ SUCCESS" if success else "⚠️  PARTIAL" if reduction_pct >= 70 else "❌ FAILED"
            print(f"{status} | Reduction: {reduction_pct:.1f}% | Time: {elapsed_time:.1f}s | Diversity: {final_diversity['diversity_ratio']:.3f}")

            return result

        except Exception as e:
            elapsed_time = time.time() - start_time
            error_result = {
                'scenario_name': name,
                'config': config.copy(),
                'error': str(e),
                'elapsed_time': elapsed_time,
                'success': False
            }
            print(f"❌ FAILED | Error: {e}")
            return error_result

    def run_comprehensive_test(self) -> List[Dict[str, Any]]:
        """Run comprehensive test suite."""

        print("🧬 COMPREHENSIVE GENETIC ALGORITHM PERFORMANCE TEST")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Token file: {self.token_file}")
        print(f"Generations: 1000 (comprehensive test)")
        print()

        test_scenarios = [
            # Standard mutation rate comparisons
            {
                'name': 'Standard_Mutation_0.1',
                'config': {
                    'mutation_rate': 0.1,
                    'generations': 1000,
                    'population_size': 40,
                    'max_tokens': 4,
                    'adaptive_mutation': False,
                    'early_stopping_threshold': 0.95
                }
            },
            {
                'name': 'Standard_Mutation_0.3',
                'config': {
                    'mutation_rate': 0.3,
                    'generations': 100,
                    'population_size': 40,
                    'max_tokens': 4,
                    'adaptive_mutation': False,
                    'early_stopping_threshold': 0.95
                }
            },
            {
                'name': 'Standard_Mutation_0.5',
                'config': {
                    'mutation_rate': 0.5,
                    'generations': 100,
                    'population_size': 40,
                    'max_tokens': 4,
                    'adaptive_mutation': False,
                    'early_stopping_threshold': 0.95
                }
            },
            {
                'name': 'Standard_Mutation_0.8',
                'config': {
                    'mutation_rate': 0.8,
                    'generations': 100,
                    'population_size': 40,
                    'max_tokens': 4,
                    'adaptive_mutation': False,
                    'early_stopping_threshold': 0.95
                }
            },

            # Adaptive mutation scenarios
            {
                'name': 'Adaptive_Mutation_Standard',
                'config': {
                    'adaptive_mutation': True,
                    'initial_mutation_rate': 0.8,
                    'final_mutation_rate': 0.1,
                    'generations': 100,
                    'population_size': 40,
                    'max_tokens': 4,
                    'early_stopping_threshold': 0.95
                }
            },
            {
                'name': 'Adaptive_Mutation_Aggressive',
                'config': {
                    'adaptive_mutation': True,
                    'initial_mutation_rate': 0.9,
                    'final_mutation_rate': 0.05,
                    'generations': 100,
                    'population_size': 40,
                    'max_tokens': 4,
                    'early_stopping_threshold': 0.95
                }
            },
            {
                'name': 'Adaptive_Mutation_Conservative',
                'config': {
                    'adaptive_mutation': True,
                    'initial_mutation_rate': 0.6,
                    'final_mutation_rate': 0.2,
                    'generations': 100,
                    'population_size': 40,
                    'max_tokens': 4,
                    'early_stopping_threshold': 0.95
                }
            },

            # Population size variations with best mutation strategy
            {
                'name': 'Large_Population_Adaptive',
                'config': {
                    'adaptive_mutation': True,
                    'initial_mutation_rate': 0.8,
                    'final_mutation_rate': 0.1,
                    'generations': 100,
                    'population_size': 60,
                    'max_tokens': 4,
                    'early_stopping_threshold': 0.95
                }
            },
            {
                'name': 'Small_Population_Adaptive',
                'config': {
                    'adaptive_mutation': True,
                    'initial_mutation_rate': 0.8,
                    'final_mutation_rate': 0.1,
                    'generations': 100,
                    'population_size': 25,
                    'max_tokens': 4,
                    'early_stopping_threshold': 0.95
                }
            },

            # Token count variations
            {
                'name': 'Max_Tokens_6_Adaptive',
                'config': {
                    'adaptive_mutation': True,
                    'initial_mutation_rate': 0.8,
                    'final_mutation_rate': 0.1,
                    'generations': 100,
                    'population_size': 40,
                    'max_tokens': 6,
                    'early_stopping_threshold': 0.95
                }
            },
            {
                'name': 'Max_Tokens_2_Adaptive',
                'config': {
                    'adaptive_mutation': True,
                    'initial_mutation_rate': 0.8,
                    'final_mutation_rate': 0.1,
                    'generations': 100,
                    'population_size': 40,
                    'max_tokens': 2,
                    'early_stopping_threshold': 0.95
                }
            }
        ]

        # Run all scenarios
        results = []
        for scenario in test_scenarios:
            result = self.run_test_scenario(scenario['name'], scenario['config'])
            results.append(result)
            self.results.append(result)

        return results

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze and summarize test results."""

        successful_results = [r for r in results if r.get('success', False)]
        partial_results = [r for r in results if not r.get('success', False) and r.get('reduction_percentage', 0) >= 70]
        failed_results = [r for r in results if r.get('reduction_percentage', 0) < 70]

        analysis = {
            'total_scenarios': len(results),
            'successful_scenarios': len(successful_results),
            'partial_scenarios': len(partial_results),
            'failed_scenarios': len(failed_results),
            'success_rate': len(successful_results) / len(results) * 100 if results else 0
        }

        # Find best performing scenarios
        if successful_results:
            best_result = max(successful_results, key=lambda x: x['reduction_percentage'])
            analysis['best_scenario'] = best_result

        # Analyze by mutation strategy
        adaptive_results = [r for r in results if r.get('config', {}).get('adaptive_mutation', False)]
        standard_results = [r for r in results if not r.get('config', {}).get('adaptive_mutation', False)]

        if adaptive_results:
            adaptive_success_rate = len([r for r in adaptive_results if r.get('success', False)]) / len(adaptive_results) * 100
            analysis['adaptive_mutation_success_rate'] = adaptive_success_rate

        if standard_results:
            standard_success_rate = len([r for r in standard_results if r.get('success', False)]) / len(standard_results) * 100
            analysis['standard_mutation_success_rate'] = standard_success_rate

        # Performance by mutation rate
        mutation_rates = {}
        for result in standard_results:
            rate = result.get('config', {}).get('mutation_rate', 0)
            if rate not in mutation_rates:
                mutation_rates[rate] = []
            mutation_rates[rate].append(result)

        analysis['mutation_rate_performance'] = {}
        for rate, rate_results in mutation_rates.items():
            successes = len([r for r in rate_results if r.get('success', False)])
            avg_reduction = statistics.mean([r.get('reduction_percentage', 0) for r in rate_results])
            analysis['mutation_rate_performance'][rate] = {
                'success_count': successes,
                'total_count': len(rate_results),
                'success_rate': successes / len(rate_results) * 100,
                'avg_reduction': avg_reduction
            }

        return analysis

    def generate_visualization(self, results: List[Dict[str, Any]], output_file: str = "comprehensive_genetic_test_results.png"):
        """Generate comprehensive visualization of results."""

        # Filter successful results for plotting
        plot_results = [r for r in results if 'reduction_percentage' in r]

        if not plot_results:
            print("❌ No results to plot")
            return

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Genetic Algorithm Performance Analysis\n100 Generations Test', fontsize=16, fontweight='bold')

        # Plot 1: Reduction Performance by Scenario
        scenario_names = [r['scenario_name'] for r in plot_results]
        reduction_percentages = [r['reduction_percentage'] for r in plot_results]
        success_colors = ['green' if r.get('success', False) else 'orange' if r['reduction_percentage'] >= 70 else 'red' for r in plot_results]

        bars1 = ax1.bar(range(len(scenario_names)), reduction_percentages, color=success_colors, alpha=0.7)
        ax1.axhline(y=95, color='red', linestyle='--', label='95% Target')
        ax1.axhline(y=70, color='orange', linestyle='--', label='70% Partial')
        ax1.set_ylabel('Probability Reduction %')
        ax1.set_title('Reduction Performance by Scenario')
        ax1.set_xticks(range(len(scenario_names)))
        ax1.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, pct in zip(bars1, reduction_percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

        # Plot 2: Mutation Strategy Comparison
        adaptive_results = [r for r in plot_results if r.get('config', {}).get('adaptive_mutation', False)]
        standard_results = [r for r in plot_results if not r.get('config', {}).get('adaptive_mutation', False)]

        strategy_data = []
        strategy_labels = []

        if adaptive_results:
            adaptive_reductions = [r['reduction_percentage'] for r in adaptive_results]
            strategy_data.append(adaptive_reductions)
            strategy_labels.append(f'Adaptive\n(n={len(adaptive_results)})')

        if standard_results:
            standard_reductions = [r['reduction_percentage'] for r in standard_results]
            strategy_data.append(standard_reductions)
            strategy_labels.append(f'Standard\n(n={len(standard_results)})')

        if strategy_data:
            box_plot = ax2.boxplot(strategy_data, labels=strategy_labels, patch_artist=True)
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
                patch.set_facecolor(color)

            ax2.axhline(y=95, color='red', linestyle='--', label='95% Target')
            ax2.set_ylabel('Probability Reduction %')
            ax2.set_title('Mutation Strategy Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Plot 3: Execution Time vs Performance
        times = [r['elapsed_time'] for r in plot_results]
        reductions = [r['reduction_percentage'] for r in plot_results]

        scatter = ax3.scatter(times, reductions, c=success_colors, s=100, alpha=0.7)
        ax3.axhline(y=95, color='red', linestyle='--', label='95% Target')
        ax3.set_xlabel('Execution Time (seconds)')
        ax3.set_ylabel('Probability Reduction %')
        ax3.set_title('Performance vs Execution Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Diversity vs Performance
        diversity_ratios = [r.get('final_diversity_ratio', 0) for r in plot_results]

        scatter2 = ax4.scatter(diversity_ratios, reductions, c=success_colors, s=100, alpha=0.7)
        ax4.axhline(y=95, color='red', linestyle='--', label='95% Target')
        ax4.set_xlabel('Final Population Diversity Ratio')
        ax4.set_ylabel('Probability Reduction %')
        ax4.set_title('Performance vs Population Diversity')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Comprehensive visualization saved to: {output_file}")

    def print_summary(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Print comprehensive summary of results."""

        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
        print("=" * 80)

        print(f"Total scenarios tested: {analysis['total_scenarios']}")
        print(f"Successful (≥95%): {analysis['successful_scenarios']} ({analysis['success_rate']:.1f}%)")
        print(f"Partial (70-94%): {analysis['partial_scenarios']}")
        print(f"Failed (<70%): {analysis['failed_scenarios']}")

        # Best performing scenario
        if 'best_scenario' in analysis:
            best = analysis['best_scenario']
            print(f"\n🏆 BEST PERFORMING SCENARIO:")
            print(f"Name: {best['scenario_name']}")
            print(f"Reduction: {best['reduction_percentage']:.2f}%")
            print(f"Time: {best['elapsed_time']:.1f}s")
            print(f"Tokens: {best['best_token_texts']}")
            print(f"Config: {best['config']}")

        # Mutation strategy comparison
        print(f"\n🧬 MUTATION STRATEGY ANALYSIS:")
        if 'adaptive_mutation_success_rate' in analysis:
            print(f"Adaptive mutation success rate: {analysis['adaptive_mutation_success_rate']:.1f}%")
        if 'standard_mutation_success_rate' in analysis:
            print(f"Standard mutation success rate: {analysis['standard_mutation_success_rate']:.1f}%")

        # Standard mutation rates performance
        if 'mutation_rate_performance' in analysis:
            print(f"\n📈 STANDARD MUTATION RATE PERFORMANCE:")
            for rate, perf in analysis['mutation_rate_performance'].items():
                print(f"Rate {rate:.1f}: {perf['success_count']}/{perf['total_count']} success "
                      f"({perf['success_rate']:.1f}%), avg reduction: {perf['avg_reduction']:.1f}%")

        # Top 5 scenarios by performance
        successful_results = [r for r in results if 'reduction_percentage' in r]
        successful_results.sort(key=lambda x: x['reduction_percentage'], reverse=True)

        print(f"\n🎯 TOP 5 SCENARIOS BY PERFORMANCE:")
        for i, result in enumerate(successful_results[:5], 1):
            status = "✅" if result.get('success', False) else "⚠️"
            print(f"{i}. {status} {result['scenario_name']}: {result['reduction_percentage']:.1f}% "
                  f"({result['elapsed_time']:.1f}s)")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")

        if analysis.get('adaptive_mutation_success_rate', 0) > analysis.get('standard_mutation_success_rate', 0):
            print("✅ Use adaptive mutation for best performance")
            print("   - Starts with high exploration (0.8-0.9)")
            print("   - Gradually reduces to exploitation (0.1)")
            print("   - Balances exploration and convergence")
        else:
            best_standard_rate = max(analysis.get('mutation_rate_performance', {}).items(),
                                   key=lambda x: x[1]['success_rate'], default=(None, {}))[0]
            if best_standard_rate:
                print(f"✅ Use standard mutation rate: {best_standard_rate}")

        # Find best population size and max tokens from successful results
        successful_configs = [r['config'] for r in successful_results[:3]]
        if successful_configs:
            pop_sizes = [c.get('population_size', 40) for c in successful_configs]
            max_tokens = [c.get('max_tokens', 4) for c in successful_configs]
            avg_pop_size = statistics.mean(pop_sizes)
            avg_max_tokens = statistics.mean(max_tokens)
            print(f"✅ Recommended population size: ~{avg_pop_size:.0f}")
            print(f"✅ Recommended max tokens: ~{avg_max_tokens:.0f}")

        print("\n" + "=" * 80)

    def save_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any],
                    output_file: str = "comprehensive_genetic_test_results.json"):
        """Save detailed results to JSON file."""

        output_data = {
            'test_metadata': {
                'model_name': self.model_name,
                'token_file': self.token_file,
                'test_type': 'comprehensive_genetic_algorithm_performance',
                'generations_per_test': 100,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'summary_analysis': analysis,
            'detailed_results': results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"💾 Detailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive genetic algorithm performance test")
    parser.add_argument("model_name", help="HuggingFace model name")
    parser.add_argument("--token-file", default="email_extraction_all_glitches.json",
                       help="Glitch tokens file")
    parser.add_argument("--output-json", default="comprehensive_genetic_test_results.json",
                       help="Output JSON file")
    parser.add_argument("--output-plot", default="comprehensive_genetic_test_results.png",
                       help="Output plot file")
    parser.add_argument("--no-plot", action="store_true",
                       help="Skip generating plots")

    args = parser.parse_args()

    # Create tester
    tester = ComprehensiveGeneticTester(args.model_name, args.token_file)

    # Run comprehensive test
    results = tester.run_comprehensive_test()

    # Analyze results
    analysis = tester.analyze_results(results)

    # Generate visualizations
    if not args.no_plot:
        tester.generate_visualization(results, args.output_plot)

    # Print summary
    tester.print_summary(results, analysis)

    # Save detailed results
    tester.save_results(results, analysis, args.output_json)


if __name__ == "__main__":
    main()
