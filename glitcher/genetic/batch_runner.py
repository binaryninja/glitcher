#!/usr/bin/env python3
"""
Batch Genetic Algorithm Runner for Probability Reducer Token Combinations

This script runs the genetic algorithm across multiple base texts and scenarios
to discover patterns in effective probability-reducing token combinations.

Author: Claude
Date: 2024
"""

import json
import argparse
import logging
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

import pandas as pd
from .reducer import GeneticProbabilityReducer


class GeneticBatchRunner:
    """
    Batch runner for genetic algorithm experiments across multiple scenarios.

    批量运行遗传算法实验的多场景运行器。
    """

    def __init__(self, model_name: str, token_file: str, ascii_only: bool = False):
        """
        Initialize the batch runner.

        Args:
            model_name: HuggingFace model identifier
            token_file: Path to glitch tokens JSON file
            ascii_only: If True, filter to only include tokens with ASCII-only decoded text
        """
        self.model_name = model_name
        self.token_file = token_file
        self.ascii_only = ascii_only

        # Experiment scenarios
        self.scenarios = []

        # Results storage
        self.experiment_results = []

        # Analysis results
        self.pattern_analysis = {}

        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def add_scenario(self, name: str, base_text: str, target_token: str = None,
                    ga_params: Dict[str, Any] = None):
        """
        Add an experiment scenario.

        Args:
            name: Scenario name
            base_text: Base text for the scenario
            target_token: Target token (auto-detected if None)
            ga_params: Custom GA parameters for this scenario
        """
        scenario = {
            'name': name,
            'base_text': base_text,
            'target_token': target_token,
            'ga_params': ga_params or {}
        }
        self.scenarios.append(scenario)
        self.logger.info(f"Added scenario: {name}")

    def add_predefined_scenarios(self):
        """Add a set of predefined interesting scenarios."""
        scenarios = [
            {
                'name': 'strong_prediction_fox',
                'base_text': 'The quick brown',
                'description': 'Strong prediction case (fox ~95%)'
            },
            {
                'name': 'weak_prediction_weather',
                'base_text': 'The weather is',
                'description': 'Weak prediction case'
            },
            {
                'name': 'weak_prediction_think',
                'base_text': 'I think that',
                'description': 'Another weak prediction case'
            },
            {
                'name': 'coding_context',
                'base_text': 'def function_name(',
                'description': 'Code completion context'
            },
            {
                'name': 'conversation_start',
                'base_text': 'Hello, how are',
                'description': 'Conversational context'
            },
            {
                'name': 'question_start',
                'base_text': 'What is the',
                'description': 'Question context'
            },
            {
                'name': 'math_context',
                'base_text': '2 + 2 =',
                'description': 'Mathematical context'
            },
            {
                'name': 'incomplete_sentence',
                'base_text': 'The cat sat on the',
                'description': 'Incomplete common phrase'
            }
        ]

        for scenario in scenarios:
            self.add_scenario(
                name=scenario['name'],
                base_text=scenario['base_text']
            )

    def run_experiment(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run genetic algorithm for a single scenario.

        Args:
            scenario: Scenario configuration

        Returns:
            Experiment results
        """
        self.logger.info(f"Running experiment: {scenario['name']}")

        # Create analyzer
        analyzer = GeneticProbabilityReducer(
            model_name=self.model_name,
            base_text=scenario['base_text'],
            target_token=scenario['target_token']
        )

        # Apply custom GA parameters
        ga_params = scenario.get('ga_params', {})
        for param, value in ga_params.items():
            if hasattr(analyzer, param):
                setattr(analyzer, param, value)

        try:
            # Load model and tokens
            analyzer.load_model()
            analyzer.load_glitch_tokens(self.token_file, ascii_only=self.ascii_only)

            # Run evolution
            final_population = analyzer.run_evolution()

            # Extract results
            top_results = []
            for individual in final_population[:20]:  # Top 20
                if individual.fitness > 0:
                    token_texts = [analyzer.tokenizer.decode([tid]) for tid in individual.tokens]
                    top_results.append({
                        'tokens': individual.tokens,
                        'token_texts': token_texts,
                        'fitness': individual.fitness,
                        'baseline_prob': individual.baseline_prob,
                        'modified_prob': individual.modified_prob,
                        'reduction_percentage': (individual.fitness / individual.baseline_prob) * 100
                    })

            result = {
                'scenario_name': scenario['name'],
                'base_text': scenario['base_text'],
                'target_token_id': analyzer.target_token_id,
                'target_token_text': analyzer.tokenizer.decode([analyzer.target_token_id]),
                'baseline_probability': analyzer.baseline_probability,
                'top_results': top_results,
                'total_valid_results': len([r for r in final_population if r.fitness > 0]),
                'best_fitness': max(individual.fitness for individual in final_population),
                'ga_params': {
                    'population_size': analyzer.population_size,
                    'max_generations': analyzer.max_generations,
                    'mutation_rate': analyzer.mutation_rate,
                    'crossover_rate': analyzer.crossover_rate
                }
            }

            self.logger.info(f"Experiment completed: {scenario['name']}")
            self.logger.info(f"Best fitness: {result['best_fitness']:.4f}")

            return result

        except Exception as e:
            self.logger.error(f"Error in experiment {scenario['name']}: {e}")
            return {
                'scenario_name': scenario['name'],
                'error': str(e),
                'status': 'failed'
            }

    def run_all_experiments(self) -> List[Dict[str, Any]]:
        """
        Run genetic algorithm for all scenarios.

        Returns:
            List of experiment results
        """
        self.logger.info(f"Starting batch run with {len(self.scenarios)} scenarios")

        results = []
        for i, scenario in enumerate(self.scenarios):
            self.logger.info(f"Progress: {i+1}/{len(self.scenarios)}")
            result = self.run_experiment(scenario)
            results.append(result)

        self.experiment_results = results
        self.logger.info("All experiments completed")
        return results

    def analyze_token_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in successful token combinations across experiments.

        Returns:
            Pattern analysis results
        """
        self.logger.info("Analyzing token patterns across experiments")

        # Collect all successful tokens
        all_tokens = []
        token_frequencies = Counter()
        combination_frequencies = Counter()
        token_effectiveness = defaultdict(list)

        for result in self.experiment_results:
            if 'error' in result:
                continue

            scenario_name = result['scenario_name']
            baseline_prob = result['baseline_probability']

            for top_result in result.get('top_results', []):
                tokens = tuple(top_result['tokens'])
                fitness = top_result['fitness']
                reduction_pct = top_result['reduction_percentage']

                all_tokens.extend(top_result['tokens'])
                token_frequencies.update(top_result['tokens'])
                combination_frequencies[tokens] += 1

                # Track effectiveness per token
                for token in top_result['tokens']:
                    token_effectiveness[token].append({
                        'scenario': scenario_name,
                        'fitness': fitness,
                        'reduction_pct': reduction_pct,
                        'baseline_prob': baseline_prob
                    })

        # Analyze most effective individual tokens
        top_tokens = []
        for token, occurrences in token_frequencies.most_common(20):
            effectiveness_data = token_effectiveness[token]
            avg_fitness = sum(d['fitness'] for d in effectiveness_data) / len(effectiveness_data)
            avg_reduction = sum(d['reduction_pct'] for d in effectiveness_data) / len(effectiveness_data)
            scenarios_count = len(set(d['scenario'] for d in effectiveness_data))

            top_tokens.append({
                'token_id': token,
                'frequency': occurrences,
                'avg_fitness': avg_fitness,
                'avg_reduction_pct': avg_reduction,
                'scenarios_effective': scenarios_count,
                'effectiveness_ratio': scenarios_count / len(self.scenarios)
            })

        # Analyze most effective combinations
        top_combinations = []
        for combination, frequency in combination_frequencies.most_common(15):
            # Find all instances of this combination
            combination_data = []
            for result in self.experiment_results:
                if 'error' in result:
                    continue
                for top_result in result.get('top_results', []):
                    if tuple(top_result['tokens']) == combination:
                        combination_data.append({
                            'scenario': result['scenario_name'],
                            'fitness': top_result['fitness'],
                            'reduction_pct': top_result['reduction_percentage']
                        })

            if combination_data:
                avg_fitness = sum(d['fitness'] for d in combination_data) / len(combination_data)
                avg_reduction = sum(d['reduction_pct'] for d in combination_data) / len(combination_data)
                scenarios_count = len(set(d['scenario'] for d in combination_data))

                top_combinations.append({
                    'tokens': list(combination),
                    'frequency': frequency,
                    'avg_fitness': avg_fitness,
                    'avg_reduction_pct': avg_reduction,
                    'scenarios_effective': scenarios_count,
                    'effectiveness_ratio': scenarios_count / len(self.scenarios)
                })

        # Analyze scenario difficulty
        scenario_analysis = []
        for result in self.experiment_results:
            if 'error' in result:
                continue

            valid_results = len(result.get('top_results', []))
            best_reduction = max((r['reduction_percentage'] for r in result.get('top_results', [])), default=0)
            avg_reduction = sum(r['reduction_percentage'] for r in result.get('top_results', [])) / max(valid_results, 1)

            scenario_analysis.append({
                'scenario': result['scenario_name'],
                'base_text': result['base_text'],
                'baseline_probability': result['baseline_probability'],
                'valid_solutions': valid_results,
                'best_reduction_pct': best_reduction,
                'avg_reduction_pct': avg_reduction,
                'difficulty_score': result['baseline_probability'] / max(best_reduction, 0.001)  # Higher = harder
            })

        analysis = {
            'total_experiments': len([r for r in self.experiment_results if 'error' not in r]),
            'total_successful_combinations': len(combination_frequencies),
            'unique_effective_tokens': len(token_frequencies),
            'top_individual_tokens': top_tokens,
            'top_token_combinations': top_combinations,
            'scenario_analysis': sorted(scenario_analysis, key=lambda x: x['difficulty_score']),
            'summary_stats': {
                'avg_valid_solutions_per_scenario': sum(len(r.get('top_results', [])) for r in self.experiment_results if 'error' not in r) / len([r for r in self.experiment_results if 'error' not in r]),
                'best_overall_reduction': max((max((tr['reduction_percentage'] for tr in r.get('top_results', [])), default=0) for r in self.experiment_results if 'error' not in r), default=0)
            }
        }

        self.pattern_analysis = analysis
        return analysis

    def generate_report(self) -> str:
        """
        Generate a comprehensive text report of the experiments.

        Returns:
            Formatted report string
        """
        if not self.pattern_analysis:
            self.analyze_token_patterns()

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("GENETIC ALGORITHM BATCH EXPERIMENT REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Model: {self.model_name}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Experiments: {self.pattern_analysis['total_experiments']}")
        report_lines.append("")

        # Summary Statistics
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-" * 40)
        stats = self.pattern_analysis['summary_stats']
        report_lines.append(f"Average valid solutions per scenario: {stats['avg_valid_solutions_per_scenario']:.1f}")
        report_lines.append(f"Best overall reduction achieved: {stats['best_overall_reduction']:.1f}%")
        report_lines.append(f"Unique effective tokens discovered: {self.pattern_analysis['unique_effective_tokens']}")
        report_lines.append(f"Total successful combinations: {self.pattern_analysis['total_successful_combinations']}")
        report_lines.append("")

        # Top Individual Tokens
        report_lines.append("TOP INDIVIDUAL TOKENS")
        report_lines.append("-" * 40)
        for i, token_data in enumerate(self.pattern_analysis['top_individual_tokens'][:10]):
            report_lines.append(f"{i+1:2d}. Token ID {token_data['token_id']}")
            report_lines.append(f"    Frequency: {token_data['frequency']}, Avg Reduction: {token_data['avg_reduction_pct']:.1f}%")
            report_lines.append(f"    Effective in {token_data['scenarios_effective']}/{len(self.scenarios)} scenarios ({token_data['effectiveness_ratio']:.1%})")
            report_lines.append("")

        # Top Token Combinations
        report_lines.append("TOP TOKEN COMBINATIONS")
        report_lines.append("-" * 40)
        for i, combo_data in enumerate(self.pattern_analysis['top_token_combinations'][:10]):
            report_lines.append(f"{i+1:2d}. Tokens: {combo_data['tokens']}")
            report_lines.append(f"    Frequency: {combo_data['frequency']}, Avg Reduction: {combo_data['avg_reduction_pct']:.1f}%")
            report_lines.append(f"    Effective in {combo_data['scenarios_effective']}/{len(self.scenarios)} scenarios ({combo_data['effectiveness_ratio']:.1%})")
            report_lines.append("")

        # Scenario Analysis
        report_lines.append("SCENARIO DIFFICULTY ANALYSIS")
        report_lines.append("-" * 40)
        for scenario in self.pattern_analysis['scenario_analysis']:
            report_lines.append(f"Scenario: {scenario['scenario']}")
            report_lines.append(f"  Base text: '{scenario['base_text']}'")
            report_lines.append(f"  Baseline probability: {scenario['baseline_probability']:.4f}")
            report_lines.append(f"  Valid solutions found: {scenario['valid_solutions']}")
            report_lines.append(f"  Best reduction: {scenario['best_reduction_pct']:.1f}%")
            report_lines.append(f"  Difficulty score: {scenario['difficulty_score']:.2f}")
            report_lines.append("")

        # Individual Experiment Results
        report_lines.append("INDIVIDUAL EXPERIMENT RESULTS")
        report_lines.append("-" * 40)
        for result in self.experiment_results:
            if 'error' in result:
                report_lines.append(f"FAILED: {result['scenario_name']} - {result['error']}")
                continue

            report_lines.append(f"Experiment: {result['scenario_name']}")
            report_lines.append(f"  Base text: '{result['base_text']}'")
            report_lines.append(f"  Target: '{result['target_token_text']}' (ID: {result['target_token_id']})")
            report_lines.append(f"  Baseline probability: {result['baseline_probability']:.4f}")
            report_lines.append(f"  Best fitness: {result['best_fitness']:.4f}")

            # Top 3 results for this experiment
            report_lines.append("  Top 3 results:")
            for i, top_result in enumerate(result.get('top_results', [])[:3]):
                report_lines.append(f"    {i+1}. Tokens: {top_result['tokens']}")
                report_lines.append(f"       Reduction: {top_result['reduction_percentage']:.1f}% ({top_result['baseline_prob']:.4f} → {top_result['modified_prob']:.4f})")
            report_lines.append("")

        return "\n".join(report_lines)

    def save_results(self, output_dir: str):
        """
        Save all results to files.

        Args:
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw experiment results
        results_file = output_path / f"genetic_batch_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'model_name': self.model_name,
                'timestamp': timestamp,
                'experiments': self.experiment_results,
                'pattern_analysis': self.pattern_analysis
            }, f, indent=2, ensure_ascii=False)

        # Save text report
        report_file = output_path / f"genetic_batch_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self.generate_report())

        # Save CSV summary for easy analysis
        csv_file = output_path / f"genetic_batch_summary_{timestamp}.csv"
        summary_data = []
        for result in self.experiment_results:
            if 'error' in result:
                continue
            for top_result in result.get('top_results', []):
                summary_data.append({
                    'scenario': result['scenario_name'],
                    'base_text': result['base_text'],
                    'baseline_prob': result['baseline_probability'],
                    'tokens': str(top_result['tokens']),
                    'fitness': top_result['fitness'],
                    'reduction_pct': top_result['reduction_percentage']
                })

        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(csv_file, index=False)

        self.logger.info(f"Results saved to {output_path}")
        self.logger.info(f"  Raw results: {results_file}")
        self.logger.info(f"  Text report: {report_file}")
        self.logger.info(f"  CSV summary: {csv_file}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Batch Genetic Algorithm Runner for Probability Reducers"
    )
    parser.add_argument(
        "model_name",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--token-file",
        default="email_llams321.json",
        help="JSON file containing glitch tokens"
    )
    parser.add_argument(
        "--scenarios-file",
        help="JSON file with custom scenarios"
    )
    parser.add_argument(
        "--output-dir",
        default="genetic_batch_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=30,
        help="Population size for genetic algorithm"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Maximum number of generations"
    )
    parser.add_argument(
        "--use-predefined",
        action="store_true",
        help="Use predefined scenario set"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3,
        help="Maximum tokens per individual combination (default: 3)"
    )
    parser.add_argument(
        "--ascii-only",
        action="store_true",
        help="Filter tokens to only include those with ASCII-only decoded text"
    )

    args = parser.parse_args()

    # Create batch runner
    runner = GeneticBatchRunner(args.model_name, args.token_file, ascii_only=args.ascii_only)

    # Add scenarios
    if args.scenarios_file:
        with open(args.scenarios_file, 'r', encoding='utf-8') as f:
            custom_scenarios = json.load(f)
        for scenario in custom_scenarios:
            runner.add_scenario(**scenario)
    elif args.use_predefined:
        runner.add_predefined_scenarios()
    else:
        # Default minimal set
        runner.add_scenario("quick_test", "The quick brown")
        runner.add_scenario("weather_test", "The weather is")

    # Set GA parameters for all experiments
    for scenario in runner.scenarios:
        scenario.setdefault('ga_params', {}).update({
            'population_size': args.population_size,
            'max_generations': args.generations,
            'max_tokens_per_individual': args.max_tokens
        })

    try:
        # Run experiments
        runner.run_all_experiments()

        # Analyze patterns
        runner.analyze_token_patterns()

        # Save results
        runner.save_results(args.output_dir)

        # Print summary
        print(runner.generate_report())

    except Exception as e:
        logging.error(f"Error during batch execution: {e}")
        raise


# CLI integration moved to main cli.py
# if __name__ == "__main__":
#     main()
