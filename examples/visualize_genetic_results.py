#!/usr/bin/env python3
"""
Visualization Script for Genetic Algorithm Results

This script creates various visualizations of genetic algorithm experiments
for probability reducer token combinations.

Author: Claude
Date: 2024
"""

import json
import argparse
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class GeneticResultsVisualizer:
    """
    Visualizer for genetic algorithm experiment results.

    遗传算法实验结果可视化器。
    """

    def __init__(self, results_file: str):
        """
        Initialize the visualizer.

        Args:
            results_file: Path to JSON results file
        """
        self.results_file = results_file
        self.data = None
        self.experiments = []
        self.pattern_analysis = {}

        self.setup_logging()
        self.load_results()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_results(self):
        """Load results from JSON file."""
        self.logger.info(f"Loading results from: {self.results_file}")

        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)

            self.experiments = self.data.get('experiments', [])
            self.pattern_analysis = self.data.get('pattern_analysis', {})

            self.logger.info(f"Loaded {len(self.experiments)} experiments")

        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            raise

    def plot_scenario_difficulty(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot scenario difficulty analysis.

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if not self.pattern_analysis or 'scenario_analysis' not in self.pattern_analysis:
            self.logger.warning("No scenario analysis data available")
            return None

        scenario_data = self.pattern_analysis['scenario_analysis']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Scenario Difficulty Analysis', fontsize=16)

        # Extract data
        scenarios = [s['scenario'] for s in scenario_data]
        baseline_probs = [s['baseline_probability'] for s in scenario_data]
        best_reductions = [s['best_reduction_pct'] for s in scenario_data]
        valid_solutions = [s['valid_solutions'] for s in scenario_data]
        difficulty_scores = [s['difficulty_score'] for s in scenario_data]

        # Plot 1: Baseline Probability vs Best Reduction
        ax1.scatter(baseline_probs, best_reductions, alpha=0.7, s=100)
        for i, scenario in enumerate(scenarios):
            ax1.annotate(scenario.replace('_', '\n'),
                        (baseline_probs[i], best_reductions[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        ax1.set_xlabel('Baseline Probability')
        ax1.set_ylabel('Best Reduction (%)')
        ax1.set_title('Baseline Probability vs Best Reduction')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Valid Solutions Count
        bars = ax2.bar(range(len(scenarios)), valid_solutions, alpha=0.7)
        ax2.set_xlabel('Scenario')
        ax2.set_ylabel('Valid Solutions Found')
        ax2.set_title('Number of Valid Solutions per Scenario')
        ax2.set_xticks(range(len(scenarios)))
        ax2.set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=45, ha='right')

        # Color bars by difficulty
        norm_difficulty = np.array(difficulty_scores) / max(difficulty_scores)
        for bar, diff in zip(bars, norm_difficulty):
            bar.set_color(plt.cm.RdYlBu_r(diff))

        # Plot 3: Difficulty Score Distribution
        ax3.hist(difficulty_scores, bins=10, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Difficulty Score')
        ax3.set_ylabel('Number of Scenarios')
        ax3.set_title('Difficulty Score Distribution')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Baseline Probability Distribution
        ax4.hist(baseline_probs, bins=10, alpha=0.7, edgecolor='black', color='orange')
        ax4.set_xlabel('Baseline Probability')
        ax4.set_ylabel('Number of Scenarios')
        ax4.set_title('Baseline Probability Distribution')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_token_effectiveness(self, figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Plot token effectiveness analysis.

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if not self.pattern_analysis or 'top_individual_tokens' not in self.pattern_analysis:
            self.logger.warning("No token effectiveness data available")
            return None

        token_data = self.pattern_analysis['top_individual_tokens'][:15]  # Top 15

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Token Effectiveness Analysis', fontsize=16)

        # Extract data
        token_ids = [str(t['token_id']) for t in token_data]
        frequencies = [t['frequency'] for t in token_data]
        avg_reductions = [t['avg_reduction_pct'] for t in token_data]
        effectiveness_ratios = [t['effectiveness_ratio'] for t in token_data]
        scenarios_effective = [t['scenarios_effective'] for t in token_data]

        # Plot 1: Token Frequency
        bars1 = ax1.barh(range(len(token_ids)), frequencies, alpha=0.7)
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Token ID')
        ax1.set_title('Token Usage Frequency')
        ax1.set_yticks(range(len(token_ids)))
        ax1.set_yticklabels(token_ids)

        # Plot 2: Average Reduction Percentage
        bars2 = ax2.barh(range(len(token_ids)), avg_reductions, alpha=0.7, color='orange')
        ax2.set_xlabel('Average Reduction (%)')
        ax2.set_ylabel('Token ID')
        ax2.set_title('Average Reduction Percentage')
        ax2.set_yticks(range(len(token_ids)))
        ax2.set_yticklabels(token_ids)

        # Plot 3: Effectiveness Ratio (scatter)
        scatter = ax3.scatter(frequencies, avg_reductions,
                            s=[r*200 for r in effectiveness_ratios],
                            alpha=0.6, c=effectiveness_ratios,
                            cmap='viridis')
        ax3.set_xlabel('Frequency')
        ax3.set_ylabel('Average Reduction (%)')
        ax3.set_title('Frequency vs Reduction (size = effectiveness ratio)')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Effectiveness Ratio')

        # Plot 4: Scenarios Effective
        bars4 = ax4.barh(range(len(token_ids)), scenarios_effective, alpha=0.7, color='green')
        ax4.set_xlabel('Scenarios Effective')
        ax4.set_ylabel('Token ID')
        ax4.set_title('Number of Scenarios Where Token is Effective')
        ax4.set_yticks(range(len(token_ids)))
        ax4.set_yticklabels(token_ids)

        plt.tight_layout()
        return fig

    def plot_combination_patterns(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot token combination patterns.

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if not self.pattern_analysis or 'top_token_combinations' not in self.pattern_analysis:
            self.logger.warning("No combination data available")
            return None

        combo_data = self.pattern_analysis['top_token_combinations'][:10]  # Top 10

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Token Combination Patterns', fontsize=16)

        # Extract data
        combo_labels = [f"{c['tokens']}" for c in combo_data]
        combo_short_labels = [f"C{i+1}" for i in range(len(combo_data))]  # Shorter labels
        frequencies = [c['frequency'] for c in combo_data]
        avg_reductions = [c['avg_reduction_pct'] for c in combo_data]
        effectiveness_ratios = [c['effectiveness_ratio'] for c in combo_data]

        # Plot 1: Combination Frequency
        ax1.bar(combo_short_labels, frequencies, alpha=0.7)
        ax1.set_xlabel('Combination')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Combination Frequency')
        ax1.tick_params(axis='x', rotation=45)

        # Plot 2: Average Reduction
        ax2.bar(combo_short_labels, avg_reductions, alpha=0.7, color='orange')
        ax2.set_xlabel('Combination')
        ax2.set_ylabel('Average Reduction (%)')
        ax2.set_title('Average Reduction by Combination')
        ax2.tick_params(axis='x', rotation=45)

        # Plot 3: Effectiveness vs Frequency
        scatter = ax3.scatter(frequencies, avg_reductions,
                            s=[r*300 for r in effectiveness_ratios],
                            alpha=0.6, c=effectiveness_ratios,
                            cmap='plasma')
        ax3.set_xlabel('Frequency')
        ax3.set_ylabel('Average Reduction (%)')
        ax3.set_title('Frequency vs Reduction (size = effectiveness)')

        # Annotate points
        for i, label in enumerate(combo_short_labels):
            ax3.annotate(label, (frequencies[i], avg_reductions[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)

        plt.colorbar(scatter, ax=ax3, label='Effectiveness Ratio')

        # Plot 4: Combination length distribution
        combo_lengths = [len(c['tokens']) for c in combo_data]
        length_counts = Counter(combo_lengths)

        ax4.bar(length_counts.keys(), length_counts.values(), alpha=0.7, color='green')
        ax4.set_xlabel('Combination Length (# tokens)')
        ax4.set_ylabel('Count')
        ax4.set_title('Distribution of Combination Lengths')
        ax4.set_xticks(list(length_counts.keys()))

        plt.tight_layout()
        return fig

    def plot_experiment_overview(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot overview of all experiments.

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Prepare data
        experiment_data = []
        for exp in self.experiments:
            if 'error' in exp:
                continue

            for result in exp.get('top_results', []):
                experiment_data.append({
                    'scenario': exp['scenario_name'],
                    'baseline_prob': exp['baseline_probability'],
                    'reduction_pct': result['reduction_percentage'],
                    'fitness': result['fitness'],
                    'num_tokens': len(result['tokens'])
                })

        if not experiment_data:
            self.logger.warning("No experiment data available for overview")
            return None

        df = pd.DataFrame(experiment_data)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Experiment Overview', fontsize=16)

        # Plot 1: Reduction distribution by scenario
        scenarios = df['scenario'].unique()
        reduction_data = [df[df['scenario'] == s]['reduction_pct'].values for s in scenarios]

        box_plot = ax1.boxplot(reduction_data, labels=[s.replace('_', '\n') for s in scenarios])
        ax1.set_xlabel('Scenario')
        ax1.set_ylabel('Reduction Percentage')
        ax1.set_title('Reduction Distribution by Scenario')
        ax1.tick_params(axis='x', rotation=45)

        # Plot 2: Baseline probability vs reduction scatter
        scatter = ax2.scatter(df['baseline_prob'], df['reduction_pct'],
                            alpha=0.6, c=df['num_tokens'], cmap='viridis')
        ax2.set_xlabel('Baseline Probability')
        ax2.set_ylabel('Reduction Percentage')
        ax2.set_title('Baseline Probability vs Reduction')
        plt.colorbar(scatter, ax=ax2, label='Number of Tokens')

        # Plot 3: Number of tokens distribution
        token_counts = df['num_tokens'].value_counts().sort_index()
        ax3.bar(token_counts.index, token_counts.values, alpha=0.7)
        ax3.set_xlabel('Number of Tokens in Combination')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Token Combination Sizes')
        ax3.set_xticks(token_counts.index)

        # Plot 4: Fitness distribution
        ax4.hist(df['fitness'], bins=20, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Fitness Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Fitness Score Distribution')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_heatmap_token_scenarios(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot heatmap of token effectiveness across scenarios.

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Prepare data for heatmap
        token_scenario_data = defaultdict(lambda: defaultdict(float))
        scenarios = set()
        top_tokens = set()

        for exp in self.experiments:
            if 'error' in exp:
                continue

            scenario = exp['scenario_name']
            scenarios.add(scenario)

            for result in exp.get('top_results', [])[:5]:  # Top 5 per scenario
                for token in result['tokens']:
                    top_tokens.add(token)
                    # Use reduction percentage as effectiveness measure
                    token_scenario_data[token][scenario] = max(
                        token_scenario_data[token][scenario],
                        result['reduction_percentage']
                    )

        # Limit to most common tokens
        if self.pattern_analysis and 'top_individual_tokens' in self.pattern_analysis:
            top_tokens = [t['token_id'] for t in self.pattern_analysis['top_individual_tokens'][:15]]
        else:
            top_tokens = list(top_tokens)[:15]

        scenarios = sorted(scenarios)

        # Create matrix
        matrix = np.zeros((len(top_tokens), len(scenarios)))
        for i, token in enumerate(top_tokens):
            for j, scenario in enumerate(scenarios):
                matrix[i, j] = token_scenario_data[token][scenario]

        # Create heatmap
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        heatmap = sns.heatmap(matrix,
                             xticklabels=[s.replace('_', '\n') for s in scenarios],
                             yticklabels=[str(t) for t in top_tokens],
                             annot=True,
                             fmt='.1f',
                             cmap='YlOrRd',
                             ax=ax)

        ax.set_title('Token Effectiveness Across Scenarios\n(% Reduction)', fontsize=14)
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Token ID')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        return fig

    def create_summary_report(self, output_dir: str):
        """
        Create a comprehensive visual report.

        Args:
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Creating visual report in: {output_path}")

        # Create all plots
        plots = [
            ('scenario_difficulty', self.plot_scenario_difficulty),
            ('token_effectiveness', self.plot_token_effectiveness),
            ('combination_patterns', self.plot_combination_patterns),
            ('experiment_overview', self.plot_experiment_overview),
            ('token_scenario_heatmap', self.plot_heatmap_token_scenarios)
        ]

        for plot_name, plot_func in plots:
            try:
                fig = plot_func()
                if fig:
                    plot_path = output_path / f"{plot_name}.png"
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    self.logger.info(f"Saved plot: {plot_path}")
                else:
                    self.logger.warning(f"Could not create plot: {plot_name}")
            except Exception as e:
                self.logger.error(f"Error creating plot {plot_name}: {e}")

        self.logger.info("Visual report creation completed")

    def show_interactive_plots(self):
        """Show all plots interactively."""
        plots = [
            self.plot_scenario_difficulty,
            self.plot_token_effectiveness,
            self.plot_combination_patterns,
            self.plot_experiment_overview,
            self.plot_heatmap_token_scenarios
        ]

        for plot_func in plots:
            try:
                fig = plot_func()
                if fig:
                    plt.show()
                else:
                    self.logger.warning(f"Could not create plot: {plot_func.__name__}")
            except Exception as e:
                self.logger.error(f"Error displaying plot {plot_func.__name__}: {e}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Visualize Genetic Algorithm Results"
    )
    parser.add_argument(
        "results_file",
        help="JSON file with genetic algorithm results"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save plots (if not provided, shows interactively)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved plots"
    )

    args = parser.parse_args()

    try:
        # Create visualizer
        visualizer = GeneticResultsVisualizer(args.results_file)

        if args.output_dir:
            # Save plots to directory
            visualizer.create_summary_report(args.output_dir)
        else:
            # Show plots interactively
            visualizer.show_interactive_plots()

    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise


if __name__ == "__main__":
    main()
