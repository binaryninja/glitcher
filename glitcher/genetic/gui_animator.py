#!/usr/bin/env python3
"""
Enhanced Real-Time GUI Animator for Genetic Algorithm Evolution

This module provides a comprehensive real-time visualization for genetic algorithm
evolution, supporting both target reduction and wanted token maximization modes.

Features:
- Adaptive layout for single-objective vs multi-objective optimization
- Real-time fitness evolution charts
- Token combination visualization with readable text
- Baseline probability comparison
- Enhanced context display with string construction
- LLM response comparison (baseline vs current best)
- Modern matplotlib interface with proper formatting

Author: Claude
Date: 2024
"""

import threading
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.axes import Axes


@dataclass
class EvolutionMetrics:
    """Container for evolution metrics and data."""
    generation: int = 0
    best_fitness: float = 0.0
    average_fitness: float = 0.0
    diversity: float = 0.0
    stagnation: int = 0

    # Token-specific metrics
    target_reduction: float = 0.0
    wanted_increase: float = 0.0
    target_prob_before: float = 0.0
    target_prob_after: float = 0.0
    wanted_prob_before: float = 0.0
    wanted_prob_after: float = 0.0

    # Best individual data
    best_tokens: List[int] = None
    best_token_texts: List[str] = None
    top_predicted_tokens: List[Tuple[int, str, float]] = None

    # LLM responses
    baseline_response: str = ""
    current_response: str = ""
    full_input_string: str = ""
    baseline_input_string: str = ""

    def __post_init__(self):
        if self.best_tokens is None:
            self.best_tokens = []
        if self.best_token_texts is None:
            self.best_token_texts = []
        if self.top_predicted_tokens is None:
            self.top_predicted_tokens = []


class EnhancedGeneticAnimator:
    """
    Enhanced real-time GUI animator for genetic algorithm evolution.

    Supports both single-objective (wanted token only) and multi-objective
    (target + wanted) optimization with adaptive visualization layouts.
    """

    def __init__(self, base_text: str, target_token: Optional[str] = None,
                 wanted_token: Optional[str] = None, max_generations: int = 100):
        """
        Initialize the enhanced genetic animator.

        Args:
            base_text: Base text being modified
            target_token: Token to reduce (optional)
            wanted_token: Token to maximize (optional)
            max_generations: Maximum number of generations
        """
        self.base_text = base_text
        self.target_token = target_token
        self.wanted_token = wanted_token
        self.max_generations = max_generations

        # Determine operation mode
        self.is_wanted_only = target_token is None and wanted_token is not None
        self.is_multi_objective = target_token is not None and wanted_token is not None
        self.is_target_only = target_token is not None and wanted_token is None

        # Animation state
        self.is_running = False
        self.is_complete = False
        self.animation_thread: Optional[threading.Thread] = None

        # Data storage
        self.metrics_history: List[EvolutionMetrics] = []
        self.current_metrics = EvolutionMetrics()
        self.baseline_data: Dict[str, Any] = {}

        # Matplotlib components
        self.fig: Optional[Figure] = None
        self.axes: Dict[str, Axes] = {}
        self.plots: Dict[str, Any] = {}
        self.text_elements: Dict[str, Any] = {}

        # Animation control
        self.animation_obj: Optional[animation.FuncAnimation] = None
        self.update_interval = 500  # milliseconds

        self._setup_matplotlib()
        self._create_layout()

    def _setup_matplotlib(self):
        """Setup matplotlib for interactive use."""
        try:
            import matplotlib
            matplotlib.use('TkAgg')  # Use TkAgg backend for better GUI support
        except ImportError:
            try:
                import matplotlib
                matplotlib.use('Qt5Agg')  # Fallback to Qt5
            except ImportError:
                pass  # Use default backend

        # Set style for better appearance
        plt.style.use('default')
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.linewidth': 0.8,
            'font.size': 9,
            'font.family': 'sans-serif',
            'text.color': 'black',
            'axes.labelcolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black',
            'grid.alpha': 0.3,
            'lines.linewidth': 1.5,
            'axes.grid': True,
            'grid.linewidth': 0.5
        })

    def _create_layout(self):
        """Create appropriate layout based on optimization mode."""
        if self.is_wanted_only:
            self._create_wanted_only_layout()
        elif self.is_multi_objective:
            self._create_multi_objective_layout()
        else:  # target_only
            self._create_target_only_layout()

    def _create_wanted_only_layout(self):
        """Create layout optimized for wanted token maximization only."""
        self.fig = plt.figure(figsize=(16, 12))
        title = f'Maximizing Wanted Token: "{self.wanted_token}"'
        self.fig.suptitle(title, fontsize=14, fontweight='bold', color='#228B22')

        # Create grid layout - added extra row for LLM responses
        gs = self.fig.add_gridspec(4, 2, height_ratios=[2, 1, 1.2, 1.8], width_ratios=[3, 2],
                                  hspace=0.5, wspace=0.3)

        # Main fitness evolution chart
        self.axes['fitness'] = self.fig.add_subplot(gs[0, :])
        self.axes['fitness'].set_title('Fitness Evolution (Wanted Token Probability)', fontweight='bold')
        self.axes['fitness'].set_xlabel('Generation')
        self.axes['fitness'].set_ylabel('Fitness Score')

        # Probability evolution chart
        self.axes['probability'] = self.fig.add_subplot(gs[1, 0])
        self.axes['probability'].set_title('Wanted Token Probability Evolution', fontweight='bold')
        self.axes['probability'].set_xlabel('Generation')
        self.axes['probability'].set_ylabel('Probability')

        # Token combination display
        self.axes['tokens'] = self.fig.add_subplot(gs[1, 1])
        self.axes['tokens'].set_title('Best Token Combination', fontweight='bold')
        self.axes['tokens'].axis('off')

        # Context and predictions display
        self.axes['context'] = self.fig.add_subplot(gs[2, :])
        self.axes['context'].set_title('Context & Input Strings', fontweight='bold')
        self.axes['context'].axis('off')

        # LLM responses display
        self.axes['responses'] = self.fig.add_subplot(gs[3, :])
        self.axes['responses'].set_title('LLM Responses Comparison', fontweight='bold')
        self.axes['responses'].axis('off')

        self._initialize_wanted_only_plots()

    def _create_multi_objective_layout(self):
        """Create layout for multi-objective optimization."""
        self.fig = plt.figure(figsize=(18, 14))
        title = f'Multi-Objective: Reduce "{self.target_token}" | Maximize "{self.wanted_token}"'
        self.fig.suptitle(title, fontsize=14, fontweight='bold', color='#4169E1')

        # Create grid layout - added extra row for LLM responses
        gs = self.fig.add_gridspec(5, 2, height_ratios=[2, 1, 1, 1.2, 1.8], width_ratios=[3, 2],
                                  hspace=0.5, wspace=0.3)

        # Main fitness evolution chart
        self.axes['fitness'] = self.fig.add_subplot(gs[0, :])
        self.axes['fitness'].set_title('Combined Fitness Evolution', fontweight='bold')
        self.axes['fitness'].set_xlabel('Generation')
        self.axes['fitness'].set_ylabel('Fitness Score')

        # Target probability evolution
        self.axes['target'] = self.fig.add_subplot(gs[1, 0])
        self.axes['target'].set_title('Target Token Probability (Reduce)', fontweight='bold', color='red')
        self.axes['target'].set_xlabel('Generation')
        self.axes['target'].set_ylabel('Probability')

        # Wanted probability evolution
        self.axes['wanted'] = self.fig.add_subplot(gs[1, 1])
        self.axes['wanted'].set_title('Wanted Token Probability (Maximize)', fontweight='bold', color='green')
        self.axes['wanted'].set_xlabel('Generation')
        self.axes['wanted'].set_ylabel('Probability')

        # Token combination display
        self.axes['tokens'] = self.fig.add_subplot(gs[2, :])
        self.axes['tokens'].set_title('Best Token Combination', fontweight='bold')
        self.axes['tokens'].axis('off')

        # Context and predictions display
        self.axes['context'] = self.fig.add_subplot(gs[3, :])
        self.axes['context'].set_title('Context & Input Strings', fontweight='bold')
        self.axes['context'].axis('off')

        # LLM responses display
        self.axes['responses'] = self.fig.add_subplot(gs[4, :])
        self.axes['responses'].set_title('LLM Responses Comparison', fontweight='bold')
        self.axes['responses'].axis('off')

        self._initialize_multi_objective_plots()

    def _create_target_only_layout(self):
        """Create layout for target token reduction only."""
        self.fig = plt.figure(figsize=(16, 12))
        title = f'Reducing Target Token: "{self.target_token}"'
        self.fig.suptitle(title, fontsize=14, fontweight='bold', color='#DC143C')

        # Create grid layout - added extra row for LLM responses
        gs = self.fig.add_gridspec(4, 2, height_ratios=[2, 1, 1.2, 1.8], width_ratios=[3, 2],
                                  hspace=0.5, wspace=0.3)

        # Main fitness evolution chart
        self.axes['fitness'] = self.fig.add_subplot(gs[0, :])
        self.axes['fitness'].set_title('Fitness Evolution (Target Reduction)', fontweight='bold')
        self.axes['fitness'].set_xlabel('Generation')
        self.axes['fitness'].set_ylabel('Fitness Score')

        # Probability evolution chart
        self.axes['probability'] = self.fig.add_subplot(gs[1, 0])
        self.axes['probability'].set_title('Target Token Probability Evolution', fontweight='bold')
        self.axes['probability'].set_xlabel('Generation')
        self.axes['probability'].set_ylabel('Probability')

        # Token combination display
        self.axes['tokens'] = self.fig.add_subplot(gs[1, 1])
        self.axes['tokens'].set_title('Best Token Combination', fontweight='bold')
        self.axes['tokens'].axis('off')

        # Context and predictions display
        self.axes['context'] = self.fig.add_subplot(gs[2, :])
        self.axes['context'].set_title('Context & Input Strings', fontweight='bold')
        self.axes['context'].axis('off')

        # LLM responses display
        self.axes['responses'] = self.fig.add_subplot(gs[3, :])
        self.axes['responses'].set_title('LLM Responses Comparison', fontweight='bold')
        self.axes['responses'].axis('off')

        self._initialize_target_only_plots()

    def _initialize_wanted_only_plots(self):
        """Initialize plots for wanted token optimization."""
        # Initialize fitness plot
        self.plots['fitness_best'], = self.axes['fitness'].plot([], [], 'g-',
                                                               linewidth=2, label='Best Fitness')
        self.plots['fitness_avg'], = self.axes['fitness'].plot([], [], 'b--',
                                                              alpha=0.7, label='Average Fitness')
        self.axes['fitness'].legend()
        self.axes['fitness'].grid(True, alpha=0.3)

        # Initialize probability plot
        self.plots['wanted_prob'], = self.axes['probability'].plot([], [], 'g-',
                                                                  linewidth=2, label='Wanted Probability')
        self.axes['probability'].legend()
        self.axes['probability'].grid(True, alpha=0.3)

        # Initialize text elements
        self.text_elements['tokens_text'] = self.axes['tokens'].text(0.02, 0.95, "",
                                                                    transform=self.axes['tokens'].transAxes,
                                                                    fontsize=10, verticalalignment='top',
                                                                    fontfamily='monospace')

        self.text_elements['context_text'] = self.axes['context'].text(0.02, 0.95, "",
                                                                       transform=self.axes['context'].transAxes,
                                                                       fontsize=9, verticalalignment='top',
                                                                       fontfamily='monospace')

        self.text_elements['responses_text'] = self.axes['responses'].text(0.02, 0.95, "",
                                                                           transform=self.axes['responses'].transAxes,
                                                                           fontsize=9, verticalalignment='top',
                                                                           fontfamily='monospace')

    def _initialize_multi_objective_plots(self):
        """Initialize plots for multi-objective optimization."""
        # Initialize fitness plot
        self.plots['fitness_best'], = self.axes['fitness'].plot([], [], 'b-',
                                                               linewidth=2, label='Combined Fitness')
        self.plots['fitness_avg'], = self.axes['fitness'].plot([], [], 'cyan',
                                                              alpha=0.7, linestyle='--', label='Average Fitness')
        self.axes['fitness'].legend()
        self.axes['fitness'].grid(True, alpha=0.3)

        # Initialize target probability plot
        self.plots['target_prob'], = self.axes['target'].plot([], [], 'r-',
                                                             linewidth=2, label='Target Probability')
        self.axes['target'].legend()
        self.axes['target'].grid(True, alpha=0.3)

        # Initialize wanted probability plot
        self.plots['wanted_prob'], = self.axes['wanted'].plot([], [], 'g-',
                                                             linewidth=2, label='Wanted Probability')
        self.axes['wanted'].legend()
        self.axes['wanted'].grid(True, alpha=0.3)

        # Initialize text elements
        self.text_elements['tokens_text'] = self.axes['tokens'].text(0.02, 0.95, "",
                                                                    transform=self.axes['tokens'].transAxes,
                                                                    fontsize=10, verticalalignment='top',
                                                                    fontfamily='monospace')

        self.text_elements['context_text'] = self.axes['context'].text(0.02, 0.95, "",
                                                                       transform=self.axes['context'].transAxes,
                                                                       fontsize=9, verticalalignment='top',
                                                                       fontfamily='monospace')

        self.text_elements['responses_text'] = self.axes['responses'].text(0.02, 0.95, "",
                                                                           transform=self.axes['responses'].transAxes,
                                                                           fontsize=9, verticalalignment='top',
                                                                           fontfamily='monospace')

    def _initialize_target_only_plots(self):
        """Initialize plots for target token reduction."""
        # Initialize fitness plot
        self.plots['fitness_best'], = self.axes['fitness'].plot([], [], 'r-',
                                                               linewidth=2, label='Best Fitness')
        self.plots['fitness_avg'], = self.axes['fitness'].plot([], [], 'orange',
                                                              alpha=0.7, linestyle='--', label='Average Fitness')
        self.axes['fitness'].legend()
        self.axes['fitness'].grid(True, alpha=0.3)

        # Initialize probability plot
        self.plots['target_prob'], = self.axes['probability'].plot([], [], 'r-',
                                                                  linewidth=2, label='Target Probability')
        self.axes['probability'].legend()
        self.axes['probability'].grid(True, alpha=0.3)

        # Initialize text elements
        self.text_elements['tokens_text'] = self.axes['tokens'].text(0.02, 0.95, "",
                                                                    transform=self.axes['tokens'].transAxes,
                                                                    fontsize=10, verticalalignment='top',
                                                                    fontfamily='monospace')

        self.text_elements['context_text'] = self.axes['context'].text(0.02, 0.95, "",
                                                                       transform=self.axes['context'].transAxes,
                                                                       fontsize=9, verticalalignment='top',
                                                                       fontfamily='monospace')

        self.text_elements['responses_text'] = self.axes['responses'].text(0.02, 0.95, "",
                                                                           transform=self.axes['responses'].transAxes,
                                                                           fontsize=9, verticalalignment='top',
                                                                           fontfamily='monospace')

    def update_baseline(self, baseline_data: Dict[str, Any]):
        """Update baseline data for comparison."""
        self.baseline_data = baseline_data

        # Update baseline reference lines
        if self.is_wanted_only and 'wanted_baseline_prob' in baseline_data:
            baseline_prob = baseline_data['wanted_baseline_prob']
            if 'probability' in self.axes:
                self.axes['probability'].axhline(y=baseline_prob, color='gray',
                                                linestyle=':', alpha=0.7, label='Baseline')

        elif self.is_multi_objective:
            if 'target_baseline_prob' in baseline_data and 'target' in self.axes:
                self.axes['target'].axhline(y=baseline_data['target_baseline_prob'],
                                          color='gray', linestyle=':', alpha=0.7)
            if 'wanted_baseline_prob' in baseline_data and 'wanted' in self.axes:
                self.axes['wanted'].axhline(y=baseline_data['wanted_baseline_prob'],
                                          color='gray', linestyle=':', alpha=0.7)

        elif self.is_target_only and 'target_baseline_prob' in baseline_data:
            baseline_prob = baseline_data['target_baseline_prob']
            if 'probability' in self.axes:
                self.axes['probability'].axhline(y=baseline_prob, color='gray',
                                                linestyle=':', alpha=0.7, label='Baseline')

    def update_metrics(self, metrics: EvolutionMetrics):
        """Update current metrics and refresh display."""
        self.current_metrics = metrics
        self.metrics_history.append(metrics)

        # Trigger immediate visual update
        self._update_plots_immediate()

    def update_comprehensive_search_progress(self, current: int, total: int, best_impact: float, excellent_count: int):
        """Update GUI with comprehensive search progress."""
        if not self.is_running:
            return

        # Update the title to show search progress
        progress_pct = (current / total) * 100 if total > 0 else 0
        if self.fig:
            search_title = f"🔍 Comprehensive Search: {progress_pct:.1f}% ({current:,}/{total:,} tokens) | Best Impact: {best_impact:.6f} | Excellent: {excellent_count}"

            # Get current title and update it
            current_title = self.fig._suptitle.get_text() if self.fig._suptitle else ""
            if "🔍 Comprehensive Search:" not in current_title:
                # First time showing search progress
                if self.is_wanted_only:
                    base_title = f'Maximizing Wanted Token: "{self.wanted_token}"'
                elif self.is_multi_objective:
                    base_title = f'Multi-Objective: Reduce "{self.target_token}" | Maximize "{self.wanted_token}"'
                else:
                    base_title = f'Reducing Target Token: "{self.target_token}"'

                full_title = f"{base_title}\n{search_title}"
            else:
                # Update existing search progress
                lines = current_title.split('\n')
                if len(lines) >= 2:
                    full_title = f"{lines[0]}\n{search_title}"
                else:
                    full_title = f"{current_title}\n{search_title}"

            self.fig.suptitle(full_title, fontsize=12, fontweight='bold')

            # Force a GUI update to show progress immediately
            if hasattr(self.fig.canvas, 'draw_idle'):
                self.fig.canvas.draw_idle()
            plt.pause(0.001)  # Very short pause to allow GUI update

    def _update_plots_immediate(self):
        """Update all plots immediately without animation delay."""
        if not self.is_running or not self.metrics_history:
            return

        # Update plot data
        if self.is_wanted_only:
            self._update_wanted_only_plots()
        elif self.is_multi_objective:
            self._update_multi_objective_plots()
        else:
            self._update_target_only_plots()

        # Update text displays
        self._update_token_display()
        self._update_context_display()
        self._update_responses_display()

        # Refresh the display
        if hasattr(self.fig, 'canvas') and hasattr(self.fig.canvas, 'draw_idle'):
            self.fig.canvas.draw_idle()

    def _update_wanted_only_plots(self):
        """Update plots for wanted token optimization."""
        generations = [m.generation for m in self.metrics_history]
        best_fitness = [m.best_fitness for m in self.metrics_history]
        avg_fitness = [m.average_fitness for m in self.metrics_history]
        wanted_probs = [m.wanted_prob_after for m in self.metrics_history]

        self.plots['fitness_best'].set_data(generations, best_fitness)
        self.plots['fitness_avg'].set_data(generations, avg_fitness)
        self.plots['wanted_prob'].set_data(generations, wanted_probs)

        # Auto-scale axes
        self.axes['fitness'].relim()
        self.axes['fitness'].autoscale_view()
        self.axes['probability'].relim()
        self.axes['probability'].autoscale_view()

    def _update_multi_objective_plots(self):
        """Update plots for multi-objective optimization."""
        generations = [m.generation for m in self.metrics_history]
        best_fitness = [m.best_fitness for m in self.metrics_history]
        avg_fitness = [m.average_fitness for m in self.metrics_history]
        target_probs = [m.target_prob_after for m in self.metrics_history]
        wanted_probs = [m.wanted_prob_after for m in self.metrics_history]

        self.plots['fitness_best'].set_data(generations, best_fitness)
        self.plots['fitness_avg'].set_data(generations, avg_fitness)
        self.plots['target_prob'].set_data(generations, target_probs)
        self.plots['wanted_prob'].set_data(generations, wanted_probs)

        # Auto-scale axes
        self.axes['fitness'].relim()
        self.axes['fitness'].autoscale_view()
        self.axes['target'].relim()
        self.axes['target'].autoscale_view()
        self.axes['wanted'].relim()
        self.axes['wanted'].autoscale_view()

    def _update_target_only_plots(self):
        """Update plots for target token reduction."""
        generations = [m.generation for m in self.metrics_history]
        best_fitness = [m.best_fitness for m in self.metrics_history]
        avg_fitness = [m.average_fitness for m in self.metrics_history]
        target_probs = [m.target_prob_after for m in self.metrics_history]

        self.plots['fitness_best'].set_data(generations, best_fitness)
        self.plots['fitness_avg'].set_data(generations, avg_fitness)
        self.plots['target_prob'].set_data(generations, target_probs)

        # Auto-scale axes
        self.axes['fitness'].relim()
        self.axes['fitness'].autoscale_view()
        self.axes['probability'].relim()
        self.axes['probability'].autoscale_view()

    def _update_token_display(self):
        """Update the token combination display - show token values instead of IDs."""
        tokens_text = "Current Best Combination:\n"
        if self.current_metrics.best_token_texts:
            # Show token values instead of IDs
            token_values = ", ".join([f"'{text}'" for text in self.current_metrics.best_token_texts])
            tokens_text += f"Tokens: {token_values}\n"
            if self.current_metrics.best_tokens:
                token_ids = ", ".join([str(tid) for tid in self.current_metrics.best_tokens])
                tokens_text += f"IDs: [{token_ids}]\n\n"
        else:
            tokens_text += "No tokens yet...\n\n"

        tokens_text += f"Generation: {self.current_metrics.generation}\n"
        tokens_text += f"Fitness: {self.current_metrics.best_fitness:.6f}\n"
        tokens_text += f"Diversity: {self.current_metrics.diversity:.3f}\n"
        tokens_text += f"Stagnation: {self.current_metrics.stagnation}\n\n"

        if self.is_wanted_only:
            wanted_increase_pct = (self.current_metrics.wanted_increase /
                                 (1.0 - self.current_metrics.wanted_prob_before) * 100) \
                                 if self.current_metrics.wanted_prob_before < 1.0 else 0
            tokens_text += f"Wanted Increase: {wanted_increase_pct:.2f}%\n"
            tokens_text += f"Probability: {self.current_metrics.wanted_prob_before:.6f} → {self.current_metrics.wanted_prob_after:.6f}"

        elif self.is_multi_objective:
            target_reduction_pct = (self.current_metrics.target_reduction /
                                  self.current_metrics.target_prob_before * 100) \
                                  if self.current_metrics.target_prob_before > 0 else 0
            wanted_increase_pct = (self.current_metrics.wanted_increase /
                                 (1.0 - self.current_metrics.wanted_prob_before) * 100) \
                                 if self.current_metrics.wanted_prob_before < 1.0 else 0
            tokens_text += f"Target Reduction: {target_reduction_pct:.2f}%\n"
            tokens_text += f"Wanted Increase: {wanted_increase_pct:.2f}%\n"
            tokens_text += f"Target: {self.current_metrics.target_prob_before:.6f} → {self.current_metrics.target_prob_after:.6f}\n"
            tokens_text += f"Wanted: {self.current_metrics.wanted_prob_before:.6f} → {self.current_metrics.wanted_prob_after:.6f}"

        else:  # target_only
            target_reduction_pct = (self.current_metrics.target_reduction /
                                  self.current_metrics.target_prob_before * 100) \
                                  if self.current_metrics.target_prob_before > 0 else 0
            tokens_text += f"Target Reduction: {target_reduction_pct:.2f}%\n"
            tokens_text += f"Probability: {self.current_metrics.target_prob_before:.6f} → {self.current_metrics.target_prob_after:.6f}"

        self.text_elements['tokens_text'].set_text(tokens_text)

    def _update_context_display(self):
        """Update the context and input strings display."""
        context_text = "Input Strings for LLM:\n\n"

        # Show baseline input string
        if self.current_metrics.baseline_input_string:
            context_text += f"BASELINE INPUT:\n'{self.current_metrics.baseline_input_string}'\n\n"
        else:
            context_text += f"BASELINE INPUT:\n'{self.base_text}'\n\n"

        # Show current evolved input string
        if self.current_metrics.full_input_string:
            context_text += f"CURRENT INPUT:\n'{self.current_metrics.full_input_string}'\n\n"
        else:
            # Construct evolved string if not provided
            evolved_prefix = "".join(self.current_metrics.best_token_texts) if self.current_metrics.best_token_texts else ""
            full_string = f"{evolved_prefix}{self.base_text}"
            context_text += f"CURRENT INPUT:\n'{full_string}'\n\n"

        # Show top predicted tokens
        if self.current_metrics.top_predicted_tokens:
            context_text += "Top 5 Predicted Tokens:\n"
            for i, (token_id, token_text, prob) in enumerate(self.current_metrics.top_predicted_tokens[:5]):
                context_text += f"  {i+1}. '{token_text}' (ID: {token_id}) - {prob:.4f}\n"

        self.text_elements['context_text'].set_text(context_text)

    def _update_responses_display(self):
        """Update the LLM responses comparison display."""
        responses_text = "LLM Responses Comparison:\n\n"

        # Show baseline response
        if self.current_metrics.baseline_response:
            responses_text += f"BASELINE RESPONSE:\n'{self.current_metrics.baseline_response}'\n\n"
        else:
            responses_text += "BASELINE RESPONSE:\n(No baseline response available)\n\n"

        # Show current best response
        if self.current_metrics.current_response:
            responses_text += f"CURRENT BEST RESPONSE:\n'{self.current_metrics.current_response}'\n\n"
        else:
            responses_text += "CURRENT BEST RESPONSE:\n(No current response available)\n\n"

        # Add comparison if both responses are available
        if self.current_metrics.baseline_response and self.current_metrics.current_response:
            if self.current_metrics.baseline_response == self.current_metrics.current_response:
                responses_text += "STATUS: ⚠️  Responses are identical\n"
            else:
                responses_text += "STATUS: ✅ Responses are different\n"
                # Calculate character difference
                baseline_len = len(self.current_metrics.baseline_response)
                current_len = len(self.current_metrics.current_response)
                responses_text += f"Length change: {baseline_len} → {current_len} chars ({current_len - baseline_len:+d})\n"

        self.text_elements['responses_text'].set_text(responses_text)

    def start_animation(self):
        """Start the real-time animation."""
        if self.is_running:
            return

        print("🚀 Starting GUI animation...")
        self.is_running = True

        # Show the figure
        plt.show(block=False)

        # Ensure proper display
        if self.fig and hasattr(self.fig, 'canvas'):
            try:
                self.fig.canvas.manager.show()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                print("✅ GUI window displayed")
            except Exception as e:
                print(f"⚠️  Warning: Could not optimize window display: {e}")

        # Initial update if we have data
        if self.metrics_history:
            self._update_plots_immediate()

    def stop_animation(self):
        """Stop the animation."""
        self.is_running = False

    def mark_complete(self):
        """Mark evolution as complete and add completion indicator."""
        self.is_complete = True

        # Add completion marker to fitness plot
        if self.metrics_history and 'fitness' in self.axes:
            final_gen = self.metrics_history[-1].generation
            final_fitness = self.metrics_history[-1].best_fitness

            self.axes['fitness'].annotate('✅ Complete',
                                        xy=(final_gen, final_fitness),
                                        xytext=(final_gen - 5, final_fitness + 0.05),
                                        arrowprops=dict(arrowstyle='->', color='green', lw=2),
                                        fontsize=12, fontweight='bold', color='green')

        # Update title to show completion
        if self.fig and hasattr(self.fig, '_suptitle') and self.fig._suptitle:
            current_title = self.fig._suptitle.get_text()
            self.fig.suptitle(f"✅ COMPLETE: {current_title}", fontweight='bold', color='green')

        # Force display update
        if self.fig and hasattr(self.fig, 'canvas'):
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def keep_alive(self):
        """Keep the GUI window alive after evolution completes."""
        try:
            print("🖼️  GUI animation is live. Close the window to exit.")
            while plt.get_fignums():
                plt.pause(0.5)
        except KeyboardInterrupt:
            print("\n🛑 Animation stopped by user.")
        finally:
            plt.close('all')


class GeneticAnimationCallback:
    """
    Callback interface for genetic algorithm integration.

    This class provides the interface between the genetic algorithm
    and the GUI animator, handling data updates and lifecycle events.
    """

    def __init__(self, animator: EnhancedGeneticAnimator, tokenizer=None):
        """
        Initialize the animation callback.

        Args:
            animator: The GUI animator instance
            tokenizer: Tokenizer for decoding token texts
        """
        self.animator = animator
        self.tokenizer = tokenizer

    def on_evolution_start(self, baseline_data: Dict[str, Any]):
        """Called when evolution starts."""
        self.animator.update_baseline(baseline_data)
        self.animator.start_animation()

    def set_tokenizer(self, tokenizer):
        """Set the tokenizer for token decoding."""
        self.tokenizer = tokenizer

    def on_generation_complete(self, generation: int, population: List[Any],
                             best_individual: Any, diversity: float, stagnation: int):
        """Called when a generation completes."""
        # Extract metrics from the genetic algorithm state
        metrics = EvolutionMetrics(
            generation=generation,
            best_fitness=best_individual.fitness,
            diversity=diversity,
            stagnation=stagnation,
            target_reduction=getattr(best_individual, 'target_reduction', 0.0),
            wanted_increase=getattr(best_individual, 'wanted_increase', 0.0),
            target_prob_before=getattr(best_individual, 'baseline_prob', 0.0),
            target_prob_after=getattr(best_individual, 'modified_prob', 0.0),
            wanted_prob_before=getattr(best_individual, 'wanted_baseline_prob', 0.0),
            wanted_prob_after=getattr(best_individual, 'wanted_modified_prob', 0.0),
            best_tokens=getattr(best_individual, 'tokens', []),
            best_token_texts=[],  # Will be filled below
            top_predicted_tokens=[],  # Will be filled below
            baseline_response=getattr(best_individual, 'baseline_response', ''),
            current_response=getattr(best_individual, 'current_response', ''),
            full_input_string=getattr(best_individual, 'full_input_string', ''),
            baseline_input_string=getattr(best_individual, 'baseline_input_string', '')
        )

        # Calculate average fitness
        if population:
            total_fitness = sum(getattr(ind, 'fitness', 0.0) for ind in population)
            metrics.average_fitness = total_fitness / len(population)

        # Get token texts with proper decoding (show values instead of IDs)
        if hasattr(best_individual, 'tokens') and best_individual.tokens:
            if self.tokenizer:
                try:
                    metrics.best_token_texts = [self.tokenizer.decode([token_id]) for token_id in best_individual.tokens]
                except Exception:
                    metrics.best_token_texts = [f"Token_{tid}" for tid in best_individual.tokens]
            elif hasattr(best_individual, 'token_texts'):
                metrics.best_token_texts = best_individual.token_texts
            else:
                metrics.best_token_texts = [f"Token_{tid}" for tid in best_individual.tokens]

        # Get top predicted tokens with proper decoding
        if hasattr(best_individual, 'new_top_tokens') and best_individual.new_top_tokens:
            if self.tokenizer:
                try:
                    metrics.top_predicted_tokens = [
                        (token_id, self.tokenizer.decode([token_id]), prob)
                        for token_id, prob in best_individual.new_top_tokens[:10]
                    ]
                except Exception:
                    metrics.top_predicted_tokens = [
                        (token_id, f"Token_{token_id}", prob)
                        for token_id, prob in best_individual.new_top_tokens[:10]
                    ]
            else:
                metrics.top_predicted_tokens = [
                    (token_id, f"Token_{token_id}", prob)
                    for token_id, prob in best_individual.new_top_tokens[:10]
                ]

        self.animator.update_metrics(metrics)

    def on_evolution_complete(self, final_population: List[Any]):
        """Called when evolution completes."""
        self.animator.mark_complete()

    def keep_alive(self):
        """Keep the animation alive for viewing."""
        self.animator.keep_alive()


# Backwards compatibility aliases
RealTimeGeneticAnimator = EnhancedGeneticAnimator
