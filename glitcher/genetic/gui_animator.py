#!/usr/bin/env python3
"""
Real-time GUI Animator for Genetic Algorithm Evolution

This module provides real-time visualization of genetic algorithm progress,
showing fitness evolution, token combinations, and probability reduction stats.

Author: Claude
Date: 2024
"""

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    animation = None
    MATPLOTLIB_AVAILABLE = False

import numpy as np
import threading
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import logging


class RealTimeGeneticAnimator:
    """
    Real-time animator for genetic algorithm evolution.

    Displays live updates of fitness evolution, current best tokens,
    and probability reduction statistics as the GA progresses.
    """

    def __init__(self, base_text: str, target_token_text: str = None,
                 target_token_id: int = None, baseline_probability: float = None,
                 max_generations: int = 100):
        """
        Initialize the real-time animator.

        Args:
            base_text: Base text being tested
            target_token_text: Target token text (if known)
            target_token_id: Target token ID (if known)
            baseline_probability: Baseline probability before optimization
            max_generations: Maximum number of generations to display

        Raises:
            ImportError: If matplotlib is not available
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for GUI animation. Install with: pip install matplotlib")
        self.base_text = base_text
        self.target_token_text = target_token_text or "[Auto-detected]"
        self.target_token_id = target_token_id
        self.baseline_probability = baseline_probability or 0.0
        self.max_generations = max_generations

        # Data storage for real-time updates
        self.generations = deque(maxlen=max_generations * 2)
        self.best_fitness = deque(maxlen=max_generations * 2)
        self.avg_fitness = deque(maxlen=max_generations * 2)
        self.current_generation = 0
        self.current_best_tokens = []
        self.current_token_texts = []
        self.current_best_fitness = 0.0
        self.current_avg_fitness = 0.0
        self.current_probability = baseline_probability or 0.0

        # Animation control
        self.is_running = False
        self.is_complete = False
        self.update_lock = threading.Lock()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Setup the plot
        self.setup_plot()

    def setup_plot(self):
        """Setup the matplotlib figure and subplots for real-time display."""
        # Set up matplotlib backend
        import matplotlib
        matplotlib.use('TkAgg', force=True)  # Use TkAgg backend for better compatibility

        plt.ion()  # Turn on interactive mode

        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('🧬 Genetic Algorithm Evolution: Real-time Token Breeding',
                         fontsize=16, fontweight='bold')

        # Create subplots with improved layout
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 2, 1], width_ratios=[2, 1],
                                  hspace=0.3, wspace=0.3)

        # Top: Basic info panel
        self.ax_info = self.fig.add_subplot(gs[0, :])
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis('off')

        # Middle left: Fitness evolution chart
        self.ax_fitness = self.fig.add_subplot(gs[1, 0])
        self.ax_fitness.set_title('🏆 Fitness Evolution Over Generations', fontweight='bold')
        self.ax_fitness.set_xlabel('Generation')
        self.ax_fitness.set_ylabel('Fitness Score')
        self.ax_fitness.grid(True, alpha=0.3)
        self.ax_fitness.set_xlim(0, max(50, self.max_generations))
        self.ax_fitness.set_ylim(0, 1.0)

        # Middle right: Current stats panel
        self.ax_stats = self.fig.add_subplot(gs[1, 1])
        self.ax_stats.set_title('📊 Current Statistics', fontweight='bold')
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')

        # Bottom: Token evolution panel
        self.ax_tokens = self.fig.add_subplot(gs[2, :])
        self.ax_tokens.set_title('🎯 Current Best Token Combination', fontweight='bold')
        self.ax_tokens.set_xlim(0, 1)
        self.ax_tokens.set_ylim(0, 1)
        self.ax_tokens.axis('off')

        # Initialize empty plots
        line_best_result = self.ax_fitness.plot([], [], 'b-', linewidth=3,
                                               label='Best Fitness', marker='o', markersize=4)
        self.line_best = line_best_result[0] if line_best_result else None

        line_avg_result = self.ax_fitness.plot([], [], 'r--', linewidth=2,
                                              label='Average Fitness', alpha=0.7)
        self.line_avg = line_avg_result[0] if line_avg_result else None

        self.ax_fitness.legend(loc='upper left')

        # Initial display
        self.update_display()
        plt.show(block=False)
        plt.pause(0.1)

    def update_data(self, generation: int, best_fitness: float, avg_fitness: float,
                   best_tokens: List[int], token_texts: List[str] = None,
                   current_probability: float = None):
        """
        Update the animator with new data from the genetic algorithm.

        Args:
            generation: Current generation number
            best_fitness: Best fitness score in current generation
            avg_fitness: Average fitness score in current generation
            best_tokens: List of token IDs in best individual
            token_texts: List of decoded token texts (optional)
            current_probability: Current probability after token insertion
        """
        with self.update_lock:
            self.current_generation = generation
            self.current_best_fitness = best_fitness
            self.current_avg_fitness = avg_fitness
            self.current_best_tokens = best_tokens.copy() if best_tokens else []
            self.current_token_texts = token_texts.copy() if token_texts else []

            if current_probability is not None:
                self.current_probability = current_probability

            # Store data for plotting
            self.generations.append(generation)
            self.best_fitness.append(best_fitness)
            self.avg_fitness.append(avg_fitness)

            # Update display if running
            if self.is_running:
                self.update_display()

    def update_baseline(self, baseline_probability: float, target_token_id: int = None,
                       target_token_text: str = None):
        """Update baseline information."""
        with self.update_lock:
            self.baseline_probability = baseline_probability
            if target_token_id is not None:
                self.target_token_id = target_token_id
            if target_token_text is not None:
                self.target_token_text = target_token_text

    def update_display(self):
        """Update the visual display with current data."""
        if not plt.fignum_exists(self.fig.number):
            return  # Window was closed

        try:
            # Update fitness plots
            if len(self.generations) > 0 and self.line_best is not None and self.line_avg is not None:
                x_data = list(self.generations)
                y_best = list(self.best_fitness)
                y_avg = list(self.avg_fitness)

                self.line_best.set_data(x_data, y_best)
                self.line_avg.set_data(x_data, y_avg)

                # Auto-scale axes if needed
                if len(x_data) > 0:
                    max_gen = max(x_data)
                    if max_gen > self.ax_fitness.get_xlim()[1] * 0.8:
                        self.ax_fitness.set_xlim(0, max_gen * 1.2)

                    max_fit = max(max(y_best) if y_best else [0], max(y_avg) if y_avg else [0])
                    if max_fit > self.ax_fitness.get_ylim()[1] * 0.8:
                        self.ax_fitness.set_ylim(0, min(1.0, max_fit * 1.2))

            # Update info panel
            self.ax_info.clear()
            self.ax_info.set_xlim(0, 1)
            self.ax_info.set_ylim(0, 1)
            self.ax_info.axis('off')

            target_display = f'"{self.target_token_text}"'
            if self.target_token_id is not None:
                target_display += f" (ID: {self.target_token_id})"

            info_text = f'Base Text: "{self.base_text}"  →  Target: {target_display}\n'
            info_text += f'Baseline Probability: {self.baseline_probability:.4f}'

            if self.baseline_probability > 0 and self.current_probability > 0:
                reduction = (1 - self.current_probability / self.baseline_probability) * 100
                info_text += f'  |  Current Reduction: {reduction:.1f}%'

            self.ax_info.text(0.5, 0.5, info_text, ha='center', va='center',
                             fontsize=12, bbox=dict(boxstyle="round,pad=0.5",
                                                   facecolor="lightblue", alpha=0.8))

            # Update stats panel
            self.ax_stats.clear()
            self.ax_stats.set_xlim(0, 1)
            self.ax_stats.set_ylim(0, 1)
            self.ax_stats.axis('off')

            # Calculate current stats
            reduction_pct = 0.0
            if self.baseline_probability > 0 and self.current_probability > 0:
                reduction_pct = (1 - self.current_probability / self.baseline_probability) * 100

            progress_pct = (self.current_generation / self.max_generations) * 100 if self.max_generations > 0 else 0

            stats_text = f"""Generation: {self.current_generation}

🏆 Best Fitness: {self.current_best_fitness:.4f}
📈 Avg Fitness: {self.current_avg_fitness:.4f}

🎯 Current Probability: {self.current_probability:.4f}
📉 Reduction: {reduction_pct:.1f}%

⏱️ Progress: {progress_pct:.1f}%"""

            color = "lightgreen" if self.current_best_fitness > 0.5 else "lightyellow"

            self.ax_stats.text(0.05, 0.95, stats_text, ha='left', va='top', fontsize=11,
                              bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8))

            # Update tokens panel
            self.ax_tokens.clear()
            self.ax_tokens.set_xlim(0, 1)
            self.ax_tokens.set_ylim(0, 1)
            self.ax_tokens.axis('off')

            if self.current_best_tokens:
                tokens_str = f"Token IDs: {self.current_best_tokens}"

                if self.current_token_texts:
                    # Format token texts nicely
                    texts_display = []
                    for text in self.current_token_texts:
                        # Clean up token text for display
                        clean_text = repr(text) if text else "<?>"
                        texts_display.append(clean_text)

                    tokens_str += f"\nDecoded: {texts_display}"
                else:
                    tokens_str += "\nDecoded: [Token texts not available]"

                # Add fitness info
                tokens_str += f"\nFitness: {self.current_best_fitness:.4f}"

            else:
                tokens_str = "🔄 Initializing population..."

            # Color based on fitness
            if self.current_best_fitness > 0.7:
                token_color = "lightgreen"
            elif self.current_best_fitness > 0.3:
                token_color = "lightyellow"
            else:
                token_color = "lightcoral"

            self.ax_tokens.text(0.5, 0.5, tokens_str, ha='center', va='center',
                               fontsize=10, bbox=dict(boxstyle="round,pad=0.5",
                                                     facecolor=token_color, alpha=0.8))

            # Update display
            plt.draw()
            plt.pause(0.01)  # Small pause to allow GUI updates

        except Exception as e:
            self.logger.warning(f"Error updating display: {e}")

    def start_animation(self):
        """Start the real-time animation."""
        self.is_running = True
        self.logger.info("🎬 Starting real-time genetic algorithm animation...")

    def stop_animation(self):
        """Stop the animation."""
        self.is_running = False

    def mark_complete(self, final_message: str = None):
        """Mark the evolution as complete."""
        self.is_complete = True
        self.is_running = False

        if final_message:
            # Add completion message to the plot
            self.fig.suptitle(f'🧬 Genetic Algorithm Evolution: COMPLETED\n{final_message}',
                             fontsize=16, fontweight='bold', color='green')
            plt.draw()
            plt.pause(0.1)

        self.logger.info("✅ Genetic algorithm evolution completed!")

    def save_animation_data(self, filename: str):
        """Save current animation data for later replay."""
        data = {
            'base_text': self.base_text,
            'target_token_text': self.target_token_text,
            'target_token_id': self.target_token_id,
            'baseline_probability': self.baseline_probability,
            'generations': list(self.generations),
            'best_fitness': list(self.best_fitness),
            'avg_fitness': list(self.avg_fitness),
            'final_tokens': self.current_best_tokens,
            'final_token_texts': self.current_token_texts,
            'final_fitness': self.current_best_fitness
        }

        import json
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"💾 Animation data saved to {filename}")

    def keep_alive(self, duration: float = None):
        """
        Keep the animation window alive for viewing.

        Args:
            duration: How long to keep alive (seconds). None = indefinite
        """
        if not hasattr(self, 'fig') or not plt.fignum_exists(self.fig.number):
            return

        self.logger.info("🖼️  Animation window is live. Close the window or press Ctrl+C to exit.")

        try:
            if duration is None:
                # Keep alive indefinitely
                while plt.fignum_exists(self.fig.number):
                    plt.pause(0.5)
            else:
                # Keep alive for specified duration
                start_time = time.time()
                while plt.fignum_exists(self.fig.number) and (time.time() - start_time) < duration:
                    plt.pause(0.1)

        except KeyboardInterrupt:
            self.logger.info("⏹️  Animation stopped by user.")
        except Exception as e:
            self.logger.warning(f"Animation error: {e}")
        finally:
            plt.ioff()
            if hasattr(self, 'fig') and plt.fignum_exists(self.fig.number):
                plt.close(self.fig)


class GeneticAnimationCallback:
    """
    Callback interface for the genetic algorithm to update the animator.

    This class bridges the genetic algorithm with the real-time animator,
    providing methods that the GA can call to update the visualization.
    """

    def __init__(self, animator: RealTimeGeneticAnimator):
        """Initialize with an animator instance."""
        self.animator = animator
        self.logger = logging.getLogger(__name__)

    def on_evolution_start(self, baseline_prob: float, target_token_id: int = None,
                          target_token_text: str = None):
        """Called when evolution starts."""
        self.animator.update_baseline(baseline_prob, target_token_id, target_token_text)
        self.animator.start_animation()

    def on_generation_complete(self, generation: int, best_individual, avg_fitness: float,
                              current_probability: float = None, tokenizer=None):
        """Called when a generation completes."""
        # Decode tokens if tokenizer is available
        token_texts = None
        if tokenizer and hasattr(best_individual, 'tokens'):
            try:
                token_texts = [tokenizer.decode([token_id]) for token_id in best_individual.tokens]
            except Exception as e:
                self.logger.warning(f"Failed to decode tokens: {e}")

        # Update animator
        best_tokens = getattr(best_individual, 'tokens', [])
        best_fitness = getattr(best_individual, 'fitness', 0.0)

        self.animator.update_data(
            generation=generation,
            best_fitness=best_fitness,
            avg_fitness=avg_fitness,
            best_tokens=best_tokens,
            token_texts=token_texts,
            current_probability=current_probability
        )

    def on_evolution_complete(self, final_population, total_generations: int):
        """Called when evolution completes."""
        if final_population:
            best_individual = final_population[0]
            final_fitness = getattr(best_individual, 'fitness', 0.0)
            final_tokens = getattr(best_individual, 'tokens', [])

            message = f"Best fitness: {final_fitness:.4f} | Best tokens: {final_tokens}"
            self.animator.mark_complete(message)
        else:
            self.animator.mark_complete("Evolution completed")

    def save_results(self, filename: str):
        """Save animation data to file."""
        self.animator.save_animation_data(filename)

    def keep_alive(self, duration: float = None):
        """Keep the animation alive for viewing."""
        self.animator.keep_alive(duration)
