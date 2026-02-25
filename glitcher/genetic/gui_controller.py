"""
GUI Controller for Genetic Algorithm Evolution

This module provides a comprehensive graphical user interface for controlling
and monitoring genetic algorithm evolution experiments with real-time parameter
adjustment and process control.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
import json
import os  # noqa: F401 (used in _run_evolution_with_controls)
from typing import Optional, Callable
from dataclasses import dataclass
import time

# Matplotlib integration for plotting
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Use Tkinter backend
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = None
    FigureCanvasTkAgg = None

from .reducer import GeneticProbabilityReducer
from .gui_animator import EnhancedGeneticAnimator


class GUICallback:
    """
    Callback interface for genetic algorithm integration with GUI.

    This class provides the interface between the genetic algorithm
    and the GUI controller, handling data updates and lifecycle events.
    """

    def __init__(self, gui_controller):
        """
        Initialize the GUI callback.

        Args:
            gui_controller: The GUI controller instance
        """
        self.gui_controller = gui_controller
        self.tokenizer = None

    def on_evolution_start(self, baseline_data):
        """Called when evolution starts."""
        self.gui_controller.baseline_probability = baseline_data.get('target_prob', 0.0)
        self.gui_controller.root.after(0, lambda: self.gui_controller.log_message("Evolution started"))

    def set_tokenizer(self, tokenizer):
        """Set the tokenizer for token decoding."""
        self.tokenizer = tokenizer
        if hasattr(self.gui_controller, 'reducer') and self.gui_controller.reducer:
            self.gui_controller.reducer.tokenizer = tokenizer

    def on_generation_complete(self, generation, population, best_individual, diversity, stagnation):
        """Called when a generation completes."""
        try:
            # Check for pause/stop controls
            import time
            while self.gui_controller.is_paused and not self.gui_controller.should_stop:
                time.sleep(0.1)

            if self.gui_controller.should_stop:
                # Signal to stop evolution - this will cause an exception that stops the loop
                raise KeyboardInterrupt("Evolution stopped by user")

            # Calculate progress percentage
            progress = (generation / self.gui_controller.config.generations) * 100 if self.gui_controller.config.generations > 0 else 0

            # Calculate fitness metrics
            best_fitness = best_individual.fitness if best_individual else 0.0

            # Calculate average fitness from population
            if population and len(population) > 0:
                avg_fitness = sum(ind.fitness for ind in population) / len(population)
            else:
                avg_fitness = 0.0

            # Update progress display in main thread
            def update_gui():
                self.gui_controller._update_progress(generation, progress, best_fitness, avg_fitness, best_individual)

            self.gui_controller.root.after(0, update_gui)

        except KeyboardInterrupt:
            # Re-raise to stop evolution
            raise
        except Exception as e:
            self.gui_controller.root.after(0, lambda e=e: self.gui_controller.log_message(f"Error updating progress: {e}"))

    def on_evolution_complete(self, results):
        """Called when evolution completes."""
        # Convert list results from run_evolution to expected dictionary format
        if isinstance(results, list) and len(results) > 0:
            best_individual = max(results, key=lambda x: x.fitness)
            formatted_results = {
                'best_individual': best_individual,
                'final_population': results,
                'generations_completed': self.gui_controller.config.generations,
                'best_fitness': best_individual.fitness
            }
        else:
            formatted_results = {
                'best_individual': None,
                'final_population': [],
                'generations_completed': 0,
                'best_fitness': 0.0
            }

        self.gui_controller.root.after(0, lambda: self.gui_controller._on_evolution_complete(formatted_results))


@dataclass
class GeneticConfig:
    """Configuration for genetic algorithm parameters"""
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    base_text: str = "The quick brown"
    target_token: str = ""
    wanted_token: str = ""
    token_file: str = "glitch_tokens.json"

    # Evolution parameters
    population_size: int = 50
    generations: int = 100
    max_tokens: int = 3
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_size: int = 5
    early_stopping_threshold: float = 0.999

    # Validation parameters
    asr_threshold: float = 0.5
    num_attempts: int = 3
    validation_tokens: int = 50

    # Mode flags
    ascii_only: bool = True
    enhanced_validation: bool = True
    comprehensive_search: bool = False
    include_normal_tokens: bool = False
    baseline_seeding: bool = True
    baseline_seeding_ratio: float = 0.7
    sequence_diversity: bool = True
    sequence_diversity_ratio: float = 0.3
    exact_token_count: bool = True
    enable_shuffle_mutation: bool = False

    # Output settings
    output_file: str = "genetic_results.json"
    baseline_output: str = "token_impact_baseline.json"
    show_gui_animation: bool = True


class GeneticControllerGUI:
    """Main GUI controller for genetic algorithm evolution"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Glitcher Genetic Algorithm Controller")
        self.root.geometry("1400x900")

        # State management
        self.config = GeneticConfig()
        self.reducer: Optional[GeneticProbabilityReducer] = None
        self.animator: Optional[EnhancedGeneticAnimator] = None
        self.evolution_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.is_paused = False
        self.should_stop = False

        # Data for plotting and visualization
        self.generation_history = []
        self.fitness_history = []
        self.avg_fitness_history = []
        self.current_best_individual = None
        self.baseline_probability = None

        # Matplotlib components
        self.fig = None
        self.ax_fitness = None
        self.canvas = None

        self.setup_gui()
        self.update_gui_from_config()

        # Callbacks
        self.on_generation_complete: Optional[Callable] = None
        self.on_evolution_complete: Optional[Callable] = None

    def setup_gui(self):
        """Create the main GUI layout"""
        # Create main frame with scrollable content
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True)

        # Create tabs
        self.setup_config_tab()
        self.setup_control_tab()
        self.setup_progress_tab()
        self.setup_results_tab()

    def setup_config_tab(self):
        """Setup configuration parameters tab"""
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="Configuration")

        # Create scrollable frame
        canvas = tk.Canvas(config_frame)
        scrollbar = ttk.Scrollbar(config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Model Configuration
        model_group = ttk.LabelFrame(scrollable_frame, text="Model Configuration", padding=10)
        model_group.pack(fill='x', pady=5)

        ttk.Label(model_group, text="Model Name:").grid(row=0, column=0, sticky='w', padx=5)
        self.model_name_var = tk.StringVar(value=self.config.model_name)
        ttk.Entry(model_group, textvariable=self.model_name_var, width=50).grid(row=0, column=1, padx=5)

        ttk.Label(model_group, text="Token File:").grid(row=1, column=0, sticky='w', padx=5)
        self.token_file_var = tk.StringVar(value=self.config.token_file)
        file_frame = ttk.Frame(model_group)
        file_frame.grid(row=1, column=1, sticky='ew', padx=5)
        ttk.Entry(file_frame, textvariable=self.token_file_var, width=40).pack(side='left')
        ttk.Button(file_frame, text="Browse", command=self.browse_token_file).pack(side='right', padx=(5,0))

        # Text Configuration
        text_group = ttk.LabelFrame(scrollable_frame, text="Text Configuration", padding=10)
        text_group.pack(fill='x', pady=5)

        ttk.Label(text_group, text="Base Text:").grid(row=0, column=0, sticky='w', padx=5)
        self.base_text_var = tk.StringVar(value=self.config.base_text)
        ttk.Entry(text_group, textvariable=self.base_text_var, width=50).grid(row=0, column=1, padx=5)

        ttk.Label(text_group, text="Target Token:").grid(row=1, column=0, sticky='w', padx=5)
        self.target_token_var = tk.StringVar(value=self.config.target_token)
        ttk.Entry(text_group, textvariable=self.target_token_var, width=50).grid(row=1, column=1, padx=5)

        ttk.Label(text_group, text="Wanted Token:").grid(row=2, column=0, sticky='w', padx=5)
        self.wanted_token_var = tk.StringVar(value=self.config.wanted_token)
        ttk.Entry(text_group, textvariable=self.wanted_token_var, width=50).grid(row=2, column=1, padx=5)

        # Evolution Parameters
        evolution_group = ttk.LabelFrame(scrollable_frame, text="Evolution Parameters", padding=10)
        evolution_group.pack(fill='x', pady=5)

        # Population size
        ttk.Label(evolution_group, text="Population Size:").grid(row=0, column=0, sticky='w', padx=5)
        self.population_size_var = tk.IntVar(value=self.config.population_size)
        ttk.Spinbox(evolution_group, from_=10, to=200, textvariable=self.population_size_var, width=10).grid(row=0, column=1, padx=5, sticky='w')

        # Generations
        ttk.Label(evolution_group, text="Generations:").grid(row=0, column=2, sticky='w', padx=5)
        self.generations_var = tk.IntVar(value=self.config.generations)
        ttk.Spinbox(evolution_group, from_=10, to=1000, textvariable=self.generations_var, width=10).grid(row=0, column=3, padx=5, sticky='w')

        # Max tokens
        ttk.Label(evolution_group, text="Max Tokens:").grid(row=1, column=0, sticky='w', padx=5)
        self.max_tokens_var = tk.IntVar(value=self.config.max_tokens)
        ttk.Spinbox(evolution_group, from_=1, to=10, textvariable=self.max_tokens_var, width=10).grid(row=1, column=1, padx=5, sticky='w')

        # Mutation rate
        ttk.Label(evolution_group, text="Mutation Rate:").grid(row=1, column=2, sticky='w', padx=5)
        self.mutation_rate_var = tk.DoubleVar(value=self.config.mutation_rate)
        ttk.Spinbox(evolution_group, from_=0.0, to=1.0, increment=0.01, textvariable=self.mutation_rate_var, width=10).grid(row=1, column=3, padx=5, sticky='w')

        # Crossover rate
        ttk.Label(evolution_group, text="Crossover Rate:").grid(row=2, column=0, sticky='w', padx=5)
        self.crossover_rate_var = tk.DoubleVar(value=self.config.crossover_rate)
        ttk.Spinbox(evolution_group, from_=0.0, to=1.0, increment=0.01, textvariable=self.crossover_rate_var, width=10).grid(row=2, column=1, padx=5, sticky='w')

        # Elite size
        ttk.Label(evolution_group, text="Elite Size:").grid(row=2, column=2, sticky='w', padx=5)
        self.elite_size_var = tk.IntVar(value=self.config.elite_size)
        ttk.Spinbox(evolution_group, from_=1, to=20, textvariable=self.elite_size_var, width=10).grid(row=2, column=3, padx=5, sticky='w')

        # Early stopping threshold
        ttk.Label(evolution_group, text="Early Stopping Threshold:").grid(row=3, column=0, sticky='w', padx=5)
        self.early_stopping_var = tk.DoubleVar(value=self.config.early_stopping_threshold)
        ttk.Spinbox(evolution_group, from_=0.0, to=1.0, increment=0.001, textvariable=self.early_stopping_var, width=10).grid(row=3, column=1, padx=5, sticky='w')

        # Validation Parameters
        validation_group = ttk.LabelFrame(scrollable_frame, text="Validation Parameters", padding=10)
        validation_group.pack(fill='x', pady=5)

        ttk.Label(validation_group, text="ASR Threshold:").grid(row=0, column=0, sticky='w', padx=5)
        self.asr_threshold_var = tk.DoubleVar(value=self.config.asr_threshold)
        ttk.Spinbox(validation_group, from_=0.0, to=1.0, increment=0.1, textvariable=self.asr_threshold_var, width=10).grid(row=0, column=1, padx=5, sticky='w')

        ttk.Label(validation_group, text="Num Attempts:").grid(row=0, column=2, sticky='w', padx=5)
        self.num_attempts_var = tk.IntVar(value=self.config.num_attempts)
        ttk.Spinbox(validation_group, from_=1, to=10, textvariable=self.num_attempts_var, width=10).grid(row=0, column=3, padx=5, sticky='w')

        ttk.Label(validation_group, text="Validation Tokens:").grid(row=1, column=0, sticky='w', padx=5)
        self.validation_tokens_var = tk.IntVar(value=self.config.validation_tokens)
        ttk.Spinbox(validation_group, from_=10, to=1000, textvariable=self.validation_tokens_var, width=10).grid(row=1, column=1, padx=5, sticky='w')

        # Mode Flags
        flags_group = ttk.LabelFrame(scrollable_frame, text="Mode Flags", padding=10)
        flags_group.pack(fill='x', pady=5)

        self.ascii_only_var = tk.BooleanVar(value=self.config.ascii_only)
        ttk.Checkbutton(flags_group, text="ASCII Only", variable=self.ascii_only_var).grid(row=0, column=0, sticky='w', padx=5)

        self.enhanced_validation_var = tk.BooleanVar(value=self.config.enhanced_validation)
        ttk.Checkbutton(flags_group, text="Enhanced Validation", variable=self.enhanced_validation_var).grid(row=0, column=1, sticky='w', padx=5)

        self.comprehensive_search_var = tk.BooleanVar(value=self.config.comprehensive_search)
        ttk.Checkbutton(flags_group, text="Comprehensive Search", variable=self.comprehensive_search_var).grid(row=0, column=2, sticky='w', padx=5)

        self.include_normal_tokens_var = tk.BooleanVar(value=self.config.include_normal_tokens)
        ttk.Checkbutton(flags_group, text="Include Normal Tokens", variable=self.include_normal_tokens_var).grid(row=1, column=0, sticky='w', padx=5)

        self.baseline_seeding_var = tk.BooleanVar(value=self.config.baseline_seeding)
        ttk.Checkbutton(flags_group, text="Baseline Seeding", variable=self.baseline_seeding_var).grid(row=1, column=1, sticky='w', padx=5)

        self.sequence_diversity_var = tk.BooleanVar(value=self.config.sequence_diversity)
        ttk.Checkbutton(flags_group, text="Sequence Diversity", variable=self.sequence_diversity_var).grid(row=1, column=2, sticky='w', padx=5)

        self.exact_token_count_var = tk.BooleanVar(value=self.config.exact_token_count)
        ttk.Checkbutton(flags_group, text="Exact Token Count", variable=self.exact_token_count_var).grid(row=2, column=0, sticky='w', padx=5)

        self.show_gui_animation_var = tk.BooleanVar(value=self.config.show_gui_animation)
        ttk.Checkbutton(flags_group, text="Show GUI Animation", variable=self.show_gui_animation_var).grid(row=2, column=1, sticky='w', padx=5)

        # Advanced Parameters
        advanced_group = ttk.LabelFrame(scrollable_frame, text="Advanced Parameters", padding=10)
        advanced_group.pack(fill='x', pady=5)

        ttk.Label(advanced_group, text="Baseline Seeding Ratio:").grid(row=0, column=0, sticky='w', padx=5)
        self.baseline_seeding_ratio_var = tk.DoubleVar(value=self.config.baseline_seeding_ratio)
        ttk.Spinbox(advanced_group, from_=0.0, to=1.0, increment=0.1, textvariable=self.baseline_seeding_ratio_var, width=10).grid(row=0, column=1, padx=5, sticky='w')

        ttk.Label(advanced_group, text="Sequence Diversity Ratio:").grid(row=0, column=2, sticky='w', padx=5)
        self.sequence_diversity_ratio_var = tk.DoubleVar(value=self.config.sequence_diversity_ratio)
        ttk.Spinbox(advanced_group, from_=0.0, to=1.0, increment=0.1, textvariable=self.sequence_diversity_ratio_var, width=10).grid(row=0, column=3, padx=5, sticky='w')

        # Configuration buttons
        config_buttons = ttk.Frame(scrollable_frame)
        config_buttons.pack(fill='x', pady=10)

        ttk.Button(config_buttons, text="Save Configuration", command=self.save_config).pack(side='left', padx=5)
        ttk.Button(config_buttons, text="Load Configuration", command=self.load_config).pack(side='left', padx=5)
        ttk.Button(config_buttons, text="Reset to Defaults", command=self.reset_config).pack(side='left', padx=5)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def setup_control_tab(self):
        """Setup evolution control tab"""
        control_frame = ttk.Frame(self.notebook)
        self.notebook.add(control_frame, text="Control")

        # Status display
        status_group = ttk.LabelFrame(control_frame, text="Status", padding=10)
        status_group.pack(fill='x', pady=5)

        self.status_label = ttk.Label(status_group, text="Ready", font=('Arial', 12, 'bold'))
        self.status_label.pack()

        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=20)

        self.start_button = ttk.Button(button_frame, text="Start Evolution",
                                      command=self.start_evolution, style='Green.TButton')
        self.start_button.pack(side='left', padx=10)

        self.pause_button = ttk.Button(button_frame, text="Pause",
                                      command=self.pause_evolution, state='disabled')
        self.pause_button.pack(side='left', padx=10)

        self.stop_button = ttk.Button(button_frame, text="Stop",
                                     command=self.stop_evolution, state='disabled', style='Red.TButton')
        self.stop_button.pack(side='left', padx=10)

        # Progress display
        progress_group = ttk.LabelFrame(control_frame, text="Progress", padding=10)
        progress_group.pack(fill='x', pady=10)

        self.progress_var = tk.StringVar(value="Generation: 0/0")
        ttk.Label(progress_group, textvariable=self.progress_var).pack()

        self.progress_bar = ttk.Progressbar(progress_group, mode='determinate')
        self.progress_bar.pack(fill='x', pady=5)

        # Current best display
        best_group = ttk.LabelFrame(control_frame, text="Current Best", padding=10)
        best_group.pack(fill='both', expand=True, pady=5)

        self.best_text = scrolledtext.ScrolledText(best_group, height=10, state='disabled')
        self.best_text.pack(fill='both', expand=True)

        # Setup button styles
        style = ttk.Style()
        style.configure('Green.TButton', foreground='green')
        style.configure('Red.TButton', foreground='red')

    def setup_progress_tab(self):
        """Setup progress monitoring tab with enhanced visualization"""
        progress_frame = ttk.Frame(self.notebook)
        self.notebook.add(progress_frame, text="Progress")

        # Create main paned window for layout
        main_paned = ttk.PanedWindow(progress_frame, orient='horizontal')
        main_paned.pack(fill='both', expand=True, padx=5, pady=5)

        # Left panel for metrics and plots
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=3)

        # Real-time metrics
        metrics_group = ttk.LabelFrame(left_frame, text="Real-time Metrics", padding=10)
        metrics_group.pack(fill='x', pady=5)

        metrics_frame = ttk.Frame(metrics_group)
        metrics_frame.pack(fill='x')

        # Current generation
        ttk.Label(metrics_frame, text="Generation:").grid(row=0, column=0, sticky='w', padx=5)
        self.current_generation_var = tk.StringVar(value="0")
        ttk.Label(metrics_frame, textvariable=self.current_generation_var, font=('Arial', 10, 'bold')).grid(row=0, column=1, sticky='w', padx=5)

        # Best fitness
        ttk.Label(metrics_frame, text="Best Fitness:").grid(row=0, column=2, sticky='w', padx=5)
        self.best_fitness_var = tk.StringVar(value="0.000")
        ttk.Label(metrics_frame, textvariable=self.best_fitness_var, font=('Arial', 10, 'bold')).grid(row=0, column=3, sticky='w', padx=5)

        # Average fitness
        ttk.Label(metrics_frame, text="Avg Fitness:").grid(row=1, column=0, sticky='w', padx=5)
        self.avg_fitness_var = tk.StringVar(value="0.000")
        ttk.Label(metrics_frame, textvariable=self.avg_fitness_var).grid(row=1, column=1, sticky='w', padx=5)

        # Probability reduction
        ttk.Label(metrics_frame, text="Reduction:").grid(row=1, column=2, sticky='w', padx=5)
        self.reduction_var = tk.StringVar(value="0.0%")
        ttk.Label(metrics_frame, textvariable=self.reduction_var, font=('Arial', 10, 'bold')).grid(row=1, column=3, sticky='w', padx=5)

        # Fitness evolution plot
        if MATPLOTLIB_AVAILABLE:
            plot_group = ttk.LabelFrame(left_frame, text="Fitness Evolution", padding=10)
            plot_group.pack(fill='both', expand=True, pady=5)

            self.fig = Figure(figsize=(8, 4), dpi=100)
            self.ax_fitness = self.fig.add_subplot(111)
            self.ax_fitness.set_title('Fitness Evolution Over Generations')
            self.ax_fitness.set_xlabel('Generation')
            self.ax_fitness.set_ylabel('Fitness')
            self.ax_fitness.grid(True, alpha=0.3)

            self.canvas = FigureCanvasTkAgg(self.fig, plot_group)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill='both', expand=True)
        else:
            # Fallback when matplotlib is not available
            plot_group = ttk.LabelFrame(left_frame, text="Fitness Evolution (matplotlib not available)", padding=10)
            plot_group.pack(fill='both', expand=True, pady=5)

            warning_text = scrolledtext.ScrolledText(plot_group, height=6, state='disabled',
                                                   font=('Courier', 9), bg='#f0f0f0')
            warning_text.pack(fill='both', expand=True)

            warning_text.config(state='normal')
            warning_text.insert(1.0, "⚠️  Matplotlib not available\n\n")
            warning_text.insert(tk.END, "To see fitness evolution plots, install matplotlib:\n")
            warning_text.insert(tk.END, "pip install matplotlib\n\n")
            warning_text.insert(tk.END, "Fitness data will still be shown in the metrics above.")
            warning_text.config(state='disabled')

        # Right panel for detailed information
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)

        # Best individual details
        best_group = ttk.LabelFrame(right_frame, text="Best Individual", padding=10)
        best_group.pack(fill='x', pady=5)

        self.best_text = scrolledtext.ScrolledText(best_group, height=8, state='disabled', font=('Courier', 9))
        self.best_text.pack(fill='x')

        # Context strings display
        context_group = ttk.LabelFrame(right_frame, text="Input Strings & Context", padding=10)
        context_group.pack(fill='both', expand=True, pady=5)

        self.context_text = scrolledtext.ScrolledText(context_group, state='disabled', font=('Courier', 9))
        self.context_text.pack(fill='both', expand=True)

        # Log display
        log_group = ttk.LabelFrame(right_frame, text="Evolution Log", padding=10)
        log_group.pack(fill='both', expand=True, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_group, height=6, state='disabled', font=('Courier', 8))
        self.log_text.pack(fill='both', expand=True)

    def setup_results_tab(self):
        """Setup results display tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")

        # Results display
        self.results_text = scrolledtext.ScrolledText(results_frame, state='disabled')
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)

        # Results buttons
        results_buttons = ttk.Frame(results_frame)
        results_buttons.pack(fill='x', padx=10, pady=5)

        ttk.Button(results_buttons, text="Save Results", command=self.save_results).pack(side='left', padx=5)
        ttk.Button(results_buttons, text="Clear Results", command=self.clear_results).pack(side='left', padx=5)

    def update_gui_from_config(self):
        """Update GUI elements from current configuration"""
        self.model_name_var.set(self.config.model_name)
        self.base_text_var.set(self.config.base_text)
        self.target_token_var.set(self.config.target_token)
        self.wanted_token_var.set(self.config.wanted_token)
        self.token_file_var.set(self.config.token_file)

        self.population_size_var.set(self.config.population_size)
        self.generations_var.set(self.config.generations)
        self.max_tokens_var.set(self.config.max_tokens)
        self.mutation_rate_var.set(self.config.mutation_rate)
        self.crossover_rate_var.set(self.config.crossover_rate)
        self.elite_size_var.set(self.config.elite_size)
        self.early_stopping_var.set(self.config.early_stopping_threshold)

        self.asr_threshold_var.set(self.config.asr_threshold)
        self.num_attempts_var.set(self.config.num_attempts)
        self.validation_tokens_var.set(self.config.validation_tokens)

        self.ascii_only_var.set(self.config.ascii_only)
        self.enhanced_validation_var.set(self.config.enhanced_validation)
        self.comprehensive_search_var.set(self.config.comprehensive_search)
        self.include_normal_tokens_var.set(self.config.include_normal_tokens)
        self.baseline_seeding_var.set(self.config.baseline_seeding)
        self.sequence_diversity_var.set(self.config.sequence_diversity)
        self.exact_token_count_var.set(self.config.exact_token_count)
        self.show_gui_animation_var.set(self.config.show_gui_animation)

        self.baseline_seeding_ratio_var.set(self.config.baseline_seeding_ratio)
        self.sequence_diversity_ratio_var.set(self.config.sequence_diversity_ratio)

    def update_config_from_gui(self):
        """Update configuration from GUI elements"""
        self.config.model_name = self.model_name_var.get()
        self.config.base_text = self.base_text_var.get()
        self.config.target_token = self.target_token_var.get()
        self.config.wanted_token = self.wanted_token_var.get()
        self.config.token_file = self.token_file_var.get()

        self.config.population_size = self.population_size_var.get()
        self.config.generations = self.generations_var.get()
        self.config.max_tokens = self.max_tokens_var.get()
        self.config.mutation_rate = self.mutation_rate_var.get()
        self.config.crossover_rate = self.crossover_rate_var.get()
        self.config.elite_size = self.elite_size_var.get()
        self.config.early_stopping_threshold = self.early_stopping_var.get()

        self.config.asr_threshold = self.asr_threshold_var.get()
        self.config.num_attempts = self.num_attempts_var.get()
        self.config.validation_tokens = self.validation_tokens_var.get()

        self.config.ascii_only = self.ascii_only_var.get()
        self.config.enhanced_validation = self.enhanced_validation_var.get()
        self.config.comprehensive_search = self.comprehensive_search_var.get()
        self.config.include_normal_tokens = self.include_normal_tokens_var.get()
        self.config.baseline_seeding = self.baseline_seeding_var.get()
        self.config.sequence_diversity = self.sequence_diversity_var.get()
        self.config.exact_token_count = self.exact_token_count_var.get()
        self.config.show_gui_animation = self.show_gui_animation_var.get()

        self.config.baseline_seeding_ratio = self.baseline_seeding_ratio_var.get()
        self.config.sequence_diversity_ratio = self.sequence_diversity_ratio_var.get()

    def browse_token_file(self):
        """Browse for token file"""
        filename = filedialog.askopenfilename(
            title="Select Token File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.token_file_var.set(filename)

    def save_config(self):
        """Save current configuration to file"""
        self.update_config_from_gui()
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                config_dict = {
                    k: v for k, v in self.config.__dict__.items()
                }
                with open(filename, 'w') as f:
                    json.dump(config_dict, f, indent=2)
                messagebox.showinfo("Success", f"Configuration saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def load_config(self):
        """Load configuration from file"""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    config_dict = json.load(f)

                # Update config object
                for key, value in config_dict.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)

                self.update_gui_from_config()
                messagebox.showinfo("Success", f"Configuration loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")

    def reset_config(self):
        """Reset configuration to defaults"""
        if messagebox.askyesno("Confirm Reset", "Reset all parameters to default values?"):
            self.config = GeneticConfig()
            self.update_gui_from_config()

    def start_evolution(self):
        """Start the genetic algorithm evolution"""
        if self.is_running:
            return

        try:
            self.update_config_from_gui()
            self.should_stop = False
            self.is_paused = False

            # Update UI state
            self.start_button.config(state='disabled')
            self.pause_button.config(state='normal')
            self.stop_button.config(state='normal')
            self.status_label.config(text="Initializing...")

            # Start evolution in separate thread
            self.evolution_thread = threading.Thread(target=self._run_evolution, daemon=True)
            self.evolution_thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start evolution: {e}")
            self._reset_ui_state()

    def pause_evolution(self):
        """Pause/resume the evolution"""
        if not self.is_running:
            return

        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_button.config(text="Resume")
            self.status_label.config(text="Paused")
            self.log_message("Evolution paused")
        else:
            self.pause_button.config(text="Pause")
            self.status_label.config(text="Running")
            self.log_message("Evolution resumed")

    def stop_evolution(self):
        """Stop the evolution"""
        if not self.is_running:
            return

        if messagebox.askyesno("Confirm Stop", "Stop the current evolution?"):
            self.should_stop = True
            self.status_label.config(text="Stopping...")
            self.log_message("Stop requested, waiting for current generation to complete...")

    def _reset_ui_state(self):
        """Reset UI to initial state"""
        self.is_running = False
        self.is_paused = False
        self.should_stop = False

        self.start_button.config(state='normal')
        self.pause_button.config(state='disabled', text="Pause")
        self.stop_button.config(state='disabled')
        self.status_label.config(text="Ready")

        self.progress_bar['value'] = 0
        self.progress_var.set("Generation: 0/0")

        # Reset plotting data
        self.generation_history = []
        self.fitness_history = []
        self.avg_fitness_history = []
        self.current_best_individual = None
        self.baseline_probability = None

        # Reset metric displays
        self.current_generation_var.set("0")
        self.best_fitness_var.set("0.000")
        self.avg_fitness_var.set("0.000")
        self.reduction_var.set("0.0%")

        # Clear matplotlib plot
        if MATPLOTLIB_AVAILABLE and self.ax_fitness:
            self.ax_fitness.clear()
            self.ax_fitness.set_title('Fitness Evolution Over Generations')
            self.ax_fitness.set_xlabel('Generation')
            self.ax_fitness.set_ylabel('Fitness')
            self.ax_fitness.grid(True, alpha=0.3)
            if self.canvas:
                try:
                    self.canvas.draw()
                except RuntimeError:
                    # Ignore threading errors during canvas draw
                    pass

        # Clear text displays
        if hasattr(self, 'best_text'):
            self.best_text.config(state='normal')
            self.best_text.delete(1.0, tk.END)
            self.best_text.insert(1.0, "No evolution data yet...")
            self.best_text.config(state='disabled')

        if hasattr(self, 'context_text'):
            self.context_text.config(state='normal')
            self.context_text.delete(1.0, tk.END)
            self.context_text.insert(1.0, "Evolution context will appear here...")
            self.context_text.config(state='disabled')

    def _run_evolution(self):
        """Run the genetic algorithm evolution in a separate thread"""
        try:
            self.is_running = True
            self.root.after(0, lambda: self.status_label.config(text="Loading model..."))
            self.root.after(0, lambda: self.log_message("Starting evolution..."))

            # Initialize genetic reducer with basic parameters
            self.reducer = GeneticProbabilityReducer(
                model_name=self.config.model_name,
                base_text=self.config.base_text,
                target_token=self.config.target_token if self.config.target_token else None,
                wanted_token=self.config.wanted_token if self.config.wanted_token else None
            )

            # Set additional parameters as attributes
            self.reducer.population_size = self.config.population_size
            self.reducer.max_generations = self.config.generations
            self.reducer.max_tokens_per_individual = self.config.max_tokens
            self.reducer.mutation_rate = self.config.mutation_rate
            self.reducer.crossover_rate = self.config.crossover_rate
            self.reducer.elite_size = self.config.elite_size
            self.reducer.early_stopping_threshold = self.config.early_stopping_threshold
            self.reducer.use_baseline_seeding = self.config.baseline_seeding
            self.reducer.baseline_seeding_ratio = self.config.baseline_seeding_ratio
            self.reducer.use_sequence_aware_diversity = self.config.sequence_diversity
            self.reducer.sequence_diversity_ratio = self.config.sequence_diversity_ratio
            self.reducer.use_exact_token_count = self.config.exact_token_count
            self.reducer.enable_shuffle_mutation = self.config.enable_shuffle_mutation

            # Set additional parameters as attributes (if supported)
            if hasattr(self.reducer, 'token_file'):
                self.reducer.token_file = self.config.token_file
            if hasattr(self.reducer, 'ascii_only'):
                self.reducer.ascii_only = self.config.ascii_only
            if hasattr(self.reducer, 'enhanced_validation'):
                self.reducer.enhanced_validation = self.config.enhanced_validation
            if hasattr(self.reducer, 'comprehensive_search'):
                self.reducer.comprehensive_search = self.config.comprehensive_search
            if hasattr(self.reducer, 'include_normal_tokens'):
                self.reducer.include_normal_tokens = self.config.include_normal_tokens
            if hasattr(self.reducer, 'output_file'):
                self.reducer.output_file = self.config.output_file
            if hasattr(self.reducer, 'baseline_output'):
                self.reducer.baseline_output = self.config.baseline_output

            # Setup GUI callback for evolution updates
            gui_callback = GUICallback(self)
            if hasattr(self.reducer, 'gui_callback'):
                self.reducer.gui_callback = gui_callback

            # Disable separate animation to avoid threading conflicts with GUI
            # The GUI has its own integrated visualization
            self.animator = None

            self.root.after(0, lambda: self.status_label.config(text="Running"))
            self.root.after(0, lambda: self.log_message("Evolution started"))

            # Run evolution with pause/stop checking
            results = self._run_evolution_with_controls()

            # Handle completion
            self.root.after(0, lambda: self._on_evolution_complete(results))

        except Exception as e:
            self.root.after(0, lambda e=e: self._on_evolution_error(e))

    def _run_evolution_with_controls(self):
        """Run evolution using the reducer's built-in run_evolution method"""
        if not self.reducer:
            raise ValueError("Reducer not initialized")

        self.root.after(0, lambda: self.log_message("Loading model..."))
        self.reducer.load_model()

        # Load glitch tokens from file
        self.root.after(0, lambda: self.log_message(f"Loading glitch tokens from: {self.config.token_file}"))
        if hasattr(self.reducer, 'load_glitch_tokens'):
            self.reducer.load_glitch_tokens(
                token_file=self.config.token_file,
                ascii_only=self.config.ascii_only
            )
        else:
            self.root.after(0, lambda: self.log_message("Warning: load_glitch_tokens method not available"))

        # Use the reducer's built-in evolution method which handles callbacks properly
        self.root.after(0, lambda: self.log_message("Starting evolution..."))

        # Disable tqdm progress bars for better GUI integration
        import os
        original_tqdm_disable = os.environ.get('TQDM_DISABLE', '')
        os.environ['TQDM_DISABLE'] = '1'

        try:
            # The reducer's run_evolution method handles all the evolution logic
            # and will call our GUI callback for progress updates
            results = self.reducer.run_evolution()

            # Process results
            if results and len(results) > 0:
                best_individual = max(results, key=lambda x: x.fitness)
                return {
                    'best_individual': best_individual,
                    'final_population': results,
                    'generations_completed': self.config.generations,
                    'best_fitness': best_individual.fitness
                }
            else:
                return {
                    'best_individual': None,
                    'final_population': [],
                    'generations_completed': 0,
                    'best_fitness': 0.0
                }

        except KeyboardInterrupt:
            self.root.after(0, lambda: self.log_message("Evolution stopped by user"))
            return {
                'best_individual': None,
                'final_population': [],
                'generations_completed': 0,
                'best_fitness': 0.0,
                'stopped_early': True
            }
        except Exception as e:
            self.root.after(0, lambda e=e: self.log_message(f"Evolution error: {e}"))
            raise e
        finally:
            # Restore original tqdm setting
            if original_tqdm_disable:
                os.environ['TQDM_DISABLE'] = original_tqdm_disable
            else:
                os.environ.pop('TQDM_DISABLE', None)

    def _on_generation_update(self, generation, population, best_individual):
        """Legacy callback for generation updates - now handled by GUICallback"""
        # This method is kept for compatibility but the new GUICallback class
        # handles the actual updates through on_generation_complete
        pass

    def _update_progress(self, generation, progress, best_fitness, avg_fitness, best_individual):
        """Update progress display with enhanced visualization"""
        self.progress_bar['value'] = progress
        self.progress_var.set(f"Generation: {generation}/{self.config.generations}")
        self.current_generation_var.set(str(generation))
        self.best_fitness_var.set(f"{best_fitness:.3f}")
        self.avg_fitness_var.set(f"{avg_fitness:.3f}")

        # Store data for plotting
        self.generation_history.append(generation)
        self.fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        self.current_best_individual = best_individual

        if best_individual:
            reduction = best_fitness * 100
            self.reduction_var.set(f"{reduction:.1f}%")

            # Update best individual display with detailed information
            self._update_best_individual_display(best_individual, reduction)

            # Update context display
            self._update_context_display(best_individual)

        # Update fitness plot
        self._update_fitness_plot()

    def _update_best_individual_display(self, best_individual, reduction):
        """Update the detailed best individual display"""
        self.best_text.config(state='normal')
        self.best_text.delete(1.0, tk.END)

        display_text = f"🏆 Best Individual Details:\n\n"
        display_text += f"Token IDs: {best_individual.tokens}\n\n"

        if hasattr(self.reducer, 'tokenizer') and self.reducer.tokenizer:
            try:
                token_texts = [self.reducer.tokenizer.decode([token_id]) for token_id in best_individual.tokens]
                display_text += f"Token Texts: {token_texts}\n\n"
                # Show individual token breakdown
                display_text += "Token Breakdown:\n"
                for i, (token_id, token_text) in enumerate(zip(best_individual.tokens, token_texts)):
                    display_text += f"  {i+1}. '{token_text}' (ID: {token_id})\n"
                display_text += "\n"
            except Exception as e:
                display_text += f"Token decode error: {e}\n\n"

        display_text += f"Fitness Score: {best_individual.fitness:.6f}\n"
        display_text += f"Probability Reduction: {reduction:.2f}%\n\n"

        # Show probability transformation if available
        if hasattr(best_individual, 'target_prob_before') and hasattr(best_individual, 'target_prob_after'):
            display_text += f"Probability Change:\n"
            display_text += f"  Before: {best_individual.target_prob_before:.6f}\n"
            display_text += f"  After:  {best_individual.target_prob_after:.6f}\n"
            display_text += f"  Change: {(best_individual.target_prob_before - best_individual.target_prob_after):.6f}"

        self.best_text.insert(1.0, display_text)
        self.best_text.config(state='disabled')

    def _update_context_display(self, best_individual):
        """Update the context and input strings display"""
        self.context_text.config(state='normal')
        self.context_text.delete(1.0, tk.END)

        context_text = "📝 Input Strings for LLM:\n\n"

        # Show baseline input
        baseline_text = self.config.base_text
        context_text += f"BASELINE INPUT:\n'{baseline_text}'\n\n"

        # Show evolved input if we have tokenizer
        if hasattr(self.reducer, 'tokenizer') and self.reducer.tokenizer and best_individual:
            try:
                token_texts = [self.reducer.tokenizer.decode([token_id]) for token_id in best_individual.tokens]
                evolved_prefix = "".join(token_texts)
                full_evolved_string = f"{evolved_prefix}{baseline_text}"
                context_text += f"EVOLVED INPUT:\n'{full_evolved_string}'\n\n"
                context_text += f"Token Prefix: '{evolved_prefix}'\n"
                context_text += f"Base Text: '{baseline_text}'\n\n"
            except Exception as e:
                context_text += f"Error constructing evolved string: {e}\n\n"

        # Show target/wanted token info
        if self.config.target_token:
            context_text += f"🎯 Target Token: '{self.config.target_token}'\n"
        if self.config.wanted_token:
            context_text += f"⭐ Wanted Token: '{self.config.wanted_token}'\n"

        # Show baseline probability if available
        if self.baseline_probability is not None:
            context_text += f"\n📊 Baseline Probability: {self.baseline_probability:.6f}\n"

        self.context_text.insert(1.0, context_text)
        self.context_text.config(state='disabled')

    def _update_fitness_plot(self):
        """Update the fitness evolution plot"""
        if not MATPLOTLIB_AVAILABLE or not self.ax_fitness:
            return

        try:
            self.ax_fitness.clear()
            self.ax_fitness.set_title('Fitness Evolution Over Generations')
            self.ax_fitness.set_xlabel('Generation')
            self.ax_fitness.set_ylabel('Fitness')
            self.ax_fitness.grid(True, alpha=0.3)

            if len(self.generation_history) > 1:
                # Plot best fitness
                self.ax_fitness.plot(self.generation_history, self.fitness_history,
                                   'b-', linewidth=2, label='Best Fitness', marker='o', markersize=3)

                # Plot average fitness
                self.ax_fitness.plot(self.generation_history, self.avg_fitness_history,
                                   'r--', linewidth=1, label='Avg Fitness', alpha=0.7)

                self.ax_fitness.legend()

                # Set reasonable axis limits
                if self.fitness_history:
                    max_fitness = max(self.fitness_history)
                    min_fitness = min(self.fitness_history)
                    margin = (max_fitness - min_fitness) * 0.1 if max_fitness > min_fitness else 0.1
                    self.ax_fitness.set_ylim(min_fitness - margin, max_fitness + margin)

            if self.canvas:
                try:
                    self.canvas.draw()
                except RuntimeError:
                    # Ignore threading errors during canvas draw
                    pass
        except Exception as e:
            print(f"Error updating fitness plot: {e}")

    def _on_evolution_complete(self, results):
        """Handle evolution completion"""
        self.status_label.config(text="Complete")
        self.log_message("Evolution completed successfully")

        # Display final results
        self._display_final_results(results)

        # Reset UI state
        self._reset_ui_state()

        # Animator disabled to prevent threading conflicts with GUI
        # Results are displayed in the integrated GUI instead
        pass

    def _on_evolution_error(self, error):
        """Handle evolution error"""
        self.status_label.config(text="Error")
        self.log_message(f"Evolution failed: {error}")
        messagebox.showerror("Evolution Error", f"Evolution failed: {error}")
        self._reset_ui_state()

    def _display_final_results(self, results):
        """Display final results in results tab"""
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)

        # Handle both dictionary and direct results
        if results is None or (isinstance(results, list) and len(results) == 0):
            self.results_text.insert(tk.END, "❌ No results available - evolution may have been stopped early.\n")
            self.results_text.config(state='disabled')
            return

        # Extract results safely
        best_individual = results.get('best_individual') if isinstance(results, dict) else None
        generations_completed = results.get('generations_completed', 0) if isinstance(results, dict) else 0
        final_population = results.get('final_population', []) if isinstance(results, dict) else []

        self.results_text.insert(tk.END, "🏆 EVOLUTION RESULTS\n")
        self.results_text.insert(tk.END, "=" * 50 + "\n\n")

        self.results_text.insert(tk.END, f"Configuration:\n")
        self.results_text.insert(tk.END, f"  Model: {self.config.model_name}\n")
        self.results_text.insert(tk.END, f"  Base Text: '{self.config.base_text}'\n")
        self.results_text.insert(tk.END, f"  Target Token: '{self.config.target_token}'\n")
        self.results_text.insert(tk.END, f"  Wanted Token: '{self.config.wanted_token}'\n")
        self.results_text.insert(tk.END, f"  Generations: {generations_completed}/{self.config.generations}\n\n")

        if best_individual:
            self.results_text.insert(tk.END, f"Best Individual:\n")
            self.results_text.insert(tk.END, f"  Token IDs: {best_individual.tokens}\n")

            if hasattr(self.reducer, 'tokenizer') and self.reducer.tokenizer:
                try:
                    token_texts = [self.reducer.tokenizer.decode([token_id]) for token_id in best_individual.tokens]
                    self.results_text.insert(tk.END, f"  Token Texts: {token_texts}\n")

                    # Show evolved string construction
                    evolved_prefix = "".join(token_texts)
                    full_evolved_string = f"{evolved_prefix}{self.config.base_text}"
                    self.results_text.insert(tk.END, f"  Evolved String: '{full_evolved_string}'\n")
                except Exception as e:
                    self.results_text.insert(tk.END, f"  Token decode error: {e}\n")

            self.results_text.insert(tk.END, f"  Fitness Score: {best_individual.fitness:.6f}\n")

            # Show probability changes if available
            if hasattr(best_individual, 'target_prob_before') and hasattr(best_individual, 'target_prob_after'):
                reduction = (best_individual.target_prob_before - best_individual.target_prob_after) / best_individual.target_prob_before * 100 if best_individual.target_prob_before > 0 else 0
                self.results_text.insert(tk.END, f"  Probability Reduction: {reduction:.2f}%\n")
                self.results_text.insert(tk.END, f"  Before: {best_individual.target_prob_before:.6f} → After: {best_individual.target_prob_after:.6f}\n")
            else:
                reduction = best_individual.fitness * 100
                self.results_text.insert(tk.END, f"  Probability Reduction: {reduction:.2f}%\n")

            # Show wanted token improvements if available
            if hasattr(best_individual, 'wanted_prob_before') and hasattr(best_individual, 'wanted_prob_after'):
                wanted_increase = (best_individual.wanted_prob_after - best_individual.wanted_prob_before) / (1.0 - best_individual.wanted_prob_before) * 100 if best_individual.wanted_prob_before < 1.0 else 0
                self.results_text.insert(tk.END, f"  Wanted Token Increase: {wanted_increase:.2f}%\n")
                self.results_text.insert(tk.END, f"  Wanted Before: {best_individual.wanted_prob_before:.6f} → After: {best_individual.wanted_prob_after:.6f}\n")

        else:
            self.results_text.insert(tk.END, "❌ No best individual found\n")

        # Show population statistics
        if final_population:
            self.results_text.insert(tk.END, f"\nPopulation Statistics:\n")
            self.results_text.insert(tk.END, f"  Final Population Size: {len(final_population)}\n")
            fitnesses = [ind.fitness for ind in final_population if hasattr(ind, 'fitness')]
            if fitnesses:
                avg_fitness = sum(fitnesses) / len(fitnesses)
                max_fitness = max(fitnesses)
                min_fitness = min(fitnesses)
                self.results_text.insert(tk.END, f"  Average Fitness: {avg_fitness:.6f}\n")
                self.results_text.insert(tk.END, f"  Max Fitness: {max_fitness:.6f}\n")
                self.results_text.insert(tk.END, f"  Min Fitness: {min_fitness:.6f}\n")

        else:
            self.results_text.insert(tk.END, "❌ No best individual found - evolution may have stopped early.\n")

        # Show baseline probability if available
        baseline_prob = results.get('baseline_probability', self.baseline_probability)
        if baseline_prob and baseline_prob > 0:
            self.results_text.insert(tk.END, f"\nBaseline Information:\n")
            self.results_text.insert(tk.END, f"  Baseline Probability: {baseline_prob:.6f}\n")
            if best_individual and hasattr(best_individual, 'fitness'):
                try:
                    final_prob = baseline_prob * (1 - best_individual.fitness)
                    self.results_text.insert(tk.END, f"  Estimated Final Probability: {final_prob:.6f}\n")
                except (TypeError, ValueError):
                    pass

        self.results_text.config(state='disabled')

        # Switch to results tab
        self.notebook.select(3)  # Results tab index

    def log_message(self, message):
        """Add message to log display"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def save_results(self):
        """Save current results to file"""
        if not hasattr(self, 'reducer') or not self.reducer:
            messagebox.showwarning("No Results", "No results to save. Run evolution first.")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                results_content = self.results_text.get(1.0, tk.END)
                with open(filename, 'w') as f:
                    f.write(results_content)
                messagebox.showinfo("Success", f"Results saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {e}")

    def clear_results(self):
        """Clear results display"""
        if messagebox.askyesno("Clear Results", "Clear all results?"):
            self.results_text.config(state='normal')
            self.results_text.delete(1.0, tk.END)
            self.results_text.config(state='disabled')

            self.log_text.config(state='normal')
            self.log_text.delete(1.0, tk.END)
            self.log_text.config(state='disabled')


def main():
    """Main entry point for the GUI controller"""
    root = tk.Tk()
    app = GeneticControllerGUI(root)

    # Handle window close
    def on_closing():
        if app.is_running:
            if messagebox.askyesno("Quit", "Evolution is running. Stop and quit?"):
                app.should_stop = True
                if app.evolution_thread and app.evolution_thread.is_alive():
                    app.evolution_thread.join(timeout=2.0)
                root.destroy()
        else:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    main()
