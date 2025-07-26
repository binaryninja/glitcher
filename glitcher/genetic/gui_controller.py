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
import os
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import time

from .reducer import GeneticProbabilityReducer
from .gui_animator import EnhancedGeneticAnimator, GeneticAnimationCallback


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
        self.root.geometry("1200x800")

        # State management
        self.config = GeneticConfig()
        self.reducer: Optional[GeneticProbabilityReducer] = None
        self.animator: Optional[EnhancedGeneticAnimator] = None
        self.evolution_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.is_paused = False
        self.should_stop = False

        # Create GUI
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
        """Setup progress monitoring tab"""
        progress_frame = ttk.Frame(self.notebook)
        self.notebook.add(progress_frame, text="Progress")

        # Real-time metrics
        metrics_group = ttk.LabelFrame(progress_frame, text="Real-time Metrics", padding=10)
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

        # Log display
        log_group = ttk.LabelFrame(progress_frame, text="Evolution Log", padding=10)
        log_group.pack(fill='both', expand=True, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_group, state='disabled')
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

            # Set token file path for loading
            self.reducer.token_file = self.config.token_file
            self.reducer.ascii_only = self.config.ascii_only
            self.reducer.enhanced_validation = self.config.enhanced_validation
            self.reducer.comprehensive_search = self.config.comprehensive_search
            self.reducer.include_normal_tokens = self.config.include_normal_tokens
            self.reducer.output_file = self.config.output_file
            self.reducer.baseline_output = self.config.baseline_output

            # Setup animation if requested
            if self.config.show_gui_animation:
                self.animator = EnhancedGeneticAnimator(
                    base_text=self.config.base_text,
                    target_token=self.config.target_token,
                    wanted_token=self.config.wanted_token
                )
                self.reducer.animation_callback = GeneticAnimationCallback(self.animator)

            # Setup custom callback for GUI updates
            self.reducer.gui_callback = self._on_generation_update

            self.root.after(0, lambda: self.status_label.config(text="Running"))
            self.root.after(0, lambda: self.log_message("Evolution started"))

            # Run evolution with pause/stop checking
            results = self._run_evolution_with_controls()

            # Handle completion
            self.root.after(0, lambda: self._on_evolution_complete(results))

        except Exception as e:
            self.root.after(0, lambda: self._on_evolution_error(e))

    def _run_evolution_with_controls(self):
        """Run evolution with pause/stop control support"""
        # Custom evolution loop that respects pause/stop
        self.reducer.load_model()

        # Load glitch tokens from file
        self.root.after(0, lambda: self.log_message(f"Loading glitch tokens from: {self.config.token_file}"))
        self.reducer.load_glitch_tokens(
            token_file=self.config.token_file,
            ascii_only=self.config.ascii_only
        )

        # Get baseline probability
        baseline_prob_result = self.reducer.get_baseline_probability()
        # baseline_prob_result is a tuple: (target_id, target_prob, wanted_id, wanted_prob)
        if isinstance(baseline_prob_result, tuple) and len(baseline_prob_result) >= 2:
            target_id, target_prob, wanted_id, wanted_prob = baseline_prob_result
            baseline_prob = target_prob  # Use target probability
        else:
            baseline_prob = baseline_prob_result if not isinstance(baseline_prob_result, tuple) else 0.0
        baseline_msg = f"Baseline probability: {baseline_prob:.6f}"
        self.root.after(0, lambda msg=baseline_msg: self.log_message(msg))

        # Run comprehensive search if enabled
        if self.config.comprehensive_search and self.config.wanted_token:
            self.root.after(0, lambda: self.log_message("Running comprehensive wanted token search..."))
            try:
                comprehensive_results = self.reducer.comprehensive_wanted_token_search()
                search_msg = f"Comprehensive search found {len(comprehensive_results)} effective tokens"
                self.root.after(0, lambda msg=search_msg: self.log_message(msg))
            except Exception as e:
                error_msg = f"Comprehensive search failed: {e}"
                self.root.after(0, lambda msg=error_msg: self.log_message(msg))

        # Run baseline analysis if enabled
        if self.config.baseline_seeding:
            self.root.after(0, lambda: self.log_message("Running token impact baseline analysis..."))
            try:
                baseline_results = self.reducer.baseline_token_impacts()
                if baseline_results and len(baseline_results) > 0:
                    # baseline_results is a list of tuples - get the impact score safely
                    first_result = baseline_results[0]
                    if isinstance(first_result, (list, tuple)) and len(first_result) > 1:
                        top_impact = first_result[1]
                    else:
                        top_impact = 0.0
                    impact_msg = f"Baseline analysis complete, top impact: {top_impact:.3f}"
                    self.root.after(0, lambda msg=impact_msg: self.log_message(msg))
                else:
                    self.root.after(0, lambda: self.log_message("Baseline analysis complete, no results"))
            except Exception as e:
                error_msg = f"Baseline analysis failed: {e}"
                self.root.after(0, lambda msg=error_msg: self.log_message(msg))

        # Initialize population
        self.root.after(0, lambda: self.log_message("Initializing population..."))
        population = self.reducer.create_initial_population()

        best_individual = None
        best_fitness = 0
        stagnation_count = 0
        generation = 0

        for generation in range(self.config.generations):
            # Check for stop/pause
            while self.is_paused and not self.should_stop:
                time.sleep(0.1)

            if self.should_stop:
                self.root.after(0, lambda: self.log_message("Evolution stopped by user"))
                break

            # Evolve generation
            try:
                population = self.reducer.evolve_generation(population)
            except Exception as e:
                error_msg = f"Evolution error: {e}"
                self.root.after(0, lambda msg=error_msg: self.log_message(msg))
                break

            # Find best individual
            if population:
                current_best = max(population, key=lambda x: x.fitness)
                if current_best.fitness > best_fitness:
                    best_individual = current_best
                    best_fitness = current_best.fitness
                    stagnation_count = 0
                else:
                    stagnation_count += 1

                # Update GUI
                progress = (generation + 1) / self.config.generations * 100
                avg_fitness = sum(ind.fitness for ind in population) / len(population)
                current_gen = generation + 1
                current_best = best_individual
                self.root.after(0, lambda g=current_gen, p=progress, bf=best_fitness, af=avg_fitness, bi=current_best:
                              self._update_progress(g, p, bf, af, bi))

                # Check early stopping
                if best_fitness >= self.config.early_stopping_threshold:
                    stop_msg = f"Early stopping at generation {generation+1}, threshold reached"
                    self.root.after(0, lambda msg=stop_msg: self.log_message(msg))
                    break

        return {
            'best_individual': best_individual,
            'final_population': population,
            'generations_completed': generation + 1 if not self.should_stop else generation,
            'baseline_probability': baseline_prob
        }

    def _on_generation_update(self, generation, population, best_individual):
        """Callback for generation updates"""
        if hasattr(self, 'animator') and self.animator:
            # Update animator if available
            pass

    def _update_progress(self, generation, progress, best_fitness, avg_fitness, best_individual):
        """Update progress display"""
        self.progress_bar['value'] = progress
        self.progress_var.set(f"Generation: {generation}/{self.config.generations}")
        self.current_generation_var.set(str(generation))
        self.best_fitness_var.set(f"{best_fitness:.3f}")
        self.avg_fitness_var.set(f"{avg_fitness:.3f}")

        if best_individual:
            reduction = best_fitness * 100
            self.reduction_var.set(f"{reduction:.1f}%")

            # Update best individual display
            self.best_text.config(state='normal')
            self.best_text.delete(1.0, tk.END)
            self.best_text.insert(1.0, f"Tokens: {best_individual.tokens}\n")
            if hasattr(self.reducer, 'tokenizer'):
                try:
                    token_texts = [self.reducer.tokenizer.decode([token_id]) for token_id in best_individual.tokens]
                    self.best_text.insert(tk.END, f"Decoded: {token_texts}\n")
                except:
                    pass
            self.best_text.insert(tk.END, f"Fitness: {best_individual.fitness:.6f}\n")
            self.best_text.insert(tk.END, f"Reduction: {reduction:.2f}%")
            self.best_text.config(state='disabled')

    def _on_evolution_complete(self, results):
        """Handle evolution completion"""
        self.status_label.config(text="Complete")
        self.log_message("Evolution completed successfully")

        # Display final results
        self._display_final_results(results)

        # Reset UI state
        self._reset_ui_state()

        # Keep animator alive if it exists
        if hasattr(self, 'animator') and self.animator:
            self.animator.mark_complete()

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

        best_individual = results.get('best_individual')
        if best_individual:
            self.results_text.insert(tk.END, "🏆 EVOLUTION RESULTS\n")
            self.results_text.insert(tk.END, "=" * 50 + "\n\n")

            self.results_text.insert(tk.END, f"Configuration:\n")
            self.results_text.insert(tk.END, f"  Model: {self.config.model_name}\n")
            self.results_text.insert(tk.END, f"  Base Text: '{self.config.base_text}'\n")
            self.results_text.insert(tk.END, f"  Target Token: '{self.config.target_token}'\n")
            self.results_text.insert(tk.END, f"  Wanted Token: '{self.config.wanted_token}'\n")
            self.results_text.insert(tk.END, f"  Generations: {results['generations_completed']}/{self.config.generations}\n\n")

            self.results_text.insert(tk.END, f"Best Individual:\n")
            self.results_text.insert(tk.END, f"  Token IDs: {best_individual.tokens}\n")

            if hasattr(self.reducer, 'tokenizer'):
                try:
                    token_texts = [self.reducer.tokenizer.decode([token_id]) for token_id in best_individual.tokens]
                    self.results_text.insert(tk.END, f"  Token Texts: {token_texts}\n")
                except:
                    pass

            self.results_text.insert(tk.END, f"  Fitness Score: {best_individual.fitness:.6f}\n")
            reduction = best_individual.fitness * 100
            self.results_text.insert(tk.END, f"  Probability Reduction: {reduction:.2f}%\n")

            baseline_prob = results.get('baseline_probability', 0)
            if baseline_prob and baseline_prob > 0:
                try:
                    final_prob = baseline_prob * (1 - best_individual.fitness)
                    self.results_text.insert(tk.END, f"  Baseline Probability: {baseline_prob:.6f}\n")
                    self.results_text.insert(tk.END, f"  Final Probability: {final_prob:.6f}\n")
                except (TypeError, ValueError) as e:
                    self.results_text.insert(tk.END, f"  Baseline Probability: {baseline_prob}\n")

        else:
            self.results_text.insert(tk.END, "No results available - evolution may have been stopped early.\n")

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
