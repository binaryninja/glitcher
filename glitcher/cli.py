#!/usr/bin/env python3
"""
Glitcher CLI - A command-line tool for mining and testing glitch tokens in language models.
"""

import argparse
import json
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from glitcher.model import (
    mine_glitch_tokens,
    strictly_glitch_verify,
    get_template_for_model,
    chat_token,
    initialize_model_and_tokenizer
)

# Import genetic algorithm functionality
from .genetic import GeneticProbabilityReducer, GeneticBatchRunner, EnhancedGeneticAnimator, GeneticAnimationCallback

# Import range mining functionality
try:
    import sys
    import os
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from range_mining import range_based_mining
except ImportError:
    range_based_mining = None

# Import validation tests
# from glitcher.tests.validation.run_all_tests import run_all_validation_tests


class GlitcherCLI:
    """CLI for Glitcher to find and test glitch tokens with save/resume functionality."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.args: Optional[argparse.Namespace] = None
        self.progress = {
            "model_path": "",
            "num_iterations": 0,
            "iterations_completed": 0,
            "total_tokens_checked": 0,
            "glitch_tokens": [],
            "glitch_token_ids": [],
            "start_time": 0,
            "last_saved": 0,
        }
        self.running = True

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)

    def _fix_seeds(self):
        """Fix random seeds for reproducibility if --seed was provided."""
        seed = getattr(self.args, "seed", None)
        if seed is None:
            return
        import torch
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Random seeds fixed to {seed}")

    def _collect_snapshot(self) -> dict:
        """Collect an environment snapshot for embedding in result files."""
        import platform
        from datetime import datetime, timezone
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
        }
        for lib in ["torch", "transformers", "accelerate", "bitsandbytes"]:
            try:
                mod = __import__(lib)
                snapshot[lib] = getattr(mod, "__version__", "unknown")
            except ImportError:
                snapshot[lib] = "not installed"
        try:
            import torch
            if torch.cuda.is_available():
                snapshot["gpu"] = torch.cuda.get_device_name(0)
                snapshot["cuda"] = torch.version.cuda or "unknown"
        except ImportError:
            pass
        snapshot["device"] = getattr(self.args, "device", "cuda")
        snapshot["quantization"] = getattr(self.args, "quant_type", "bfloat16")
        seed = getattr(self.args, "seed", None)
        snapshot["seed"] = seed
        return snapshot

    def _inject_snapshot(self, filepath: str):
        """Inject environment snapshot into an existing JSON file."""
        try:
            with open(filepath) as f:
                data = json.load(f)
            if isinstance(data, dict) and "_environment" not in data:
                data["_environment"] = self._collect_snapshot()
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass  # non-critical, don't break the run

    def setup_parser(self):
        """Setup command line argument parser."""
        parser = argparse.ArgumentParser(
            description="Glitcher CLI - Find and test glitch tokens in language models"
        )

        subparsers = parser.add_subparsers(dest="command", help="Command to run")

        # Mine command
        mine_parser = subparsers.add_parser("mine", help="Mine for glitch tokens")
        mine_parser.add_argument("model_path", help="Path or name of the model to test")
        mine_parser.add_argument(
            "--num-iterations", type=int, default=50,
            help="Number of iterations to run (default: 50)"
        )
        mine_parser.add_argument(
            "--batch-size", type=int, default=8,
            help="Batch size for token testing (default: 8)"
        )
        mine_parser.add_argument(
            "--k", type=int, default=32,
            help="Number of nearest tokens to consider (default: 32)"
        )
        mine_parser.add_argument(
            "--output", type=str, default="glitch_tokens.json",
            help="Output file for results (default: glitch_tokens.json)"
        )
        mine_parser.add_argument(
            "--save-interval", type=int, default=5,
            help="Save progress every N iterations (default: 5)"
        )
        mine_parser.add_argument(
            "--resume", action="store_true",
            help="Resume from previous progress file"
        )
        mine_parser.add_argument(
            "--progress-file", type=str, default="glitch_progress.json",
            help="File to save/load progress (default: glitch_progress.json)"
        )
        mine_parser.add_argument(
            "--device", type=str, default="cuda",
            help="Device to use (default: cuda)"
        )
        mine_parser.add_argument(
            "--quant-type", type=str, default="bfloat16",
            choices=["auto", "bfloat16", "float16", "int8", "int4"],
            help="Quantization type (default: bfloat16)"
        )
        mine_parser.add_argument(
            "--enhanced-validation", action="store_true", default=True,
            help="Use enhanced validation that generates multiple tokens (default: True)"
        )
        mine_parser.add_argument(
            "--disable-enhanced-validation", action="store_true",
            help="Disable enhanced validation and use standard method"
        )
        mine_parser.add_argument(
            "--validation-tokens", type=int, default=50,
            help="Maximum tokens to generate in enhanced validation (default: 50)"
        )
        mine_parser.add_argument(
            "--num-attempts", type=int, default=1,
            help="Number of times to test each token for non-deterministic validation (default: 1)"
        )
        mine_parser.add_argument(
            "--asr-threshold", type=float, default=0.5,
            help="ASR threshold for considering token a glitch (default: 0.5)"
        )
        mine_parser.add_argument(
            "--reasoning-level", type=str, default="medium",
            choices=["none", "medium", "deep"],
            help="Reasoning level for Harmony-enabled models (default: medium)"
        )
        mine_parser.add_argument(
            "--seed", type=int, default=None,
            help="Random seed for reproducibility (fixes torch, numpy, and python random)"
        )

        # Range mining options
        mine_parser.add_argument(
            "--mode", type=str, default="entropy",
            choices=["entropy", "range", "unicode", "special"],
            help="Mining mode: entropy (default), range, unicode, or special"
        )
        mine_parser.add_argument(
            "--range-start", type=int,
            help="Starting token ID for range mining (requires --mode range)"
        )
        mine_parser.add_argument(
            "--range-end", type=int,
            help="Ending token ID for range mining (requires --mode range)"
        )
        mine_parser.add_argument(
            "--sample-rate", type=float, default=0.1,
            help="Fraction of tokens to sample in range mining (0.0-1.0, default: 0.1)"
        )
        mine_parser.add_argument(
            "--max-tokens-per-range", type=int, default=100,
            help="Maximum tokens to test per range in range mining (default: 100)"
        )

        # Test command
        test_parser = subparsers.add_parser("test", help="Test specific tokens")
        test_parser.add_argument("model_path", help="Path or name of the model to test")
        test_parser.add_argument(
            "--token-ids", type=str,
            help="Comma-separated list of token IDs to test"
        )
        test_parser.add_argument(
            "--token-file", type=str,
            help="JSON file containing token IDs to test"
        )
        test_parser.add_argument(
            "--output", type=str, default="test_results.json",
            help="Output file for results (default: test_results.json)"
        )
        test_parser.add_argument(
            "--device", type=str, default="cuda",
            help="Device to use (default: cuda)"
        )
        test_parser.add_argument(
            "--quant-type", type=str, default="bfloat16",
            choices=["auto", "bfloat16", "float16", "int8", "int4"],
            help="Quantization type (default: bfloat16)"
        )
        test_parser.add_argument(
            "--enhanced", action="store_true",
            help="Use enhanced validation that generates multiple tokens"
        )
        test_parser.add_argument(
            "--max-tokens", type=int, default=100,
            help="Maximum tokens to generate in enhanced validation (default: 100)"
        )
        test_parser.add_argument(
            "--quiet", action="store_true",
            help="Suppress transformer warnings during validation"
        )
        test_parser.add_argument(
            "--num-attempts", type=int, default=1,
            help="Number of times to test each token for non-deterministic validation (default: 1)"
        )
        test_parser.add_argument(
            "--asr-threshold", type=float, default=0.5,
            help="ASR threshold for considering token a glitch (default: 0.5)"
        )
        test_parser.add_argument(
            "--seed", type=int, default=None,
            help="Random seed for reproducibility (fixes torch, numpy, and python random)"
        )

        # Compare command
        compare_parser = subparsers.add_parser("compare", help="Compare standard vs enhanced validation methods")
        compare_parser.add_argument("model_path", help="Path or name of the model to test")
        compare_parser.add_argument(
            "--token-ids", type=str,
            help="Comma-separated list of token IDs to test"
        )
        compare_parser.add_argument(
            "--token-file", type=str,
            help="JSON file containing token IDs to test"
        )
        compare_parser.add_argument(
            "--output", type=str, default="comparison_results.json",
            help="Output file for results (default: comparison_results.json)"
        )
        compare_parser.add_argument(
            "--device", type=str, default="cuda",
            help="Device to use (default: cuda)"
        )
        compare_parser.add_argument(
            "--quant-type", type=str, default="bfloat16",
            choices=["auto", "bfloat16", "float16", "int8", "int4"],
            help="Quantization type (default: bfloat16)"
        )
        compare_parser.add_argument(
            "--max-tokens", type=int, default=100,
            help="Maximum tokens to generate in enhanced validation (default: 100)"
        )
        compare_parser.add_argument(
            "--num-attempts", type=int, default=1,
            help="Number of times to test each token for non-deterministic validation (default: 1)"
        )
        compare_parser.add_argument(
            "--asr-threshold", type=float, default=0.5,
            help="ASR threshold for considering token a glitch (default: 0.5)"
        )

        # Chat command
        chat_parser = subparsers.add_parser("chat", help="Chat with a token to see its behavior")
        chat_parser.add_argument("model_path", help="Path or name of the model to test")
        chat_parser.add_argument("token_id", type=int, help="Token ID to test")
        chat_parser.add_argument(
            "--max-size", type=int, default=10,
            help="Maximum number of tokens to generate (default: 10)"
        )
        chat_parser.add_argument(
            "--device", type=str, default="cuda",
            help="Device to use (default: cuda)"
        )
        chat_parser.add_argument(
            "--quant-type", type=str, default="bfloat16",
            choices=["bfloat16", "float16", "int8", "int4"],
            help="Quantization type (default: bfloat16)"
        )

        # Validate command
        validate_parser = subparsers.add_parser("validate", help="Run validation tests on a model")
        validate_parser.add_argument("model_path", help="Path or name of the model to test")
        validate_parser.add_argument(
            "--device", type=str, default="cuda",
            help="Device to use (default: cuda)"
        )
        validate_parser.add_argument(
            "--quant-type", type=str, default="bfloat16",
            choices=["bfloat16", "float16"],
            help="Quantization type (default: bfloat16)"
        )
        validate_parser.add_argument(
            "--output-dir", type=str, default="validation_results",
            help="Directory to store test results (default: validation_results)"
        )

        # Domain extraction command
        domain_parser = subparsers.add_parser("domain", help="Test domain extraction from log files")
        domain_parser.add_argument("model_path", help="Path or name of the model to test")
        domain_parser.add_argument(
            "--token-ids", type=str,
            help="Comma-separated list of token IDs to test"
        )
        domain_parser.add_argument(
            "--token-file", type=str,
            help="JSON file containing token IDs to test"
        )
        domain_parser.add_argument(
            "--test-cpptypes", action="store_true",
            help="Test the known CppTypeDefinitionSizes glitch token"
        )
        domain_parser.add_argument(
            "--device", type=str, default="cuda",
            help="Device to use (default: cuda)"
        )
        domain_parser.add_argument(
            "--quant-type", type=str, default="bfloat16",
            choices=["bfloat16", "float16", "int8", "int4"],
            help="Quantization type (default: bfloat16)"
        )
        domain_parser.add_argument(
            "--output", type=str, default="domain_extraction_results.json",
            help="Output file for results (default: domain_extraction_results.json)"
        )
        domain_parser.add_argument(
            "--skip-normal", action="store_true",
            help="Skip testing normal tokens as control group"
        )
        domain_parser.add_argument(
            "--normal-count", type=int, default=5,
            help="Number of normal tokens to test as control (default: 5)"
        )

        # Classify command
        classify_parser = subparsers.add_parser("classify", help="Classify glitch tokens by their effects")
        classify_parser.add_argument("model_path", help="Path or name of the model to test")
        classify_parser.add_argument(
            "--token-ids", type=str,
            help="Comma-separated list of token IDs to classify"
        )
        classify_parser.add_argument(
            "--token-file", type=str,
            help="JSON file containing token IDs to classify"
        )
        classify_parser.add_argument(
            "--output", type=str, default="classified_tokens.json",
            help="Output file for results (default: classified_tokens.json)"
        )
        classify_parser.add_argument(
            "--device", type=str, default="cuda",
            help="Device to use (default: cuda)"
        )
        classify_parser.add_argument(
            "--quant-type", type=str, default="bfloat16",
            choices=["bfloat16", "float16", "int8", "int4"],
            help="Quantization type (default: bfloat16)"
        )
        classify_parser.add_argument(
            "--temperature", type=float, default=0.7,
            help="Temperature for model inference (default: 0.7)"
        )
        classify_parser.add_argument(
            "--top-p", type=float, default=0.95,
            help="Top-p (nucleus) sampling threshold (default: 0.95)"
        )
        classify_parser.add_argument(
            "--max-tokens", type=int, default=8192,
            help="Maximum tokens to generate per test (default: 8192)"
        )
        classify_parser.add_argument(
            "--debug-responses", action="store_true",
            help="Enable detailed response logging"
        )
        classify_parser.add_argument(
            "--email-extraction-only", action="store_true",
            help="Only run email extraction tests"
        )
        classify_parser.add_argument(
            "--domain-extraction-only", action="store_true",
            help="Only run domain extraction tests"
        )
        classify_parser.add_argument(
            "--control-char-only", action="store_true",
            help="Only run control character confusion tests (requires --token-ids or --token-file)"
        )
        classify_parser.add_argument(
            "--control-char-standalone", action="store_true",
            help="Run standalone control character confusion tests (no glitch tokens needed)"
        )
        classify_parser.add_argument(
            "--encoded-char-only", action="store_true",
            help="Only run encoded character confusion tests (requires --token-ids or --token-file)"
        )
        classify_parser.add_argument(
            "--encoded-char-standalone", action="store_true",
            help="Run standalone encoded character confusion tests (no glitch tokens needed)"
        )
        classify_parser.add_argument(
            "--encoded-char-plaintext", action="store_true",
            help="Run plaintext-only encoded char tests: URL, bare hex, caret, ASCII name, CTRL-dash (no glitch tokens needed)"
        )
        classify_parser.add_argument(
            "--behavioral-only", action="store_true",
            help="Only run behavioral tests"
        )
        classify_parser.add_argument(
            "--functional-only", action="store_true",
            help="Only run functional tests"
        )

        # Genetic command
        genetic_parser = subparsers.add_parser("genetic", help="Use genetic algorithm to breed glitch token combinations")
        genetic_parser.add_argument("model_path", help="Path or name of the model to test")
        genetic_parser.add_argument(
            "--base-text", type=str, default="The quick brown",
            help="Base text to test probability reduction on (default: 'The quick brown')"
        )
        genetic_parser.add_argument(
            "--target-token", type=str,
            help="Specific token to target for reduction (auto-detected if not provided)"
        )
        genetic_parser.add_argument(
            "--wanted-token", type=str,
            help="Specific token to target for increase (optional)"
        )
        genetic_parser.add_argument(
            "--token-file", type=str, default="glitch_tokens.json",
            help="JSON file containing glitch tokens (default: glitch_tokens.json)"
        )
        genetic_parser.add_argument(
            "--population-size", type=int, default=50,
            help="Population size for genetic algorithm (default: 50)"
        )
        genetic_parser.add_argument(
            "--generations", type=int, default=100,
            help="Maximum number of generations (default: 100)"
        )
        genetic_parser.add_argument(
            "--mutation-rate", type=float, default=0.1,
            help="Mutation rate (0.0-1.0, default: 0.1)"
        )
        genetic_parser.add_argument(
            "--crossover-rate", type=float, default=0.7,
            help="Crossover rate (0.0-1.0, default: 0.7)"
        )
        genetic_parser.add_argument(
            "--elite-size", type=int, default=5,
            help="Elite size for genetic algorithm (default: 5)"
        )
        genetic_parser.add_argument(
            "--max-tokens", type=int, default=3,
            help="Maximum tokens per individual combination (default: 3)"
        )
        genetic_parser.add_argument(
            "--early-stopping-threshold", type=float, default=0.999,
            help="Early stopping threshold for probability reduction (0.0-1.0, default: 0.999)"
        )
        genetic_parser.add_argument(
            "--output", type=str, default="genetic_results.json",
            help="Output file for results (default: genetic_results.json)"
        )
        genetic_parser.add_argument(
            "--device", type=str, default="cuda",
            help="Device to use (default: cuda)"
        )
        genetic_parser.add_argument(
            "--quant-type", type=str, default="bfloat16",
            choices=["bfloat16", "float16", "int8", "int4"],
            help="Quantization type (default: bfloat16)"
        )
        genetic_parser.add_argument(
            "--seed", type=int, default=None,
            help="Random seed for reproducibility (fixes torch, numpy, and python random)"
        )
        genetic_parser.add_argument(
            "--batch", action="store_true",
            help="Run batch experiments across multiple scenarios"
        )
        genetic_parser.add_argument(
            "--gui", action="store_true",
            help="Show real-time GUI animation of genetic algorithm evolution"
        )
        genetic_parser.add_argument(
            "--ascii-only", action="store_true",
            help="Filter tokens to only include those with ASCII-only decoded text"
        )
        genetic_parser.add_argument(
            "--include-normal-tokens", action="store_true",
            help="Include normal ASCII tokens from model vocabulary in addition to glitch tokens"
        )
        genetic_parser.add_argument(
            "--comprehensive-search", action="store_true",
            help="Perform comprehensive search through all vocabulary tokens for wanted token optimization (slower but thorough)"
        )
        genetic_parser.add_argument(
            "--no-cache", action="store_true",
            help="Disable caching of comprehensive search results"
        )
        genetic_parser.add_argument(
            "--clear-cache", action="store_true",
            help="Clear comprehensive search cache before running"
        )
        genetic_parser.add_argument(
            "--adaptive-mutation", action="store_true",
            help="Use adaptive mutation rate that starts high (0.8) and decreases to low (0.1) over generations"
        )
        genetic_parser.add_argument(
            "--initial-mutation-rate", type=float, default=0.8,
            help="Initial mutation rate for adaptive mutation (default: 0.8)"
        )
        genetic_parser.add_argument(
            "--final-mutation-rate", type=float, default=0.1,
            help="Final mutation rate for adaptive mutation (default: 0.1)"
        )
        genetic_parser.add_argument(
            "--baseline-only", action="store_true",
            help="Only run token impact baseline analysis without genetic algorithm evolution"
        )
        genetic_parser.add_argument(
            "--skip-baseline", action="store_true",
            help="Skip token impact baseline analysis and go straight to genetic algorithm"
        )
        genetic_parser.add_argument(
            "--baseline-output", type=str, default="token_impact_baseline.json",
            help="Output file for token impact baseline results (default: token_impact_baseline.json)"
        )
        genetic_parser.add_argument(
            "--baseline-top-n", type=int, default=10,
            help="Number of top tokens to display in baseline results (default: 10)"
        )
        genetic_parser.add_argument(
            "--baseline-seeding", action="store_true", default=True,
            help="Use baseline results to intelligently seed initial population (default: enabled)"
        )
        genetic_parser.add_argument(
            "--no-baseline-seeding", action="store_true",
            help="Disable baseline-guided population seeding, use random initialization only"
        )
        genetic_parser.add_argument(
            "--baseline-seeding-ratio", type=float, default=0.7,
            help="Fraction of population to seed with baseline guidance (0.0-1.0, default: 0.7)"
        )
        genetic_parser.add_argument(
            "--exact-token-count", action="store_true", default=True,
            help="Use exact max_tokens count for all individuals (default: enabled)"
        )
        genetic_parser.add_argument(
            "--variable-token-count", action="store_true",
            help="Allow variable token count (1 to max_tokens) for individuals"
        )
        genetic_parser.add_argument(
            "--sequence-aware-diversity", action="store_true", default=True,
            help="Enable sequence-aware diversity injection (default: enabled)"
        )
        genetic_parser.add_argument(
            "--no-sequence-diversity", action="store_true",
            help="Disable sequence-aware diversity injection, use traditional diversity only"
        )
        genetic_parser.add_argument(
            "--sequence-diversity-ratio", type=float, default=0.3,
            help="Fraction of diversity injection to use sequence-aware strategies (0.0-1.0, default: 0.3)"
        )
        genetic_parser.add_argument(
            "--enable-shuffle-mutation", action="store_true",
            help="Enable shuffle mutation (disabled by default to preserve token combinations)"
        )
        genetic_parser.add_argument(
            "--disable-shuffle-mutation", action="store_true", default=True,
            help="Disable shuffle mutation to preserve token combinations (default behavior)"
        )

        # GUI command
        gui_parser = subparsers.add_parser("gui", help="Launch graphical user interface for genetic algorithm control")
        gui_parser.add_argument(
            "--config", type=str,
            help="Configuration file to load on startup"
        )

        return parser

    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.args.model_path}")
        # Thread reasoning level to model via environment for Harmony-based prompts
        if hasattr(self.args, "reasoning_level") and self.args.reasoning_level:
            os.environ["GLITCHER_REASONING_LEVEL"] = self.args.reasoning_level
        self.model, self.tokenizer = initialize_model_and_tokenizer(
            model_path=self.args.model_path,
            device=self.args.device,
            quant_type=self.args.quant_type
        )
        print(f"Model loaded on {self.args.device}")

    def load_progress(self):
        """Load progress from file if it exists."""
        progress_path = Path(self.args.progress_file)
        if progress_path.exists():
            try:
                with open(progress_path, "r") as f:
                    self.progress = json.load(f)
                print(f"Loaded progress: {self.progress['iterations_completed']} iterations completed")
                print(f"Found {len(self.progress['glitch_token_ids'])} glitch tokens so far")
                return True
            except Exception as e:
                print(f"Error loading progress: {e}")
        return False

    def save_progress(self, final=False):
        """Save current progress to file."""
        try:
            # Update the progress data
            self.progress["last_saved"] = time.time()

            with open(self.args.progress_file, "w") as f:
                json.dump(self.progress, f, indent=2)

            if final:
                print("Final progress saved.")
            else:
                print(f"Progress saved after {self.progress['iterations_completed']} iterations.")
        except Exception as e:
            print(f"Error saving progress: {e}")

    def save_results(self, filename, data):
        """Save results to a file, auto-injecting an environment snapshot."""
        try:
            if isinstance(data, dict) and "_environment" not in data:
                data["_environment"] = self._collect_snapshot()
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")

    def handle_interrupt(self, sig, frame):
        """Handle interrupt signal (Ctrl+C) to save progress and exit gracefully."""
        print("\nInterrupted. Saving progress and exiting...")
        self.running = False

    def run_mining(self):
        """Run the mining process."""
        # Check if this is range mining mode
        if self.args.mode in ["range", "unicode", "special"]:
            self.run_range_mining()
            return
        # Initialize progress
        if self.args.resume and self.load_progress():
            # Verify the model path is the same
            if self.progress["model_path"] != self.args.model_path:
                print("Warning: Model path in progress file doesn't match the specified model.")
                response = input("Continue anyway? (y/n): ")
                if response.lower() != "y":
                    return

            remaining_iterations = self.args.num_iterations - self.progress["iterations_completed"]
        else:
            # Initialize new progress
            self.progress = {
                "model_path": self.args.model_path,
                "num_iterations": self.args.num_iterations,
                "iterations_completed": 0,
                "total_tokens_checked": 0,
                "glitch_tokens": [],
                "glitch_token_ids": [],
                "start_time": time.time(),
                "last_saved": time.time(),
            }
            remaining_iterations = self.args.num_iterations

        if remaining_iterations <= 0:
            print("All iterations completed. Use a higher --num-iterations value to continue.")
            return

        # Set up mining parameters
        batch_size = self.args.batch_size
        k = self.args.k

        # Run the mining process
        print(f"Starting mining process with {remaining_iterations} iterations...")
        print(f"Using batch_size={batch_size}, k={k}")

        self.progress["start_time"] = time.time()

        def checkpoint_callback(data):
            """Callback function to save progress during mining"""
            self.progress["iterations_completed"] = data["iteration"] + 1
            self.progress["total_tokens_checked"] = data["total_tokens_checked"]
            self.progress["glitch_tokens"] = data["glitch_tokens"]
            self.progress["glitch_token_ids"] = data["glitch_token_ids"]

            # Save progress at the specified interval
            if (data["iteration"] + 1) % self.args.save_interval == 0:
                self.save_progress()

            # Check if we should continue running
            return self.running

        try:
            # Run glitch token mining with the specified number of iterations
            log_file = f"glitch_mining_log_{int(time.time())}.jsonl"
            print(f"Detailed mining logs will be saved to: {log_file}")

            # Determine validation method
            use_enhanced = self.args.enhanced_validation and not getattr(self.args, 'disable_enhanced_validation', False)

            if use_enhanced:
                asr_threshold = getattr(self.args, 'asr_threshold', 0.5)
                print(f"Using enhanced validation (max tokens: {self.args.validation_tokens}, attempts: {self.args.num_attempts}, ASR threshold: {asr_threshold})")
            else:
                print("Using standard validation method")

            glitch_tokens, glitch_token_ids = mine_glitch_tokens(
                model=self.model,
                tokenizer=self.tokenizer,
                num_iterations=remaining_iterations,
                batch_size=batch_size,
                k=k,
                verbose=True,
                language="ENG",
                checkpoint_callback=checkpoint_callback,
                log_file=log_file,
                enhanced_validation=use_enhanced,
                max_tokens=self.args.validation_tokens,
                num_attempts=self.args.num_attempts,
                asr_threshold=getattr(self.args, 'asr_threshold', 0.5)
            )

            # Update progress
            self.progress["iterations_completed"] = self.args.num_iterations
            self.progress["glitch_tokens"].extend(glitch_tokens)
            self.progress["glitch_token_ids"].extend(glitch_token_ids)

            # Save final results
            self.save_progress(final=True)

            # Save results
            results = {
                "model_path": self.args.model_path,
                "glitch_tokens": self.progress["glitch_tokens"],
                "glitch_token_ids": self.progress["glitch_token_ids"],
                "total_iterations": self.args.num_iterations,
                "runtime_seconds": time.time() - self.progress["start_time"],
            }

            self.save_results(self.args.output, results)

        except Exception as e:
            print(f"Error during mining: {e}")
            # Save progress on error
            self.save_progress(final=True)

    def run_range_mining(self):
        """Run range-based mining process."""
        if range_based_mining is None:
            print("Error: range_mining module not available. Please ensure range_mining.py is accessible.")
            return
        # Validate range mining arguments
        if self.args.mode == "range":
            if self.args.range_start is None or self.args.range_end is None:
                print("Error: --range-start and --range-end are required for range mode")
                return
            if self.args.range_start >= self.args.range_end:
                print("Error: --range-start must be less than --range-end")
                return

        if self.args.sample_rate <= 0 or self.args.sample_rate > 1:
            print("Error: --sample-rate must be between 0 and 1")
            return

        # Set up range mining parameters
        range_start = self.args.range_start
        range_end = self.args.range_end
        unicode_ranges = self.args.mode == "unicode"
        special_ranges = self.args.mode == "special"
        sample_rate = self.args.sample_rate
        max_tokens_per_range = self.args.max_tokens_per_range

        # Generate output filename based on mode
        if self.args.output == "glitch_tokens.json":  # default output
            if self.args.mode == "range":
                output_file = f"range_mining_{self.args.range_start}_{self.args.range_end}.json"
            elif self.args.mode == "unicode":
                output_file = "unicode_range_mining.json"
            elif self.args.mode == "special":
                output_file = "special_range_mining.json"
        else:
            output_file = self.args.output

        print(f"Starting {self.args.mode} mining mode...")
        print(f"Sample rate: {sample_rate}")
        print(f"Max tokens per range: {max_tokens_per_range}")
        print(f"Output file: {output_file}")

        try:
            # Run range-based mining
            results = range_based_mining(
                model_path=self.args.model_path,
                range_start=range_start,
                range_end=range_end,
                unicode_ranges=unicode_ranges,
                special_ranges=special_ranges,
                sample_rate=sample_rate,
                max_tokens_per_range=max_tokens_per_range,
                device=self.args.device,
                output_file=output_file,
                model=self.model,
                tokenizer=self.tokenizer
            )

            print(f"\n=== RANGE MINING COMPLETED ===")
            print(f"Total tokens tested: {results['total_tokens_tested']}")
            print(f"Total glitch tokens found: {results['total_glitch_tokens']}")
            if results['total_tokens_tested'] > 0:
                glitch_rate = results['total_glitch_tokens'] / results['total_tokens_tested']
                print(f"Glitch rate: {glitch_rate:.1%}")
            print(f"Results saved to: {output_file}")

        except Exception as e:
            print(f"Error during range mining: {e}")
            raise

    def run_token_test(self):
        """Test specific tokens for glitch properties."""
        # Get token IDs to test
        token_ids = []

        if self.args.token_ids:
            try:
                token_ids = [int(tid.strip()) for tid in self.args.token_ids.split(",")]
            except ValueError:
                print("Error: Token IDs must be comma-separated integers")
                return

        elif self.args.token_file:
            try:
                with open(self.args.token_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        token_ids = data
                    elif "glitch_token_ids" in data:
                        token_ids = data["glitch_token_ids"]
                    else:
                        print("Error: Could not find token IDs in the file")
                        return
            except Exception as e:
                print(f"Error loading token file: {e}")
                return

        if not token_ids:
            print("Error: No token IDs provided")
            return

        # Choose validation method
        if hasattr(self.args, 'enhanced') and self.args.enhanced:
            num_attempts = getattr(self.args, 'num_attempts', 1)
            asr_threshold = getattr(self.args, 'asr_threshold', 0.5)
            if num_attempts > 1:
                print(f"Testing {len(token_ids)} tokens using enhanced validation (generating up to {self.args.max_tokens} tokens, {num_attempts} attempts per token, ASR threshold: {asr_threshold})...")
            else:
                print(f"Testing {len(token_ids)} tokens using enhanced validation (generating up to {self.args.max_tokens} tokens, ASR threshold: {asr_threshold})...")
            validation_method = "enhanced"
        else:
            print(f"Testing {len(token_ids)} tokens using standard validation...")
            validation_method = "standard"

        # Get chat template
        chat_template = get_template_for_model(self.model.config._name_or_path, self.tokenizer)

        # Create log file for detailed verification results
        log_file = f"verification_log_{int(time.time())}.jsonl"
        print(f"Detailed verification logs will be saved to: {log_file}")

        # Add header to log file
        with open(log_file, 'w') as f:
            f.write("# Token Verification Log - Detailed analysis\n")
            model_info = {
                "event": "start_verification",
                "model_path": self.args.model_path,
                "num_tokens": len(token_ids),
                "device": self.args.device,
                "quant_type": self.args.quant_type,
                "validation_method": validation_method,
                "max_tokens": getattr(self.args, 'max_tokens', None),
                "num_attempts": getattr(self.args, 'num_attempts', 1),
                "asr_threshold": getattr(self.args, 'asr_threshold', 0.5),
                "is_llama32": "llama3.2" in self.model.config._name_or_path.lower() or "llama32" in self.model.config._name_or_path.lower()
            }
            f.write(json.dumps(model_info) + "\n")

        # Test each token
        results = []
        for i, token_id in enumerate(token_ids):
            if not self.running:
                break

            try:
                token = self.tokenizer.decode([token_id])

                # Log token being tested
                with open(log_file, 'a') as f:
                    f.write(json.dumps({
                        "event": "start_token_test",
                        "token": token,
                        "token_id": token_id,
                        "index": i+1,
                        "total": len(token_ids)
                    }) + "\n")

                # Use appropriate validation method
                if validation_method == "enhanced":
                    from .enhanced_validation import enhanced_glitch_verify
                    is_glitch, asr = enhanced_glitch_verify(
                        self.model, self.tokenizer, token_id, chat_template, log_file,
                        self.args.max_tokens, quiet=getattr(self.args, 'quiet', True),
                        num_attempts=getattr(self.args, 'num_attempts', 1),
                        asr_threshold=getattr(self.args, 'asr_threshold', 0.5)
                    )
                else:
                    is_glitch = strictly_glitch_verify(self.model, self.tokenizer, token_id, chat_template, log_file)
                    asr = None  # Standard validation doesn't provide ASR

                result = {
                    "token_id": token_id,
                    "token": token,
                    "is_glitch": is_glitch,
                    "asr": asr if validation_method == "enhanced" else None
                }

                results.append(result)
                if validation_method == "enhanced" and asr is not None:
                    print(f"[{i+1}/{len(token_ids)}] Token: '{token}', ID: {token_id}, Is glitch: {is_glitch}, ASR: {asr:.2%}")
                else:
                    print(f"[{i+1}/{len(token_ids)}] Token: '{token}', ID: {token_id}, Is glitch: {is_glitch}")

                # Log token test completion
                with open(log_file, 'a') as f:
                    f.write(json.dumps({
                        "event": "end_token_test",
                        "token": token,
                        "token_id": token_id,
                        "is_glitch": is_glitch
                    }) + "\n")

            except Exception as e:
                print(f"Error testing token {token_id}: {e}")
                # Log error
                with open(log_file, 'a') as f:
                    f.write(json.dumps({
                        "event": "token_test_error",
                        "token_id": token_id,
                        "error": str(e)
                    }) + "\n")

        # Save results
        test_results = {
            "model_path": self.args.model_path,
            "test_results": results,
            "timestamp": time.time(),
            "log_file": log_file,
            "validation_method": validation_method
        }

        # Add summary to log file
        with open(log_file, 'a') as f:
            glitch_count = sum(1 for r in results if r.get("is_glitch", False))
            summary = {
                "event": "verification_summary",
                "total_tokens": len(results),
                "glitch_tokens": glitch_count,
                "glitch_rate": glitch_count / len(results) if results else 0,
                "validation_method": validation_method
            }
            f.write(json.dumps(summary) + "\n")

        self.save_results(self.args.output, test_results)

    def run_chat_test(self):
        """Test how a model responds to a specific token in a chat context."""
        token_id = self.args.token_id

        try:
            # Run chat test
            print(f"Testing token ID {token_id} in chat context...")
            result = chat_token(
                self.args.model_path,
                token_id,
                max_size=self.args.max_size,
                device=self.args.device,
                quant_type=self.args.quant_type
            )

            # Print results
            print("\nChat Test Results:")
            print(f"Token ID: {result['token_id']}")
            print(f"Token: '{result['token']}'")
            print(f"Probability of correct repetition: {result['target_token_prob']:.6f}")
            print(f"Most likely token: '{result['top_token']}' (ID: {result['top_token_id']})")
            print(f"Most likely token probability: {result['top_token_prob']:.6f}")
            print("\nModel output:")
            print(result['generated_text'])

            # Save result
            chat_result = {
                "model_path": self.args.model_path,
                "token_id": token_id,
                "token": result['token'],
                "correct_prob": result['target_token_prob'],
                "top_tokens": result['top_5_indices'],
                "top_token": result['top_token'],
                "top_token_id": result['top_token_id'],
                "top_token_prob": result['top_token_prob'],
                "model_output": result['generated_text'],
                "timestamp": time.time()
            }

            output_file = f"chat_token_{token_id}.json"
            self.save_results(output_file, chat_result)

        except Exception as e:
            print(f"Error testing token in chat: {e}")

    def run_validation(self):
        """Run model validation tests - quick smoke test of core functionality"""
        import os
        output_dir = getattr(self.args, 'output_dir', 'validation_results')
        os.makedirs(output_dir, exist_ok=True)

        print(f"Running validation tests for model: {self.args.model_path}")

        # Load model
        self.load_model()

        # Run a quick test with a few known token IDs to validate the model works
        test_token_ids = []
        # Try to load from default glitch_tokens.json if it exists
        if os.path.exists('glitch_tokens.json'):
            try:
                with open('glitch_tokens.json', 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict) and 'glitch_token_ids' in data:
                    test_token_ids = data['glitch_token_ids'][:5]
                elif isinstance(data, list):
                    test_token_ids = data[:5]
            except Exception:
                pass

        if not test_token_ids:
            # Use a few common token IDs as fallback
            test_token_ids = [1, 2, 3]

        # Run basic token tests
        results = {
            'model_path': self.args.model_path,
            'device': self.args.device,
            'quant_type': self.args.quant_type,
            'tokens_tested': len(test_token_ids),
            'test_results': []
        }

        chat_template = get_template_for_model(self.model.config._name_or_path, self.tokenizer)

        for token_id in test_token_ids:
            try:
                token_text = self.tokenizer.decode([token_id])
                is_glitch = strictly_glitch_verify(
                    self.model, self.tokenizer, token_id, chat_template
                )
                results['test_results'].append({
                    'token_id': token_id,
                    'token_text': token_text,
                    'is_glitch': is_glitch,
                    'success': True
                })
            except Exception as e:
                results['test_results'].append({
                    'token_id': token_id,
                    'error': str(e),
                    'success': False
                })

        # Save results
        output_file = os.path.join(output_dir, 'validation_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Validation complete: {len(test_token_ids)} tokens tested")
        print(f"Results saved to {output_dir}/")

    def run_comparison(self):
        """Compare standard vs enhanced validation methods"""
        # Load model for comparison
        self.load_model()

        # Get token IDs to test
        token_ids = []

        if self.args.token_ids:
            try:
                token_ids = [int(tid.strip()) for tid in self.args.token_ids.split(",")]
            except ValueError:
                print("Error: Token IDs must be comma-separated integers")
                return

        elif self.args.token_file:
            try:
                with open(self.args.token_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        token_ids = data
                    elif "glitch_token_ids" in data:
                        token_ids = data["glitch_token_ids"]
                    else:
                        print("Error: Could not find token IDs in the file")
                        return
            except Exception as e:
                print(f"Error loading token file: {e}")
                return

        if not token_ids:
            print("Error: No token IDs provided")
            return

        print(f"Comparing validation methods for {len(token_ids)} tokens...")

        # Get chat template
        chat_template = get_template_for_model(self.model.config._name_or_path, self.tokenizer)

        # Create log file for detailed comparison results
        log_file = f"comparison_log_{int(time.time())}.jsonl"
        print(f"Detailed comparison logs will be saved to: {log_file}")

        # Test each token with both methods
        comparison_results = []
        agreements = 0
        disagreements = 0

        for i, token_id in enumerate(token_ids):
            if not self.running:
                break

            try:
                from .enhanced_validation import compare_validation_methods

                result = compare_validation_methods(
                    self.model, self.tokenizer, token_id, chat_template, self.args.max_tokens,
                    quiet=True, num_attempts=getattr(self.args, 'num_attempts', 1),
                    asr_threshold=getattr(self.args, 'asr_threshold', 0.5)
                )

                comparison_results.append(result)

                if result["methods_agree"]:
                    agreements += 1
                else:
                    disagreements += 1

                print(f"[{i+1}/{len(token_ids)}] Token: '{result['token']}' "
                      f"(ID: {token_id}) - Original: {result['original_method']}, "
                      f"Enhanced: {result['enhanced_method']}, "
                      f"Agree: {result['methods_agree']}")

                # Log comparison result
                with open(log_file, 'a') as f:
                    f.write(json.dumps({
                        "event": "token_comparison",
                        "index": i + 1,
                        "total": len(token_ids),
                        **result
                    }) + "\n")

            except Exception as e:
                print(f"Error comparing token {token_id}: {e}")
                with open(log_file, 'a') as f:
                    f.write(json.dumps({
                        "event": "comparison_error",
                        "token_id": token_id,
                        "error": str(e)
                    }) + "\n")

        # Save comparison results
        final_results = {
            "model_path": self.args.model_path,
            "comparison_results": comparison_results,
            "summary": {
                "total_tokens": len(comparison_results),
                "agreements": agreements,
                "disagreements": disagreements,
                "agreement_rate": agreements / len(comparison_results) if comparison_results else 0
            },
            "timestamp": time.time(),
            "log_file": log_file,
            "max_tokens": self.args.max_tokens
        }

        # Add summary to log file
        with open(log_file, 'a') as f:
            f.write(json.dumps({
                "event": "comparison_summary",
                "total_tokens": len(comparison_results),
                "agreements": agreements,
                "disagreements": disagreements,
                "agreement_rate": agreements / len(comparison_results) if comparison_results else 0
            }) + "\n")

        self.save_results(self.args.output, final_results)

        print(f"\nComparison Summary:")
        print(f"Total tokens tested: {len(comparison_results)}")
        print(f"Methods agree: {agreements}")
        print(f"Methods disagree: {disagreements}")
        print(f"Agreement rate: {agreements / len(comparison_results) * 100:.1f}%" if comparison_results else "N/A")

    def run_domain_extraction(self):
        """Run domain extraction tests."""
        try:
            # Import here to avoid circular imports
            import sys
            import os

            # Add the tests directory to the path so we can import test_domain_extraction
            tests_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests')
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            from test_domain_extraction import DomainExtractionTester

            print(f"🌐 Testing domain extraction with model: {self.args.model_path}")
            print("=" * 60)

            # Validate arguments
            if not any([getattr(self.args, 'token_ids', None),
                       getattr(self.args, 'token_file', None),
                       getattr(self.args, 'test_cpptypes', False)]):
                print("❌ Error: Must specify at least one of: --token-ids, --token-file, or --test-cpptypes")
                return

            # Initialize tester
            tester = DomainExtractionTester(
                model_path=self.args.model_path,
                device=getattr(self.args, 'device', 'cuda'),
                quant_type=getattr(self.args, 'quant_type', 'bfloat16')
            )

            # Run tests
            results = tester.run_tests(self.args)

            # Save results
            output_file = getattr(self.args, 'output', 'domain_extraction_results.json')
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\n✅ Results saved to: {output_file}")
            print("🌐 Domain extraction testing completed!")

        except Exception as e:
            print(f"❌ Error in domain extraction: {e}")
            raise

    def run_classification(self):
        """Run glitch token classification."""
        try:
            # Import the new classification system
            from .classification.glitch_classifier import GlitchClassifier
            from .classification.types import TestConfig
            from .classification.cli import load_token_ids

            print(f"🔍 Classifying glitch tokens with model: {self.args.model_path}")
            print("=" * 60)

            # Standalone modes don't need token IDs
            is_standalone = getattr(
                self.args, 'control_char_standalone', False
            ) or getattr(
                self.args, 'encoded_char_standalone', False
            ) or getattr(
                self.args, 'encoded_char_plaintext', False
            )

            # Validate arguments
            if not is_standalone and not any([
                getattr(self.args, 'token_ids', None),
                getattr(self.args, 'token_file', None),
            ]):
                print("Error: Must specify either --token-ids or --token-file")
                return

            # Load token IDs (skip for standalone mode)
            token_ids = [] if is_standalone else load_token_ids(self.args)
            if not is_standalone:
                print(f"Loaded {len(token_ids)} token IDs to classify")

            # Create test configuration
            config = TestConfig(
                max_tokens=getattr(self.args, 'max_tokens', 8192),
                temperature=getattr(self.args, 'temperature', 0.7),
                top_p=getattr(self.args, 'top_p', 0.95),
                enable_debug=getattr(self.args, 'debug_responses', False),
                simple_template=False
            )

            # Initialize classifier
            classifier = GlitchClassifier(
                model_path=self.args.model_path,
                device=getattr(self.args, 'device', 'cuda'),
                quant_type=getattr(self.args, 'quant_type', 'bfloat16'),
                config=config
            )

            # Handle special test modes
            if getattr(self.args, 'email_extraction_only', False):
                print("Running email extraction tests only...")
                summary = classifier.run_email_extraction_only(token_ids)
                output_file = self.args.output

            elif getattr(self.args, 'domain_extraction_only', False):
                print("Running domain extraction tests only...")
                summary = classifier.run_domain_extraction_only(token_ids)
                output_file = self.args.output

            elif getattr(self.args, 'control_char_only', False):
                print("Running control character confusion tests only...")
                summary = classifier.run_control_char_tests_only(token_ids)
                output_file = self.args.output

            elif getattr(self.args, 'control_char_standalone', False):
                print("Running standalone control character confusion tests...")
                summary = classifier.run_control_char_standalone()
                output_file = self.args.output

            elif getattr(self.args, 'encoded_char_only', False):
                print("Running encoded character confusion tests only...")
                summary = classifier.run_encoded_char_tests_only(token_ids)
                output_file = self.args.output

            elif getattr(self.args, 'encoded_char_standalone', False):
                print("Running standalone encoded character confusion tests...")
                summary = classifier.run_encoded_char_standalone()
                output_file = self.args.output

            elif getattr(self.args, 'encoded_char_plaintext', False):
                print("Running plaintext-only encoded character confusion tests...")
                output_file = self.args.output
                summary = classifier.run_encoded_char_plaintext(
                    output_file=output_file,
                )

            else:
                # Run full classification
                print(f"Running full classification on {len(token_ids)} tokens...")

                # Filter tests based on mode
                if getattr(self.args, 'behavioral_only', False):
                    print("Filtering to behavioral tests only...")
                elif getattr(self.args, 'functional_only', False):
                    print("Filtering to functional tests only...")

                # Run classification
                results = classifier.classify_tokens(token_ids)

                # Get summary
                summary = classifier.get_results_summary()
                summary["classifications"] = [result.to_dict() for result in results]

                # Add detailed extraction results
                if hasattr(classifier, '_detailed_email_results') and classifier._detailed_email_results:
                    summary["detailed_email_results"] = classifier._detailed_email_results
                if hasattr(classifier, '_detailed_domain_results') and classifier._detailed_domain_results:
                    summary["detailed_domain_results"] = classifier._detailed_domain_results

                output_file = self.args.output

                # Print summary table
                classifier.print_summary_table()

            # Save results
            import json
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n✅ Results saved to {output_file}")

        except Exception as e:
            print(f"❌ Error in classification: {e}")
            import traceback
            traceback.print_exc()
            raise

    def run_genetic(self):
        """Run genetic algorithm for breeding glitch token combinations."""
        try:
            print("Starting genetic algorithm for breeding glitch token combinations...")
            print(f"Model: {self.args.model_path}")
            print(f"Base text: '{self.args.base_text}'")
            print(f"Population size: {self.args.population_size}")
            print(f"Generations: {self.args.generations}")

            # Setup GUI if requested
            gui_callback = None
            if self.args.gui:
                try:
                    print("Initializing real-time GUI animation...")
                    animator = EnhancedGeneticAnimator(
                        base_text=self.args.base_text,
                        target_token=self.args.target_token,
                        wanted_token=getattr(self.args, 'wanted_token', None),
                        max_generations=self.args.generations
                    )
                    gui_callback = GeneticAnimationCallback(animator)
                except ImportError as e:
                    print(f"WARNING: GUI not available: {e}")
                    print("Install matplotlib for GUI support: pip install matplotlib")
                    gui_callback = None
                except Exception as e:
                    print(f"WARNING: Failed to initialize GUI: {e}")
                    gui_callback = None

            if self.args.batch:
                # Run batch experiments
                print("Running batch experiments...")

                # Create batch runner
                runner = GeneticBatchRunner(self.args.model_path, self.args.token_file, ascii_only=self.args.ascii_only)

                # Set up scenarios (you can customize these)
                scenarios = [
                    {
                        'name': 'quick_brown_fox',
                        'base_text': 'The quick brown',
                        'target_token': None,
                        'ga_params': {
                            'population_size': self.args.population_size,
                            'max_generations': self.args.generations,
                            'max_tokens_per_individual': self.args.max_tokens,
                            'early_stopping_threshold': self.args.early_stopping_threshold
                        }
                    },
                    {
                        'name': 'common_phrase',
                        'base_text': 'Hello world',
                        'target_token': None,
                        'ga_params': {
                            'population_size': self.args.population_size,
                            'max_generations': self.args.generations,
                            'max_tokens_per_individual': self.args.max_tokens,
                            'early_stopping_threshold': self.args.early_stopping_threshold
                        }
                    }
                ]

                # Add scenarios and run
                for scenario in scenarios:
                    runner.add_scenario(**scenario)

                # Run all experiments
                results = runner.run_all_experiments()

                # Save results
                output_path = Path(self.args.output).parent
                runner.save_results(str(output_path))

                print(f"\n✅ Batch results saved to {output_path}")
                print(runner.generate_report())

            else:
                # Create genetic algorithm instance
                ga = GeneticProbabilityReducer(
                    model_name=self.args.model_path,
                    base_text=self.args.base_text,
                    target_token=self.args.target_token,
                    wanted_token=getattr(self.args, 'wanted_token', None),
                    gui_callback=gui_callback
                )

                # Pass tokenizer to GUI callback after model loading
                if gui_callback and hasattr(ga, 'tokenizer') and ga.tokenizer:
                    gui_callback.set_tokenizer(ga.tokenizer)

                # Configure GA parameters
                ga.population_size = self.args.population_size
                ga.max_generations = self.args.generations
                ga.mutation_rate = self.args.mutation_rate
                ga.crossover_rate = self.args.crossover_rate
                ga.elite_size = self.args.elite_size
                ga.max_tokens_per_individual = self.args.max_tokens
                ga.early_stopping_threshold = self.args.early_stopping_threshold

                # Configure adaptive mutation
                ga.adaptive_mutation = self.args.adaptive_mutation
                ga.initial_mutation_rate = self.args.initial_mutation_rate
                ga.final_mutation_rate = self.args.final_mutation_rate
                if self.args.adaptive_mutation:
                    ga.current_mutation_rate = self.args.initial_mutation_rate

                # Configure baseline seeding
                if self.args.no_baseline_seeding:
                    ga.use_baseline_seeding = False
                else:
                    ga.use_baseline_seeding = self.args.baseline_seeding
                ga.baseline_seeding_ratio = max(0.0, min(1.0, self.args.baseline_seeding_ratio))

                # Configure token count behavior
                if self.args.variable_token_count:
                    ga.use_exact_token_count = False
                else:
                    ga.use_exact_token_count = self.args.exact_token_count

                # Configure sequence-aware diversity
                if self.args.no_sequence_diversity:
                    ga.use_sequence_aware_diversity = False
                else:
                    ga.use_sequence_aware_diversity = self.args.sequence_aware_diversity
                ga.sequence_diversity_ratio = max(0.0, min(1.0, self.args.sequence_diversity_ratio))

                # Configure shuffle mutation
                ga.enable_shuffle_mutation = self.args.enable_shuffle_mutation

                # Load model and tokens
                ga.load_model()
                # Handle cache clearing if requested
                if getattr(self.args, 'clear_cache', False):
                    from pathlib import Path
                    import shutil
                    cache_dir = Path("cache/comprehensive_search")
                    if cache_dir.exists():
                        shutil.rmtree(cache_dir)
                        print(f"Cleared comprehensive search cache: {cache_dir}")

                ga.load_glitch_tokens(
                    token_file=self.args.token_file,
                    ascii_only=self.args.ascii_only,
                    include_normal_tokens=getattr(self.args, 'include_normal_tokens', False),
                    comprehensive_search=getattr(self.args, 'comprehensive_search', False)
                )

                # Set cache preferences if specified
                if getattr(self.args, 'no_cache', False):
                    ga.use_cache = False

                if self.args.baseline_only:
                    # Only run token impact baseline analysis
                    print("Running token impact baseline analysis only...")
                    print(f"Model: {self.args.model_path}")
                    print(f"Base text: '{self.args.base_text}'")

                    target_id, target_prob, wanted_id, wanted_prob = ga.get_baseline_probability()
                    ga.target_token_id = target_id
                    ga.baseline_target_probability = target_prob
                    ga.wanted_token_id = wanted_id
                    ga.baseline_wanted_probability = wanted_prob or 0.0
                    ga.baseline_token_impacts()
                    ga.display_token_impact_results(top_n=self.args.baseline_top_n)
                    ga.save_token_impact_results(self.args.output)
                    self._inject_snapshot(self.args.output)

                    print(f"\nToken impact baseline results saved to {self.args.output}")
                else:
                    # Run genetic algorithm evolution
                    print("Running genetic algorithm evolution...")
                    print(f"Model: {self.args.model_path}")
                    print(f"Base text: '{self.args.base_text}'")
                    print(f"Population size: {self.args.population_size}")
                    print(f"Generations: {self.args.generations}")

                    if self.args.skip_baseline:
                        print("Skipping token impact baseline analysis...")
                        # Temporarily disable baseline in run_evolution by modifying the method
                        original_baseline_method = ga.baseline_token_impacts
                        ga.baseline_token_impacts = lambda: None  # No-op function
                        final_population = ga.run_evolution()
                        ga.baseline_token_impacts = original_baseline_method  # Restore original
                    else:
                        print("Including token impact baseline analysis...")
                        final_population = ga.run_evolution()
                        ga.save_token_impact_results(self.args.baseline_output)
                        self._inject_snapshot(self.args.baseline_output)
                        print(f"Token impact baseline results saved to {self.args.baseline_output}")

                    # Prepare results
                    results = {
                        'model_name': self.args.model_path,
                        'base_text': self.args.base_text,
                        'target_token_id': ga.target_token_id,
                        'target_token_text': ga.tokenizer.decode([ga.target_token_id]) if ga.target_token_id else None,
                        'baseline_probability': ga.baseline_target_probability,
                        'ga_parameters': {
                            'population_size': ga.population_size,
                            'max_generations': ga.max_generations,
                            'mutation_rate': ga.mutation_rate,
                            'crossover_rate': ga.crossover_rate,
                            'max_tokens_per_individual': ga.max_tokens_per_individual
                        },
                        'results': []
                    }

                    # Deduplicate results by token combination
                    seen_combinations = set()
                    unique_population = []
                    for individual in final_population:
                        key = tuple(individual.tokens)
                        if key not in seen_combinations:
                            seen_combinations.add(key)
                            unique_population.append(individual)

                    # Add top results
                    for individual in unique_population[:10]:  # Top 10 unique results
                        token_texts = [ga.tokenizer.decode([token_id]) for token_id in individual.tokens]
                        results['results'].append({
                            'tokens': individual.tokens,
                            'token_texts': token_texts,
                            'fitness': individual.fitness,
                            'baseline_prob': individual.baseline_prob,
                            'modified_prob': individual.modified_prob,
                            'reduction_percentage': ((individual.baseline_prob - individual.modified_prob) / individual.baseline_prob * 100) if individual.baseline_prob > 0 else 0
                        })

                    # Inject environment snapshot
                    results["_environment"] = self._collect_snapshot()

                    # Save results
                    with open(self.args.output, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)

                    print(f"\nGenetic algorithm results saved to {self.args.output}")

                    # Display top results
                    print("\nTop Results:")
                    for i, individual in enumerate(unique_population[:5], 1):
                        token_texts = [ga.tokenizer.decode([token_id]) for token_id in individual.tokens]

                        if ga.target_token_id is None and ga.wanted_token_id is not None:
                            # Only wanted token specified - show wanted token increase
                            wanted_increase_pct = (individual.wanted_increase / (1.0 - individual.wanted_baseline_prob) * 100) if individual.wanted_baseline_prob < 1.0 else 0
                            print(f"{i}. Tokens: {individual.tokens} ({token_texts})")
                            print(f"   Fitness: {individual.fitness:.6f}")
                            print(f"   Wanted token increase: {wanted_increase_pct:.2f}%")
                            print(f"   Wanted probability: {individual.wanted_baseline_prob:.6f} → {individual.wanted_modified_prob:.6f}")
                        else:
                            # Target token reduction (original behavior)
                            reduction_pct = ((individual.baseline_prob - individual.modified_prob) / individual.baseline_prob * 100) if individual.baseline_prob > 0 else 0
                            print(f"{i}. Tokens: {individual.tokens} ({token_texts})")
                            print(f"   Fitness: {individual.fitness:.6f}")
                            print(f"   Probability reduction: {reduction_pct:.2f}%")
                        print()

                    # Keep GUI alive if it was used
                    if gui_callback:
                        print("GUI animation is live. Close the window when done viewing.")
                        try:
                            gui_callback.keep_alive()  # Keep alive until window closed
                        except KeyboardInterrupt:
                            print("GUI closed by user.")

        except Exception as e:
            print(f"ERROR: Error in genetic algorithm: {e}")
            import traceback
            traceback.print_exc()
            raise

    def run_gui(self):
        """Launch the graphical user interface for genetic algorithm control."""
        try:
            import tkinter as tk
            from tkinter import messagebox
            from .genetic.gui_controller import GeneticControllerGUI

            print("🚀 Launching Glitcher GUI...")

            # Create root window
            root = tk.Tk()
            root.title("Glitcher Genetic Algorithm Controller")
            root.geometry("1200x800")
            root.minsize(800, 600)

            # Center window on screen
            root.update_idletasks()
            width = root.winfo_width()
            height = root.winfo_height()
            x = (root.winfo_screenwidth() // 2) - (width // 2)
            y = (root.winfo_screenheight() // 2) - (height // 2)
            root.geometry(f"{width}x{height}+{x}+{y}")

            # Create the application
            app = GeneticControllerGUI(root)

            # Load configuration if provided
            if hasattr(self.args, 'config') and self.args.config:
                try:
                    import json
                    with open(self.args.config, 'r') as f:
                        config_dict = json.load(f)

                    # Update config object
                    for key, value in config_dict.items():
                        if hasattr(app.config, key):
                            setattr(app.config, key, value)

                    app.update_gui_from_config()
                    print(f"✅ Loaded configuration from {self.args.config}")
                except Exception as e:
                    print(f"⚠️  Failed to load configuration: {e}")

            # Setup cleanup handler
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

            print("✅ GUI ready!")
            print("\n📋 Usage Instructions:")
            print("   1. Configure parameters in the 'Configuration' tab")
            print("   2. Use the 'Control' tab to start/stop evolution")
            print("   3. Monitor progress in the 'Progress' tab")
            print("   4. View results in the 'Results' tab")
            print("\n🎯 Features:")
            print("   • Real-time parameter adjustment")
            print("   • Start/pause/stop controls")
            print("   • Live progress monitoring")
            print("   • Comprehensive search support")
            print("   • Configuration save/load")
            print("   • GUI animation integration")
            print()

            # Start the GUI main loop
            root.mainloop()

            print("👋 GUI closed.")

        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("GUI requires tkinter and matplotlib. Install with:")
            print("   pip install matplotlib")
            print("Note: tkinter is usually included with Python")
            raise
        except Exception as e:
            print(f"❌ Error launching GUI: {e}")
            import traceback
            traceback.print_exc()
            raise

    def run(self):
        """Run the CLI tool."""
        parser = self.setup_parser()
        self.args = parser.parse_args()

        # Fix random seeds early, before model loading or any randomized logic
        self._fix_seeds()

        if not self.args.command:
            parser.print_help()
            return

        # Validation command doesn't need model loading
        elif self.args.command == "validate":
            self.run_validation()
        elif self.args.command == "compare":
            self.run_comparison()
            return
        elif self.args.command == "domain":
            self.run_domain_extraction()
            return
        elif self.args.command == "classify":
            self.run_classification()
            return
        elif self.args.command == "genetic":
            self.run_genetic()
            return
        elif self.args.command == "gui":
            self.run_gui()
            return

        # Load model for other commands
        self.load_model()

        if self.args.command == "mine":
            self.run_mining()
        elif self.args.command == "test":
            self.run_token_test()
        elif self.args.command == "chat":
            self.run_chat_test()
        else:
            parser.print_help()


def main():
    """Main entry point for the CLI."""
    cli = GlitcherCLI()
    cli.run()


if __name__ == "__main__":
    main()
