#!/usr/bin/env python3
"""
Genetic Algorithm for Breeding Probability Reducer Token Combinations

This script uses genetic algorithms to evolve combinations of glitch tokens
that maximize probability reduction when inserted at the beginning of text.

The algorithm maintains a population of token combinations (1-N tokens each,
configurable via --max-tokens) and evolves them over generations using
selection, crossover, and mutation.

Author: Claude
Date: 2024
"""

import json
import random
import argparse
import logging
import time
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


@dataclass
class Individual:
    """Represents an individual in the genetic algorithm population."""
    tokens: List[int]  # 1-3 token IDs
    fitness: float = 0.0  # Combined fitness score
    baseline_prob: float = 0.0  # Original target probability
    modified_prob: float = 0.0  # Target probability after token insertion
    target_reduction: float = 0.0  # Target probability reduction
    wanted_baseline_prob: float = 0.0  # Original wanted probability
    wanted_modified_prob: float = 0.0  # Wanted probability after token insertion
    wanted_increase: float = 0.0  # Wanted probability increase
    new_top_tokens: Optional[List[Tuple[int, float]]] = None  # Top 10 tokens after applying evolved combination
    # Response generation for GUI display
    baseline_response: str = ""  # LLM response to baseline input
    current_response: str = ""  # LLM response to current evolved input
    full_input_string: str = ""  # Complete input string sent to model
    baseline_input_string: str = ""  # Baseline input string sent to model

    def __str__(self):
        return f"Individual(tokens={self.tokens}, fitness={self.fitness:.4f})"


class GeneticProbabilityReducer:
    """
    Genetic algorithm for evolving token combinations that reduce prediction probabilities.
    """

    def __init__(self, model_name: str, base_text: str, target_token: Optional[str] = None, wanted_token: Optional[str] = None, gui_callback=None):
        """
        Initialize the genetic algorithm.

        Args:
            model_name: HuggingFace model identifier
            base_text: Base text to test probability reduction on
            target_token: Specific token to target for reduction (auto-detected if None)
            wanted_token: Specific token to target for increase (optional)
            gui_callback: Optional GUI callback for real-time visualization

        Note:
            Token combination size is configurable via max_tokens_per_individual (default: 3).
            Use --max-tokens CLI argument to adjust this dynamically.
        """
        self.model_name = model_name
        self.base_text = base_text
        self.target_token = target_token
        self.wanted_token = wanted_token
        self.gui_callback = gui_callback

        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Token pools
        self.glitch_tokens: List[int] = []
        self.ascii_tokens: List[int] = []
        self.available_tokens: List[int] = []

        # GA parameters
        self.population_size = 50
        self.max_generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.elite_size = 2
        self.max_tokens_per_individual = 3  # Configurable via CLI --max-tokens
        self.early_stopping_threshold = 0.999  # Stop when reduction reaches 99.9%

        # Adaptive mutation parameters
        self.adaptive_mutation = True
        self.initial_mutation_rate = 0.8  # Start high for exploration
        self.final_mutation_rate = 0.1   # End low for exploitation
        self.current_mutation_rate = self.initial_mutation_rate

        # Target and wanted token information
        self.target_token_id: Optional[int] = None
        self.wanted_token_id: Optional[int] = None
        self.baseline_target_probability: float = 0.0
        self.baseline_wanted_probability: float = 0.0
        self.initial_top_tokens: List[Tuple[int, float]] = []  # Store initial top 10 tokens with probabilities
        self.token_impact_map: Dict[int, Dict[str, Any]] = {}  # Map of token_id -> impact metrics

        # Baseline-guided seeding configuration
        self.use_baseline_seeding: bool = True  # Enable baseline-guided population seeding by default
        self.baseline_seeding_ratio: float = 0.7  # Fraction of population to seed with baseline guidance

        # Token count configuration
        self.use_exact_token_count: bool = True  # Use exact max_tokens_per_individual (True) or variable 1-N (False)

        # Sequence-aware diversity configuration
        self.use_sequence_aware_diversity: bool = True  # Enable sequence-aware diversity injection
        self.sequence_diversity_ratio: float = 0.3  # Fraction of diversity injection to use sequence-aware strategies
        self.enable_shuffle_mutation: bool = False  # Enable shuffle mutation (disabled by default to preserve combinations)

        # Caching configuration
        self.cache_dir: Path = Path("cache/comprehensive_search")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache: bool = True  # Enable caching by default

        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        """Load the language model and tokenizer with chat template support."""
        self.logger.info(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Set up padding token if not available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)

        self.model.eval()

        # Detect if this is an instruct model and set up chat template
        self._setup_chat_template()

        self.logger.info(f"Model loaded on device: {self.device}")
        self.logger.info(f"Instruct model: {self.is_instruct_model}")
        if self.is_instruct_model:
            self.logger.info(f"Using chat template for proper conversation formatting")

    def _setup_chat_template(self):
        """Set up chat template and system prompt for instruct models."""
        # Check if this is an instruct model
        model_name_lower = self.model_name.lower()
        self.is_instruct_model = any(keyword in model_name_lower for keyword in
                                   ['instruct', 'chat', 'assistant', 'conv'])

        if self.is_instruct_model:
            # Create system prompt for continuation task
            self.system_prompt = (
                "You are a text completion assistant. Your task is to continue sequences of text. "
                "When the user provides text, you should predict and output the most likely next word(s) "
                "that would naturally follow in the sequence. Respond with only the continuation, "
                "without any additional explanation or formatting."
            )
            self.logger.info("Set up instruct model with continuation system prompt")
        else:
            self.system_prompt = None
            self.logger.info("Non-instruct model detected, using direct completion")

    def _format_input_for_model(self, modified_text: str) -> str:
        """Format input text according to model type (instruct vs base model)."""
        if not self.is_instruct_model or not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
            # For base models or models without chat templates, use direct text
            self.logger.debug(f"Using direct text format: '{modified_text}'")
            return modified_text

        # For instruct models, format as conversation
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append({"role": "user", "content": modified_text})

        try:
            # Use the model's chat template
            formatted_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            self.logger.debug(f"Formatted instruct input: '{formatted_input}'")
            return formatted_input
        except Exception as e:
            self.logger.warning(f"Failed to apply chat template: {e}, falling back to direct text")
            return modified_text

    def _get_assistant_start_position(self, formatted_input: str) -> int:
        """Get the token position where the assistant response should start."""
        if not self.is_instruct_model:
            # For base models, we want the position after the input
            input_tokens = self.tokenizer.encode(formatted_input, add_special_tokens=False)
            return len(input_tokens)

        # For instruct models, find where assistant response would start
        # This is right after the generation prompt
        input_tokens = self.tokenizer.encode(formatted_input, add_special_tokens=False)
        return len(input_tokens)

    def load_glitch_tokens(self, token_file: Optional[str] = None, ascii_only: bool = False, include_normal_tokens: bool = False, comprehensive_search: bool = False):
        """
        Load glitch tokens from JSON file and optionally normal ASCII tokens from vocabulary.

        Args:
            token_file: Path to JSON file containing glitch tokens (optional)
            ascii_only: If True, filter to only include tokens with ASCII-only decoded text
            include_normal_tokens: If True, also include normal ASCII tokens from vocabulary
            comprehensive_search: If True, load full vocabulary for comprehensive wanted token search
        """
        # Initialize token lists
        self.glitch_tokens = []
        self.ascii_tokens = []

        # Load glitch tokens if file provided
        if token_file:
            self.logger.info(f"Loading glitch tokens from: {token_file}")
            try:
                with open(token_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    # List of token IDs
                    self.glitch_tokens = data
                elif isinstance(data, dict):
                    # Dictionary format - extract token IDs
                    if 'classifications' in data:
                        # Format: {"classifications": [{"token_id": 123, ...}, ...]}
                        self.glitch_tokens = [t['token_id'] for t in data['classifications'] if 'token_id' in t]
                    elif 'tokens' in data:
                        self.glitch_tokens = [t['id'] for t in data['tokens'] if 'id' in t]
                    else:
                        # Assume keys are token IDs
                        self.glitch_tokens = [int(k) for k in data.keys() if k.isdigit()]

                self.logger.info(f"Loaded {len(self.glitch_tokens)} glitch tokens")

            except Exception as e:
                self.logger.error(f"Error loading glitch tokens: {e}")
                self.glitch_tokens = []

        # Load normal ASCII tokens from model vocabulary if requested
        if include_normal_tokens or comprehensive_search:
            self.ascii_tokens = self._load_ascii_tokens_from_vocab(comprehensive=comprehensive_search)
            self.logger.info(f"Loaded {len(self.ascii_tokens)} ASCII tokens from model vocabulary")

        # Combine all available tokens
        all_tokens = list(set(self.glitch_tokens + self.ascii_tokens))

        # Filter to ASCII-only tokens if requested
        if ascii_only:
            original_count = len(all_tokens)
            all_tokens = self._filter_ascii_tokens(all_tokens)
            filtered_count = len(all_tokens)
            self.logger.info(f"ASCII filtering: {original_count} -> {filtered_count} tokens "
                           f"({original_count - filtered_count} non-ASCII tokens removed)")

        self.available_tokens = all_tokens

        if not self.available_tokens:
            raise ValueError("No tokens available after filtering")

        self.logger.info(f"Total available tokens: {len(self.available_tokens)} "
                        f"(glitch: {len(self.glitch_tokens)}, normal: {len(self.ascii_tokens)})")

    def _load_ascii_tokens_from_vocab(self, comprehensive: bool = False) -> List[int]:
        """
        Load ASCII-only tokens from the model's tokenizer vocabulary.

        Args:
            comprehensive: If True, scan all tokens. If False, sample for speed.

        Returns:
            List of token IDs that decode to ASCII-only text
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be loaded first")

        ascii_tokens = []
        vocab_size = len(self.tokenizer.get_vocab())

        if comprehensive:
            self.logger.info(f"Comprehensive scan: testing all {vocab_size} tokens in vocabulary...")
            sample_rate = 1  # Test every token
        else:
            self.logger.info(f"Scanning {vocab_size} tokens in vocabulary for ASCII-only tokens...")
            sample_rate = max(1, vocab_size // 10000)  # Sample at most 10k tokens

        for token_id in range(0, vocab_size, sample_rate):
            try:
                # Decode token and check if ASCII-only
                if self._is_ascii_only([token_id]):
                    decoded = self.tokenizer.decode([token_id])
                    # Filter out special tokens and empty/whitespace-only tokens
                    if (not decoded.startswith('<') and not decoded.startswith('[') and
                        decoded.strip() and len(decoded.strip()) > 0):
                        ascii_tokens.append(token_id)
            except Exception:
                # Skip tokens that can't be decoded
                continue

        return ascii_tokens

    def _filter_ascii_tokens(self, token_ids: List[int]) -> List[int]:
        """
        Filter token IDs to only include those with ASCII-only decoded text.

        Args:
            token_ids: List of token IDs to filter

        Returns:
            Filtered list of token IDs with ASCII-only decoded text
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be loaded first")

        ascii_tokens = []
        for token_id in token_ids:
            if self._is_ascii_only([token_id]):
                ascii_tokens.append(token_id)

        return ascii_tokens

    def _is_ascii_only(self, token_ids: List[int]) -> bool:
        """
        Check if the decoded text of token IDs contains only ASCII characters.

        Args:
            token_ids: List of token IDs to check

        Returns:
            True if decoded text is ASCII-only, False otherwise
        """
        try:
            if self.tokenizer is not None:
                decoded = self.tokenizer.decode(token_ids)
            else:
                return False
            # Check if all characters are ASCII (0-127)
            return all(ord(char) < 128 for char in decoded)
        except Exception:
            return False

    def get_baseline_probability(self) -> Tuple[Optional[int], float, Optional[int], Optional[float]]:
        """
        Get baseline probabilities for target and wanted tokens.

        Returns:
            Tuple of (target_token_id, target_probability, wanted_token_id, wanted_probability)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")

        self.logger.info(f"Getting baseline probability for: {self.base_text}")

        # Format input according to model type
        formatted_input = self._format_input_for_model(self.base_text)
        self.logger.info(f"Baseline formatted input: '{formatted_input}'")

        # Tokenize formatted input
        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.device)

        # Get the position where we want to measure probabilities
        assistant_start_pos = self._get_assistant_start_position(formatted_input)

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            # For instruct models, measure at assistant start position (last position)
            # For base models, measure at the end of input
            logits = outputs.logits[0, -1, :]  # Last position logits
            probs = torch.softmax(logits, dim=-1)

        # Capture initial top 10 tokens for comparison
        top_probs, top_indices = torch.topk(probs, 100)
        self.initial_top_tokens = [(int(idx.item()), float(prob.item())) for idx, prob in zip(top_indices, top_probs)]
        # Log initial top 10 tokens with decoded text for readability
        initial_tokens_readable = []
        for idx, prob in self.initial_top_tokens[:100]:
            token_text = self.tokenizer.decode([idx])
            initial_tokens_readable.append(f"'{token_text}'")
        self.logger.info(f"Initial top 10 tokens: {', '.join(initial_tokens_readable)}")

        # Handle target token (for reduction)
        if self.target_token:
            # Use specified target token
            target_tokens = self.tokenizer.encode(self.target_token, add_special_tokens=False)
            if target_tokens:
                target_id = target_tokens[0]
                target_prob = probs[int(target_id)].item()
            else:
                raise ValueError(f"Could not tokenize target: {self.target_token}")
        elif not self.wanted_token:
            # Use most likely token only if no wanted token specified
            target_id = int(torch.argmax(probs).item())
            target_prob = probs[target_id].item()
        else:
            # No target when only wanted token is specified
            target_id = None
            target_prob = 0.0

        if target_id is not None:
            if self.tokenizer is None:
                raise ValueError("Tokenizer must be loaded first")
            target_text = self.tokenizer.decode([target_id])
            if self.is_instruct_model:
                self.logger.info(f"Target baseline (instruct): '{self.base_text}' → '{target_text}' (ID: {target_id}, prob: {target_prob:.4f})")
            else:
                self.logger.info(f"Target baseline: '{self.base_text}' → '{target_text}' (ID: {target_id}, prob: {target_prob:.4f})")

        # Handle wanted token (for increase, optional)
        wanted_id = None
        wanted_prob = None
        if self.wanted_token:
            wanted_tokens = self.tokenizer.encode(self.wanted_token, add_special_tokens=False)
            if wanted_tokens:
                wanted_id = wanted_tokens[0]
                wanted_prob = probs[int(wanted_id)].item()
                wanted_text = self.tokenizer.decode([wanted_id])
                if self.is_instruct_model:
                    self.logger.info(f"Wanted baseline (instruct): '{self.base_text}' → '{wanted_text}' (ID: {wanted_id}, prob: {wanted_prob:.4f})")
                else:
                    self.logger.info(f"Wanted baseline: '{self.base_text}' → '{wanted_text}' (ID: {wanted_id}, prob: {wanted_prob:.4f})")
            else:
                raise ValueError(f"Could not tokenize wanted token: {self.wanted_token}")

        # Generate baseline response for GUI display
        self.baseline_response = self._generate_response(self.base_text)

        return target_id, target_prob, wanted_id, wanted_prob

    def _generate_cache_key(self, max_tokens: Optional[int] = None, batch_size: int = 16) -> str:
        """
        Generate a unique cache key for comprehensive search results.

        Args:
            max_tokens: Maximum number of tokens to test
            batch_size: Batch size used for processing

        Returns:
            Unique cache key string
        """
        # Create a hash based on search parameters
        cache_data = {
            'model_name': self.model_name,
            'base_text': self.base_text,
            'wanted_token': self.wanted_token,
            'wanted_token_id': self.wanted_token_id,
            'baseline_wanted_probability': self.baseline_wanted_probability,
            'available_tokens_count': len(self.available_tokens),
            'max_tokens': max_tokens,
            'batch_size': batch_size,
            'version': '1.0'  # Increment this if cache format changes
        }

        cache_string = json.dumps(cache_data, sort_keys=True)
        cache_hash = hashlib.md5(cache_string.encode()).hexdigest()
        return f"comprehensive_search_{cache_hash}.json"

    def _save_comprehensive_search_cache(self, results: Dict[int, Dict[str, Any]],
                                       max_tokens: Optional[int] = None,
                                       batch_size: int = 16) -> None:
        """
        Save comprehensive search results to cache.

        Args:
            results: Search results to cache
            max_tokens: Maximum tokens parameter used
            batch_size: Batch size parameter used
        """
        if not self.use_cache:
            return

        cache_key = self._generate_cache_key(max_tokens, batch_size)
        cache_file = self.cache_dir / cache_key

        cache_data = {
            'timestamp': time.time(),
            'model_name': self.model_name,
            'base_text': self.base_text,
            'wanted_token': self.wanted_token,
            'wanted_token_id': self.wanted_token_id,
            'baseline_wanted_probability': self.baseline_wanted_probability,
            'search_parameters': {
                'max_tokens': max_tokens,
                'batch_size': batch_size,
                'available_tokens_count': len(self.available_tokens)
            },
            'results': results
        }

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            self.logger.info("Comprehensive search results cached to: %s", cache_file)
        except Exception as e:
            self.logger.warning("Failed to save cache: %s", e)

    def _load_comprehensive_search_cache(self, max_tokens: Optional[int] = None,
                                       batch_size: int = 16) -> Optional[Dict[int, Dict[str, Any]]]:
        """
        Load comprehensive search results from cache if available and valid.

        Args:
            max_tokens: Maximum tokens parameter
            batch_size: Batch size parameter

        Returns:
            Cached search results or None if not available/invalid
        """
        if not self.use_cache:
            return None

        cache_key = self._generate_cache_key(max_tokens, batch_size)
        cache_file = self.cache_dir / cache_key

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Validate cache data
            if (cache_data.get('model_name') != self.model_name or
                cache_data.get('base_text') != self.base_text or
                cache_data.get('wanted_token') != self.wanted_token or
                cache_data.get('wanted_token_id') != self.wanted_token_id or
                abs(cache_data.get('baseline_wanted_probability', 0) - self.baseline_wanted_probability) > 1e-6):
                self.logger.info("Cache invalid: parameters changed")
                return None

            # Check cache age (optional - could add expiration)
            cache_age = time.time() - cache_data.get('timestamp', 0)
            cache_age_hours = cache_age / 3600

            # Convert string keys back to integers
            results = {}
            for token_id_str, metrics in cache_data['results'].items():
                results[int(token_id_str)] = metrics

            self.logger.info("Loaded comprehensive search results from cache (%d tokens, %.1f hours old)",
                           len(results), cache_age_hours)
            return results

        except Exception as e:
            self.logger.warning("Failed to load cache: %s", e)
            return None

    def comprehensive_wanted_token_search(self, max_tokens: Optional[int] = None, batch_size: int = 16) -> Dict[int, Dict[str, Any]]:
        """
        Perform comprehensive search through all available tokens to find best ones for wanted token increase.
        Uses optimized batching and early stopping for efficiency. Results are cached for faster subsequent runs.

        Args:
            max_tokens: Maximum number of tokens to test (None = test all available)
            batch_size: Number of tokens to process in each batch for efficiency

        Returns:
            Dictionary mapping token_id to impact metrics, sorted by wanted token impact
        """
        if self.wanted_token_id is None:
            raise ValueError("Wanted token must be specified for comprehensive search")

        if not self.available_tokens:
            raise ValueError("No tokens available for comprehensive search")

        if self.tokenizer is None or self.model is None:
            raise ValueError("Model and tokenizer must be loaded first")

        # Check cache first
        cached_results = self._load_comprehensive_search_cache(max_tokens, batch_size)
        if cached_results is not None:
            wanted_token_text = "unknown"
            if self.tokenizer and self.wanted_token_id is not None:
                wanted_token_text = self.tokenizer.decode([self.wanted_token_id])
            self.logger.info("Using cached comprehensive search results")

            # Update available tokens with cached results
            top_tokens = list(cached_results.items())[:50]
            high_impact_tokens = [token_id for token_id, metrics in top_tokens if metrics.get('wanted_impact', 0) > 0]
            if high_impact_tokens:
                remaining_tokens = [t for t in self.available_tokens if t not in high_impact_tokens]
                self.available_tokens = high_impact_tokens + remaining_tokens
                self.logger.info("Reordered available tokens using cached results: %d high-impact tokens prioritized", len(high_impact_tokens))

            return cached_results

        # Use all available tokens if max_tokens not specified
        test_tokens = self.available_tokens
        if max_tokens is not None:
            test_tokens = self.available_tokens[:max_tokens]

        wanted_token_text = "unknown"
        if self.tokenizer and self.wanted_token_id is not None:
            wanted_token_text = self.tokenizer.decode([self.wanted_token_id])

        self.logger.info("Comprehensive wanted token search: testing %d tokens", len(test_tokens))
        self.logger.info("Target wanted token: %s", wanted_token_text)
        self.logger.info("Using batch processing with batch_size=%d", batch_size)

        token_impacts = {}
        best_impact = 0.0
        best_tokens = []
        excellent_tokens = []  # Tokens with >90% impact

        # Progress tracking
        total_tokens = len(test_tokens)
        processed = 0

        # Process tokens in batches for efficiency
        for batch_start in range(0, total_tokens, batch_size):
            batch_end = min(batch_start + batch_size, total_tokens)
            batch_tokens = test_tokens[batch_start:batch_end]

            try:
                # Process batch
                batch_results = self._process_token_batch_for_wanted(batch_tokens)

                # Allow GUI updates every batch to maintain responsiveness
                if self.gui_callback and batch_start % (batch_size * 5) == 0:  # Update every 5 batches
                    try:
                        import matplotlib.pyplot as plt
                        if hasattr(plt, 'get_fignums') and plt.get_fignums():
                            plt.pause(0.001)
                    except Exception:
                        pass

                # Update results and tracking
                for token_id, metrics in batch_results.items():
                    token_impacts[token_id] = metrics
                    wanted_impact = metrics['wanted_impact']

                    # Track best tokens
                    if wanted_impact > best_impact:
                        best_impact = wanted_impact
                        best_tokens = [(token_id, metrics['token_text'], wanted_impact, metrics['wanted_prob_after'])]
                    elif wanted_impact == best_impact and len(best_tokens) < 5:
                        best_tokens.append((token_id, metrics['token_text'], wanted_impact, metrics['wanted_prob_after']))

                    # Track excellent tokens (>90% normalized impact)
                    if metrics['wanted_normalized'] > 0.9:
                        excellent_tokens.append((token_id, metrics['token_text'], wanted_impact))

                processed += len(batch_tokens)

                # Enhanced progress logging every 500 tokens
                if processed % 500 == 0 or batch_end == total_tokens:
                    progress_pct = (processed / total_tokens) * 100
                    self.logger.info("Progress: %d/%d (%.1f%%) | Best impact: %.6f | Excellent tokens found: %d",
                                   processed, total_tokens, progress_pct, best_impact, len(excellent_tokens))

                    # Update GUI during comprehensive search to prevent freezing
                    if self.gui_callback:
                        try:
                            # Create progress update data for GUI
                            progress_data = {
                                'phase': 'comprehensive_search',
                                'progress_pct': progress_pct,
                                'tokens_processed': processed,
                                'total_tokens': total_tokens,
                                'best_impact': best_impact,
                                'excellent_tokens': len(excellent_tokens)
                            }
                            # Update GUI with progress information
                            if hasattr(self.gui_callback, 'animator') and hasattr(self.gui_callback.animator, 'update_comprehensive_search_progress'):
                                self.gui_callback.animator.update_comprehensive_search_progress(progress_data)
                            else:
                                # Fallback to basic matplotlib update
                                import matplotlib.pyplot as plt
                                if hasattr(plt, 'get_fignums') and plt.get_fignums():
                                    plt.pause(0.001)
                        except Exception:
                            pass  # Ignore GUI update errors

                # Early stopping if we have many excellent tokens
                if len(excellent_tokens) >= 50:
                    self.logger.info("Early stopping: found %d excellent tokens (>90%% impact)", len(excellent_tokens))
                    break

            except Exception as e:
                self.logger.warning(f"Error processing batch {batch_start}-{batch_end}: {e}")
                continue

        # Sort by wanted impact (descending)
        sorted_impacts = dict(sorted(token_impacts.items(), key=lambda x: x[1]['wanted_impact'], reverse=True))

        # Enhanced results logging
        top_tokens = list(sorted_impacts.items())[:20]  # Show top 20
        wanted_token_text = self.tokenizer.decode([self.wanted_token_id]) if self.tokenizer and self.wanted_token_id is not None else "unknown"
        self.logger.info("Top 20 tokens for wanted '%s':", wanted_token_text)
        for i, (token_id, metrics) in enumerate(top_tokens, 1):
            impact_indicator = "[EXCELLENT]" if metrics['wanted_normalized'] > 0.9 else "[GOOD]" if metrics['wanted_normalized'] > 0.5 else ""
            self.logger.info("  %2d. Token %6d '%-20s' Impact: %8.6f (%.1f%%) Prob: %.4f -> %.4f %s",
                           i, token_id, metrics['token_text'], metrics['wanted_impact'],
                           metrics['wanted_normalized']*100, metrics['wanted_prob_before'],
                           metrics['wanted_prob_after'], impact_indicator)

        # Update available tokens to prioritize high-impact tokens more aggressively
        high_impact_tokens = [token_id for token_id, metrics in top_tokens[:50] if metrics['wanted_impact'] > 0]  # Top 50 positive impact
        if high_impact_tokens:
            # Put high-impact tokens at the beginning of available_tokens
            remaining_tokens = [t for t in self.available_tokens if t not in high_impact_tokens]
            self.available_tokens = high_impact_tokens + remaining_tokens
            self.logger.info("Reordered available tokens: %d high-impact tokens prioritized (top 50 positive)", len(high_impact_tokens))

        # Summary statistics
        positive_impacts = [m['wanted_impact'] for m in token_impacts.values() if m['wanted_impact'] > 0]
        if positive_impacts:
            avg_positive_impact = sum(positive_impacts) / len(positive_impacts)
            self.logger.info("Search summary: %d positive tokens found | Average positive impact: %.6f | Best impact: %.6f",
                           len(positive_impacts), avg_positive_impact, best_impact)
        else:
            self.logger.info("Search summary: No positive impact tokens found from %d tested tokens", len(token_impacts))

        # Save results to cache
        self._save_comprehensive_search_cache(sorted_impacts, max_tokens, batch_size)

        return sorted_impacts

    def _process_token_batch_for_wanted(self, token_batch: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Process a batch of tokens for wanted token impact analysis.

        Args:
            token_batch: List of token IDs to process

        Returns:
            Dictionary mapping token_id to impact metrics
        """
        batch_results = {}

        for token_id in token_batch:
            try:
                # Test individual token impact on wanted token
                if self.tokenizer is None:
                    continue
                token_text = self.tokenizer.decode([token_id])

                # Try multiple positioning strategies for better coverage
                test_positions = [
                    f"{token_text} {self.base_text}",  # Prefix
                    f"{self.base_text} {token_text}",  # Suffix
                    f"{token_text}: {self.base_text}",  # Colon separator
                ]

                best_impact = -float('inf')
                best_prob = self.baseline_wanted_probability

                # Test different positions and keep the best result
                for modified_text in test_positions:
                    inputs = self.tokenizer(modified_text, return_tensors="pt").to(self.device)

                    with torch.no_grad():
                        if self.model is None:
                            continue
                        outputs = self.model(**inputs)
                        logits = outputs.logits[0, -1, :]
                        probs = torch.softmax(logits, dim=-1)

                    wanted_modified_prob = probs[self.wanted_token_id].item()
                    wanted_impact = wanted_modified_prob - self.baseline_wanted_probability

                    if wanted_impact > best_impact:
                        best_impact = wanted_impact
                        best_prob = wanted_modified_prob

                # Calculate normalized impact (how much closer to 1.0 we get)
                if self.baseline_wanted_probability < 1.0:
                    wanted_normalized = best_impact / (1.0 - self.baseline_wanted_probability)
                else:
                    wanted_normalized = 0.0

                batch_results[token_id] = {
                    'token_text': token_text,
                    'wanted_impact': best_impact,
                    'wanted_normalized': wanted_normalized,
                    'wanted_prob_before': self.baseline_wanted_probability,
                    'wanted_prob_after': best_prob,
                }

            except Exception as e:
                # Skip problematic tokens but don't stop the batch
                continue

        return batch_results

    def baseline_token_impacts(self, max_tokens: int = 500) -> Dict[int, Dict[str, Any]]:
        """
        Calculate baseline impact of individual tokens on target probability.

        Args:
            max_tokens: Maximum number of tokens to test

        Returns:
            Dictionary mapping token_id to impact metrics
        """
        if not self.available_tokens:
            raise ValueError("No tokens available for baseline analysis")

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be loaded first")

        if self.model is None:
            raise ValueError("Model must be loaded first")

        self.logger.info(f"Computing baseline token impacts for {min(len(self.available_tokens), max_tokens)} tokens...")

        token_impacts = {}
        test_tokens = self.available_tokens[:max_tokens]

        for token_id in tqdm(test_tokens, desc="Baseline analysis"):
            try:
                # Test individual token impact
                token_text = self.tokenizer.decode([token_id])
                modified_text = f"({token_text}): {self.base_text}"

                inputs = self.tokenizer(modified_text, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits[0, -1, :]
                    probs = torch.softmax(logits, dim=-1)

                target_modified_prob = 0.0
                target_impact = 0.0
                if self.target_token_id is not None:
                    target_modified_prob = probs[self.target_token_id].item()
                    target_impact = self.baseline_target_probability - target_modified_prob

                wanted_modified_prob = 0.0
                wanted_impact = 0.0
                if self.wanted_token_id is not None:
                    wanted_modified_prob = probs[self.wanted_token_id].item()
                    wanted_impact = wanted_modified_prob - self.baseline_wanted_probability

                # Calculate combined impact score
                target_normalized = target_impact / self.baseline_target_probability if self.target_token_id is not None and self.baseline_target_probability > 0 else 0
                wanted_normalized = wanted_impact / (1.0 - self.baseline_wanted_probability) if self.wanted_token_id is not None and self.baseline_wanted_probability < 1.0 else 0

                if self.target_token_id is None and self.wanted_token_id is not None:
                    # Only wanted token specified - focus purely on wanted impact
                    combined_impact = wanted_normalized
                elif self.wanted_token_id is not None and self.target_token_id is not None:
                    # Both tokens specified - equal weight
                    combined_impact = 0.5 * target_normalized + 0.5 * wanted_normalized
                else:
                    # Only target specified
                    combined_impact = target_normalized

                token_impacts[token_id] = {
                    'token_text': token_text,
                    'target_impact': target_impact,
                    'target_normalized': target_normalized,
                    'wanted_impact': wanted_impact,
                    'wanted_normalized': wanted_normalized,
                    'combined_impact': combined_impact,
                    'target_prob_before': self.baseline_target_probability,
                    'target_prob_after': target_modified_prob,
                    'wanted_prob_before': self.baseline_wanted_probability,
                    'wanted_prob_after': wanted_modified_prob
                }

            except Exception as e:
                self.logger.warning(f"Error testing token {token_id}: {e}")
                continue

        # Sort by combined impact
        sorted_impacts = dict(sorted(token_impacts.items(), key=lambda x: x[1]['combined_impact'], reverse=True))
        self.token_impact_map = sorted_impacts

        self.logger.info(f"Baseline analysis complete. Top token impact: {list(sorted_impacts.values())[0]['combined_impact']:.4f}")
        return sorted_impacts

    def _generate_response(self, input_text: str, max_new_tokens: int = 50) -> str:
        """
        Generate a text response from the model given input text.

        Args:
            input_text: The input text to generate from
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Generated response text
        """
        try:
            if self.tokenizer is None or self.model is None:
                return ""

            # Format input according to model type
            formatted_input = self._format_input_for_model(input_text)
            inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # Generate response
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                # Decode only the new tokens (response part)
                input_length = inputs.input_ids.shape[1]
                response_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

                return response.strip()

        except Exception as e:
            self.logger.warning(f"Error generating response for input '{input_text[:50]}...': {e}")
            return ""

    def evaluate_fitness(self, individual: Individual) -> float:
        """
        Evaluate fitness of an individual (multi-objective: reduce target, increase wanted).

        Args:
            individual: Individual to evaluate

        Returns:
            Fitness score combining target reduction and wanted increase
        """
        # Generate baseline response if not already done
        if not hasattr(self, 'baseline_response'):
            self.baseline_response = self._generate_response(self.base_text)

        # Create modified text with tokens at beginning
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be loaded first")
        token_texts = [self.tokenizer.decode([tid]) for tid in individual.tokens]
        joined_tokens = "".join(token_texts)

        # For instruct models, include the glitch tokens in the user message
        # For base models, use the original format
        if self.is_instruct_model:
            modified_text = f"{joined_tokens} {self.base_text}".strip()
        else:
            modified_text = f"({joined_tokens}): {self.base_text}"

        try:
            if self.tokenizer is None or self.model is None:
                raise ValueError("Model and tokenizer must be loaded first")

            # Format input according to model type
            formatted_input = self._format_input_for_model(modified_text)
            self.logger.debug(f"Fitness evaluation input: '{formatted_input}'")
            inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Measure probabilities at the assistant response position
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)

            # Get target token probability (if target token is specified)
            target_modified_prob = 0.0
            target_reduction = 0.0
            if self.target_token_id is not None:
                target_modified_prob = probs[self.target_token_id].item()
                target_reduction = self.baseline_target_probability - target_modified_prob

            # Get wanted token probability (if specified)
            wanted_modified_prob = 0.0
            wanted_increase = 0.0
            if self.wanted_token_id is not None:
                wanted_modified_prob = probs[self.wanted_token_id].item()
                wanted_increase = wanted_modified_prob - self.baseline_wanted_probability

            # Capture new top 10 tokens after applying evolved combination
            top_probs, top_indices = torch.topk(probs, 10)
            individual.new_top_tokens = [(int(idx.item()), float(prob.item())) for idx, prob in zip(top_indices, top_probs)]

            # Calculate normalized fitness scores
            target_fitness = target_reduction / self.baseline_target_probability if self.baseline_target_probability > 0 else 0

            wanted_fitness = 0.0
            if self.wanted_token_id is not None and self.baseline_wanted_probability < 1.0:
                wanted_fitness = wanted_increase / (1.0 - self.baseline_wanted_probability)  # Normalize by potential increase

            # Calculate combined fitness based on what tokens are specified
            if self.target_token_id is None and self.wanted_token_id is not None:
                # Only wanted token specified - focus purely on maximizing wanted probability
                fitness = wanted_fitness
            elif self.wanted_token_id is not None and self.target_token_id is not None:
                # Both tokens specified - equal weight to both objectives
                fitness = 0.5 * target_fitness + 0.5 * wanted_fitness
            else:
                # Only target specified - focus on reduction
                fitness = target_fitness

            # Generate responses for GUI display
            individual.baseline_response = getattr(self, 'baseline_response', '')
            individual.current_response = self._generate_response(modified_text)
            individual.full_input_string = formatted_input
            individual.baseline_input_string = self._format_input_for_model(self.base_text)

            # Store metrics for display
            individual.baseline_prob = self.baseline_target_probability
            individual.modified_prob = target_modified_prob
            individual.target_reduction = target_reduction
            individual.wanted_baseline_prob = self.baseline_wanted_probability
            individual.wanted_modified_prob = wanted_modified_prob
            individual.wanted_increase = wanted_increase
            individual.fitness = fitness

            return fitness

        except Exception as e:
            self.logger.warning(f"Error evaluating individual {individual.tokens}: {e}")
            individual.fitness = -1.0  # Penalty for invalid combinations
            return -1.0

    def create_random_individual(self) -> Individual:
        """Create a random individual with random token combination."""
        if self.use_exact_token_count:
            num_tokens = self.max_tokens_per_individual
        else:
            num_tokens = random.randint(1, self.max_tokens_per_individual)

        if len(self.available_tokens) < num_tokens:
            tokens = self.available_tokens.copy()
        else:
            tokens = random.sample(self.available_tokens, num_tokens)
        return Individual(tokens=tokens)

    def create_baseline_guided_individual(self) -> Individual:
        """
        Create an individual guided by baseline token impact analysis.
        Uses weighted selection favoring high-impact tokens.
        """
        if not self.token_impact_map:
            return self.create_random_individual()

        if self.use_exact_token_count:
            num_tokens = self.max_tokens_per_individual
        else:
            num_tokens = random.randint(1, self.max_tokens_per_individual)

        # Get tokens sorted by impact
        impact_tokens = list(self.token_impact_map.keys())
        impacts = [self.token_impact_map[tid]['combined_impact'] for tid in impact_tokens]

        # Create weights (higher impact = higher probability)
        min_impact = min(impacts) if impacts else 0
        weights = [max(0.01, impact - min_impact + 0.1) for impact in impacts]

        # Weighted selection without replacement
        selected_tokens = []
        available_tokens = impact_tokens.copy()
        available_weights = weights.copy()

        for _ in range(num_tokens):
            if not available_tokens:
                break

            # Weighted random selection
            selected_idx = random.choices(range(len(available_tokens)), weights=available_weights)[0]
            selected_tokens.append(available_tokens.pop(selected_idx))
            available_weights.pop(selected_idx)

        return Individual(tokens=selected_tokens)

    def create_elite_seeded_individual(self, strategy: str = "singles") -> Individual:
        """
        Create an individual seeded with elite combinations from baseline analysis.

        Args:
            strategy: Seeding strategy ("singles", "pairs", "combinations")
        """
        if not self.token_impact_map:
            return self.create_random_individual()

        top_tokens = list(self.token_impact_map.keys())[:20]  # Top 20 tokens

        if strategy == "singles":
            # Best individual tokens
            if self.use_exact_token_count:
                num_tokens = self.max_tokens_per_individual
            else:
                num_tokens = random.randint(1, self.max_tokens_per_individual)
            tokens = top_tokens[:num_tokens]

        elif strategy == "pairs":
            # Best token pairs
            num_pairs = min(self.max_tokens_per_individual // 2, len(top_tokens) // 2)
            tokens = []
            for i in range(num_pairs):
                tokens.extend(top_tokens[i*2:(i+1)*2])

        elif strategy == "combinations":
            # Random combinations of top tokens
            if self.use_exact_token_count:
                num_tokens = self.max_tokens_per_individual
            else:
                num_tokens = random.randint(1, self.max_tokens_per_individual)
            tokens = random.sample(top_tokens[:min(len(top_tokens), num_tokens * 3)], num_tokens)

        else:
            return self.create_random_individual()

        return Individual(tokens=tokens)

    def create_initial_population(self) -> List[Individual]:
        """Create initial population with baseline-guided seeding."""
        population = []
        seeded_count = 0

        if self.use_baseline_seeding and self.token_impact_map:
            # Calculate seeded population size
            seeded_size = int(self.population_size * self.baseline_seeding_ratio)

            # Create seeded individuals using multiple strategies
            elite_singles = max(1, seeded_size // 5)
            elite_pairs = max(1, seeded_size // 5)
            elite_combinations = max(1, seeded_size // 4)
            baseline_guided = seeded_size - elite_singles - elite_pairs - elite_combinations

            # Elite singles (best individual tokens)
            for _ in range(elite_singles):
                population.append(self.create_elite_seeded_individual("singles"))
                seeded_count += 1

            # Elite pairs (best token pairs)
            for _ in range(elite_pairs):
                population.append(self.create_elite_seeded_individual("pairs"))
                seeded_count += 1

            # Elite combinations (best token combinations)
            for _ in range(elite_combinations):
                population.append(self.create_elite_seeded_individual("combinations"))
                seeded_count += 1

            # Baseline-guided weighted selection
            for _ in range(baseline_guided):
                population.append(self.create_baseline_guided_individual())
                seeded_count += 1

            self.logger.info(f"✓ Population seeded with {seeded_count} individuals using baseline guidance")

        # Fill remaining population with random individuals
        random_count = self.population_size - len(population)
        for _ in range(random_count):
            population.append(self.create_random_individual())

        self.logger.info(f"✓ Created initial population: {seeded_count} seeded + {random_count} random = {len(population)} total")

        return population

    def tournament_selection(self, population: List[Individual], tournament_size: int = 3) -> Individual:
        """Select individual using tournament selection."""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Create offspring through crossover.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Tuple of two offspring
        """
        if random.random() > self.crossover_rate:
            return Individual(tokens=parent1.tokens.copy()), Individual(tokens=parent2.tokens.copy())

        # Combine unique tokens from both parents
        combined_tokens = list(set(parent1.tokens + parent2.tokens))

        if len(combined_tokens) < 2:
            return Individual(tokens=parent1.tokens.copy()), Individual(tokens=parent2.tokens.copy())

        # Create two offspring with different token combinations
        if self.use_exact_token_count:
            target_size = self.max_tokens_per_individual
        else:
            target_size = random.randint(1, self.max_tokens_per_individual)

        # Ensure we have enough tokens
        if len(combined_tokens) >= target_size:
            child1_tokens = random.sample(combined_tokens, target_size)
        else:
            child1_tokens = combined_tokens.copy()

        if len(combined_tokens) >= target_size:
            child2_tokens = random.sample(combined_tokens, target_size)
        else:
            child2_tokens = combined_tokens.copy()

        child1 = Individual(tokens=child1_tokens)
        child2 = Individual(tokens=child2_tokens)

        # Ensure exact token count if required
        if self.use_exact_token_count:
            child1 = self._ensure_exact_token_count(child1)
            child2 = self._ensure_exact_token_count(child2)

        return child1, child2

    def _ensure_exact_token_count(self, individual: Individual) -> Individual:
        """
        Ensure individual has exactly max_tokens_per_individual tokens.

        Args:
            individual: Individual to adjust

        Returns:
            Individual with exact token count
        """
        current_count = len(individual.tokens)
        target_count = self.max_tokens_per_individual

        if current_count == target_count:
            return individual

        if current_count < target_count:
            # Add random tokens
            needed = target_count - current_count
            available = [t for t in self.available_tokens if t not in individual.tokens]
            if len(available) >= needed:
                additional = random.sample(available, needed)
                individual.tokens.extend(additional)
        else:
            # Remove random tokens
            individual.tokens = random.sample(individual.tokens, target_count)

        return individual

    def create_sequence_variations(self, top_individuals: List[Individual]) -> List[Individual]:
        """
        Create sequence variations of top-performing individuals.

        Args:
            top_individuals: List of best performing individuals

        Returns:
            List of sequence-varied individuals
        """
        variations = []

        for individual in top_individuals[:5]:  # Use top 5
            tokens = individual.tokens.copy()

            # Create permutations
            if len(tokens) > 1:
                # Reverse order
                reversed_tokens = tokens[::-1]
                variations.append(Individual(tokens=reversed_tokens))

                # Random shuffle
                shuffled_tokens = tokens.copy()
                random.shuffle(shuffled_tokens)
                variations.append(Individual(tokens=shuffled_tokens))

        return variations

    def create_sequence_aware_individual(self, top_individuals: List[Individual]) -> Individual:
        """
        Create sequence-aware individual using tokens from top performers.

        Args:
            top_individuals: List of best performing individuals

        Returns:
            New individual with sequence-aware token selection
        """
        if not top_individuals:
            return self.create_random_individual()

        # Collect tokens from top individuals
        all_tokens = []
        for ind in top_individuals[:10]:
            all_tokens.extend(ind.tokens)

        # Remove duplicates while preserving some order preference
        unique_tokens = list(dict.fromkeys(all_tokens))

        if self.use_exact_token_count:
            num_tokens = self.max_tokens_per_individual
        else:
            num_tokens = random.randint(1, self.max_tokens_per_individual)

        # Select tokens with bias toward earlier positions (higher frequency in top individuals)
        if len(unique_tokens) >= num_tokens:
            selected_tokens = unique_tokens[:num_tokens]
        else:
            # Need more tokens, add random ones
            needed = num_tokens - len(unique_tokens)
            available = [t for t in self.available_tokens if t not in unique_tokens]
            additional = random.sample(available, min(needed, len(available)))
            selected_tokens = unique_tokens + additional

        # Sometimes shuffle for diversity
        if random.random() < 0.3:
            random.shuffle(selected_tokens)

        return Individual(tokens=selected_tokens)

    def inject_diversity(self, population: List[Individual], generation: int) -> List[Individual]:
        """Inject diversity into stagnated population."""
        self.logger.info(f"⚠️  Population stagnated for {self.stagnation_counter} generations - injecting diversity!")

        # Calculate how many individuals to replace (25-50% based on stagnation severity)
        if self.stagnation_counter >= 50:
            replacement_rate = 0.5
            self.logger.info("🔥 CRITICAL stagnation - replacing 50% of population")
        elif self.stagnation_counter >= 30:
            replacement_rate = 0.4
            self.logger.info("⚡ HIGH stagnation - replacing 40% of population")
        else:
            replacement_rate = 0.25
            self.logger.info("💧 MILD stagnation - replacing 25% of population")

        num_to_replace = max(2, int(len(population) * replacement_rate))

        # Keep the best individuals, replace the worst
        new_individuals = []

        # Strategy 1: Random individuals (30%)
        random_count = max(1, int(num_to_replace * 0.3))
        for _ in range(random_count):
            new_individuals.append(self.create_random_individual())

        # Strategy 2: Baseline-guided individuals (40%)
        if self.token_impact_map:
            guided_count = max(1, int(num_to_replace * 0.4))
            for _ in range(guided_count):
                new_individuals.append(self.create_baseline_guided_individual())

        # Strategy 3: Mutated versions of best individuals (30%)
        remaining = num_to_replace - len(new_individuals)
        for i in range(remaining):
            # Take one of the top individuals and heavily mutate it
            source = population[i % min(5, len(population))]
            mutated = Individual(tokens=source.tokens.copy())

            # Apply heavy mutation (multiple token replacements)
            num_mutations = random.randint(1, min(3, len(mutated.tokens)))
            for _ in range(num_mutations):
                if mutated.tokens and self.available_tokens:
                    idx = random.randint(0, len(mutated.tokens) - 1)
                    new_token = random.choice(self.available_tokens)
                    # Ensure diversity by avoiding current tokens
                    attempts = 0
                    while new_token in mutated.tokens and attempts < 10:
                        new_token = random.choice(self.available_tokens)
                        attempts += 1
                    mutated.tokens[idx] = new_token

            new_individuals.append(mutated)

        # Replace worst performers with new diverse individuals
        for i, new_individual in enumerate(new_individuals):
            if len(population) - num_to_replace + i < len(population):
                population[len(population) - num_to_replace + i] = new_individual

        self.logger.info(f"✅ Diversity injection complete - {len(new_individuals)} individuals replaced")
        return population

    def calculate_population_diversity(self, population: List[Individual]) -> float:
        """Calculate diversity metric for the population."""
        if not population:
            return 0.0

        unique_combinations = set()
        for individual in population:
            combination = tuple(sorted(individual.tokens))
            unique_combinations.add(combination)

        return len(unique_combinations) / len(population)

    def mutate(self, individual: Individual):
        """Mutate an individual by replacing tokens with enhanced strategies."""
        if random.random() > self.mutation_rate:
            return

        if not individual.tokens:
            return

        # Adaptive mutation - sometimes apply multiple mutations for diversity
        num_mutations = 1
        if hasattr(self, 'stagnation_counter') and self.stagnation_counter > 10:
            # More aggressive mutation when stagnating
            num_mutations = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]

        for _ in range(num_mutations):
            if not individual.tokens:
                break

            mutation_type = random.choices(['replace', 'swap'], weights=[0.9, 0.1])[0]

            if mutation_type == 'replace':
                # Replace a random token with a new one
                idx = random.randint(0, len(individual.tokens) - 1)
                new_token = random.choice(self.available_tokens)

                # Try to avoid duplicates and encourage diversity
                attempts = 0
                while new_token in individual.tokens and attempts < 10:
                    new_token = random.choice(self.available_tokens)
                    attempts += 1

                individual.tokens[idx] = new_token

            elif mutation_type == 'swap' and len(individual.tokens) >= 2:
                # Swap positions of two tokens
                idx1, idx2 = random.sample(range(len(individual.tokens)), 2)
                individual.tokens[idx1], individual.tokens[idx2] = individual.tokens[idx2], individual.tokens[idx1]

    def evolve_generation(self, population: List[Individual], generation: int = 0) -> List[Individual]:
        """Evolve one generation with stagnation detection and diversity injection."""
        # Evaluate fitness for all individuals
        for individual in population:
            if individual.fitness == 0.0:
                self.evaluate_fitness(individual)

        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Check for stagnation and inject diversity if needed
        if hasattr(self, 'last_best_fitness') and hasattr(self, 'stagnation_counter'):
            current_best = population[0].fitness
            if abs(current_best - self.last_best_fitness) < 1e-6:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                self.last_best_fitness = current_best

            # Inject diversity if stagnated
            if self.stagnation_counter >= 20:
                population = self.inject_diversity(population, generation)
                self.stagnation_counter = 0

        # Create new population
        new_population = []

        # Elitism - keep best individuals
        new_population.extend(population[:self.elite_size])

        # Generate offspring
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)

            child1, child2 = self.crossover(parent1, parent2)

            self.mutate(child1)
            self.mutate(child2)

            new_population.extend([child1, child2])

        return new_population[:self.population_size]

    def run_evolution(self) -> List[Individual]:
        """Run the genetic algorithm evolution."""
        self.logger.info("Starting genetic algorithm evolution")

        # Get baseline probabilities
        target_id, target_prob, wanted_id, wanted_prob = self.get_baseline_probability()
        self.target_token_id = target_id
        self.baseline_target_probability = target_prob
        self.wanted_token_id = wanted_id
        self.baseline_wanted_probability = wanted_prob or 0.0

        # Perform comprehensive wanted token search if wanted token is specified
        if self.wanted_token_id is not None:
            self.logger.info("Performing comprehensive wanted token search before genetic algorithm...")

            # Notify GUI that comprehensive search is starting
            if self.gui_callback:
                try:
                    # Update GUI to show comprehensive search phase
                    if hasattr(self.gui_callback, 'animator') and hasattr(self.gui_callback.animator, 'update_comprehensive_search_progress'):
                        start_data = {
                            'phase': 'comprehensive_search',
                            'progress_pct': 0,
                            'tokens_processed': 0,
                            'total_tokens': len(self.available_tokens),
                            'best_impact': 0,
                            'excellent_tokens': 0
                        }
                        self.gui_callback.animator.update_comprehensive_search_progress(start_data)
                    else:
                        # Fallback to basic matplotlib update
                        import matplotlib.pyplot as plt
                        if hasattr(plt, 'get_fignums') and plt.get_fignums():
                            plt.suptitle("Comprehensive Search in Progress...", fontsize=12)
                            plt.pause(0.001)
                except Exception:
                    pass

            comprehensive_results = self.comprehensive_wanted_token_search()

            # Save comprehensive search results
            if hasattr(self, 'token_impact_map'):
                self.token_impact_map.update(comprehensive_results)
            else:
                self.token_impact_map = comprehensive_results

            # Notify GUI that comprehensive search is complete
            if self.gui_callback:
                try:
                    if hasattr(self.gui_callback, 'animator') and hasattr(self.gui_callback.animator, 'fig'):
                        # Reset title for genetic algorithm phase
                        title = f"🧬 Genetic Algorithm Evolution"
                        if self.wanted_token:
                            title += f" - Maximizing '{self.wanted_token}'"
                        if self.target_token:
                            title += f" - Reducing '{self.target_token}'"
                        self.gui_callback.animator.fig.suptitle(title, fontsize=14, fontweight='bold', color='#4169E1')
                        self.gui_callback.animator.fig.canvas.draw()
                    else:
                        # Fallback to basic matplotlib update
                        import matplotlib.pyplot as plt
                        if hasattr(plt, 'get_fignums') and plt.get_fignums():
                            plt.suptitle("Starting Genetic Algorithm Evolution...", fontsize=12)
                            plt.pause(0.001)
                except Exception:
                    pass

        # Notify GUI of evolution start
        if self.gui_callback:
            baseline_data = {
                'target_baseline_prob': self.baseline_target_probability,
                'wanted_baseline_prob': self.baseline_wanted_probability,
                'target_token_id': self.target_token_id,
                'wanted_token_id': self.wanted_token_id,
                'initial_top_tokens': [self.tokenizer.decode([token_id]) for token_id, _ in self.initial_top_tokens[:10]] if hasattr(self, 'initial_top_tokens') and self.tokenizer else []
            }
            self.gui_callback.on_evolution_start(baseline_data)

        # Run baseline token impact analysis
        if not self.use_baseline_seeding:
            self.logger.info("Skipping baseline analysis (baseline seeding disabled)")
        else:
            self.baseline_token_impacts()

        # Create initial population
        population = self.create_initial_population()

        # Initialize stagnation tracking
        self.last_best_fitness = 0.0
        self.stagnation_counter = 0

        # Evolution loop
        for generation in tqdm(range(self.max_generations), desc="Evolving"):
            population = self.evolve_generation(population, generation)

            # Update GUI with current generation data
            if self.gui_callback:
                best_individual = max(population, key=lambda x: x.fitness)
                diversity = self.calculate_population_diversity(population)

                # Decode token texts for GUI display
                if best_individual.tokens and self.tokenizer:
                    try:
                        token_texts = [self.tokenizer.decode([token_id]) for token_id in best_individual.tokens]
                        # Store in a way that GUI can access (will be passed to callback)
                    except:
                        token_texts = [f"Token_{token_id}" for token_id in best_individual.tokens]

                self.gui_callback.on_generation_complete(generation, population, best_individual, diversity, self.stagnation_counter)

                # Allow GUI to update in real-time
                try:
                    import matplotlib.pyplot as plt
                    if plt.get_fignums():  # Check if GUI windows exist
                        plt.pause(0.001)  # Small pause to allow GUI updates
                except:
                    pass  # Ignore if matplotlib not available or GUI disabled

            # Log progress with diversity metrics
            if generation % 10 == 0:
                best_individual = max(population, key=lambda x: x.fitness)
                diversity = self.calculate_population_diversity(population)
                self.logger.info(
                    f"Generation {generation}: Best fitness = {best_individual.fitness:.4f}, "
                    f"Target reduction = {best_individual.target_reduction:.4f}, "
                    f"Wanted increase = {best_individual.wanted_increase:.4f}, "
                    f"Diversity = {diversity:.3f}, Stagnation = {self.stagnation_counter}"
                )

            # Check for early stopping
            best_fitness = max(ind.fitness for ind in population)
            if best_fitness >= self.early_stopping_threshold:
                self.logger.info(f"Early stopping at generation {generation}")
                break

            # Check for early stopping when only wanted token is specified and reaches near 100%
            if self.target_token_id is None and self.wanted_token_id is not None:
                best_individual = max(population, key=lambda x: x.fitness)
                if best_individual.wanted_modified_prob >= 0.99:
                    wanted_token_text = self.tokenizer.decode([self.wanted_token_id]) if self.tokenizer and self.wanted_token_id is not None else "unknown"
                    self.logger.info(f"Early stopping at generation {generation}: wanted token '{wanted_token_text}' reached {best_individual.wanted_modified_prob:.4f} probability (99%+)")
                    break

        # Final evaluation and sorting
        for individual in population:
            if individual.fitness == 0.0:
                self.evaluate_fitness(individual)

        population.sort(key=lambda x: x.fitness, reverse=True)

        # Notify GUI of evolution completion
        if self.gui_callback:
            self.gui_callback.on_evolution_complete(population)

        return population

    def display_results(self, population: List[Individual], top_n: int = 10):
        """Display the best results from evolution."""
        if self.target_token_id is None and self.wanted_token_id is not None:
            print(f"\n=== Top {top_n} Wanted Token Maximization Results ===")
        else:
            print(f"\n=== Top {top_n} Multi-Objective Results ===")
        print(f"Base text: '{self.base_text}'")

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be loaded first")

        if self.target_token_id is not None:
            target_text = self.tokenizer.decode([self.target_token_id])
            print(f"Target token (reduce): '{target_text}' (ID: {self.target_token_id})")
            print(f"Target baseline probability: {self.baseline_target_probability:.4f}")

        if self.wanted_token_id is not None:
            wanted_text = self.tokenizer.decode([self.wanted_token_id])
            print(f"Wanted token (increase): '{wanted_text}' (ID: {self.wanted_token_id})")
            print(f"Wanted baseline probability: {self.baseline_wanted_probability:.4f}")

        # Display initial baseline top 10 tokens
        if hasattr(self, 'initial_top_tokens') and self.initial_top_tokens:
            print("\nInitial baseline top 10 predicted tokens:")
            for j, (token_id, prob) in enumerate(self.initial_top_tokens[:10]):
                token_text = self.tokenizer.decode([token_id])
                print(f"  {j+1:2d}. '{token_text}' (ID: {token_id}) - {prob:.4f}")

        print()

        for i, individual in enumerate(population[:top_n]):
            if individual.fitness <= 0:
                continue

            token_texts = [self.tokenizer.decode([tid]) for tid in individual.tokens]
            token_repr = [repr(text) for text in token_texts]

            print(f"{i+1:2d}. Tokens: {individual.tokens} → {token_repr}")
            print(f"    Combined fitness: {individual.fitness:.4f}")

            if self.target_token_id is not None:
                print(f"    Target: {self.baseline_target_probability:.4f} → {individual.modified_prob:.4f} "
                      f"(reduction: {individual.target_reduction:.4f})")

            if self.wanted_token_id is not None:
                print(f"    Wanted: {self.baseline_wanted_probability:.4f} → {individual.wanted_modified_prob:.4f} "
                      f"(increase: {individual.wanted_increase:.4f})")

            # Display top 10 predicted tokens after applying this combination
            if individual.new_top_tokens:
                print(f"    Top 10 predicted tokens after applying combination:")
                for j, (token_id, prob) in enumerate(individual.new_top_tokens[:10]):
                    token_text = self.tokenizer.decode([token_id])
                    print(f"      {j+1:2d}. '{token_text}' (ID: {token_id}) - {prob:.4f}")
            print()

    def save_results(self, population: List[Individual], output_file: str):
        """Save results to JSON file."""
        results = {
            'model_name': self.model_name,
            'base_text': self.base_text,
            'target_token_id': self.target_token_id,
            'target_token_text': self.tokenizer.decode([self.target_token_id]) if self.target_token_id and self.tokenizer else None,
            'target_baseline_probability': self.baseline_target_probability,
            'wanted_token_id': self.wanted_token_id,
            'wanted_token_text': self.tokenizer.decode([self.wanted_token_id]) if self.wanted_token_id and self.tokenizer else None,
            'wanted_baseline_probability': self.baseline_wanted_probability,
            'ga_parameters': {
                'population_size': self.population_size,
                'max_generations': self.max_generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'max_tokens_per_individual': self.max_tokens_per_individual,
                'use_normal_tokens': len(self.ascii_tokens) > 0,
                'total_available_tokens': len(self.available_tokens)
            },
            'results': []
        }

        for individual in population:
            if self.tokenizer is None:
                continue
            token_texts = [self.tokenizer.decode([token_id]) for token_id in individual.tokens]
            results['results'].append({
                'tokens': individual.tokens,
                'token_texts': token_texts,
                'combined_fitness': individual.fitness,
                'target_baseline_prob': individual.baseline_prob,
                'target_modified_prob': individual.modified_prob,
                'target_reduction': individual.target_reduction,
                'wanted_baseline_prob': individual.wanted_baseline_prob,
                'wanted_modified_prob': individual.wanted_modified_prob,
                'wanted_increase': individual.wanted_increase
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results saved to: {output_file}")

    def save_token_impact_results(self, output_file: str):
        """Save token impact baseline results to JSON file."""
        if not hasattr(self, 'token_impact_map') or not self.token_impact_map:
            self.logger.warning("No token impact data available to save")
            return

        results = {
            'model_name': self.model_name,
            'base_text': self.base_text,
            'target_token_id': self.target_token_id,
            'target_token_text': self.tokenizer.decode([self.target_token_id]) if self.target_token_id and self.tokenizer else None,
            'target_baseline_probability': self.baseline_target_probability,
            'wanted_token_id': self.wanted_token_id,
            'wanted_token_text': self.tokenizer.decode([self.wanted_token_id]) if self.wanted_token_id and self.tokenizer else None,
            'wanted_baseline_probability': self.baseline_wanted_probability,
            'analysis_parameters': {
                'total_tokens_tested': len(self.token_impact_map),
                'total_available_tokens': len(self.available_tokens),
                'use_normal_tokens': len(self.ascii_tokens) > 0 if hasattr(self, 'ascii_tokens') else False
            },
            'token_impacts': []
        }

        # Convert token impact map to list format for JSON serialization
        for token_id, impact_data in self.token_impact_map.items():
            results['token_impacts'].append({
                'token_id': token_id,
                'token_text': impact_data['token_text'],
                'target_impact': impact_data['target_impact'],
                'target_normalized': impact_data['target_normalized'],
                'wanted_impact': impact_data['wanted_impact'],
                'wanted_normalized': impact_data['wanted_normalized'],
                'combined_impact': impact_data['combined_impact'],
                'target_prob_before': impact_data['target_prob_before'],
                'target_prob_after': impact_data['target_prob_after'],
                'wanted_prob_before': impact_data['wanted_prob_before'],
                'wanted_prob_after': impact_data['wanted_prob_after']
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Token impact results saved to: {output_file}")

    def display_token_impact_results(self, top_n: int = 10):
        """Display top N token impact results from baseline analysis."""
        if not hasattr(self, 'token_impact_map') or not self.token_impact_map:
            print("No token impact data available to display")
            return

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be loaded first")

        print(f"\n=== Top {top_n} Token Impact Results ===")
        print(f"Base text: '{self.base_text}'")

        target_text = self.tokenizer.decode([self.target_token_id]) if self.target_token_id else "Unknown"
        print(f"Target token: '{target_text}' (ID: {self.target_token_id})")
        print(f"Target baseline probability: {self.baseline_target_probability:.6f}")

        if self.wanted_token_id is not None:
            wanted_text = self.tokenizer.decode([self.wanted_token_id])
            print(f"Wanted token: '{wanted_text}' (ID: {self.wanted_token_id})")
            print(f"Wanted baseline probability: {self.baseline_wanted_probability:.6f}")

        print(f"\nToken Impact Analysis:")
        print(f"{'Rank':<4} {'Token ID':<8} {'Token Text':<30} {'Impact':<8} {'Target ΔP':<12} {'Wanted ΔP':<12}")
        print("-" * 80)

        # Get top N tokens by combined impact
        sorted_impacts = sorted(self.token_impact_map.items(),
                              key=lambda x: x[1]['combined_impact'],
                              reverse=True)

        for i, (token_id, impact_data) in enumerate(sorted_impacts[:top_n]):
            token_text = impact_data['token_text']
            # Truncate long token text for display
            if len(token_text) > 28:
                token_text = token_text[:25] + "..."

            print(f"{i+1:<4} {token_id:<8} {repr(token_text):<30} "
                  f"{impact_data['combined_impact']:<8.4f} "
                  f"{impact_data['target_impact']:<12.6f} "
                  f"{impact_data['wanted_impact']:<12.6f}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Multi-Objective Genetic Algorithm for Token Probability Manipulation"
    )
    parser.add_argument("model_name", help="HuggingFace model identifier")
    parser.add_argument("--base-text", default="The quick brown", help="Base text to test on")
    parser.add_argument("--target-token", help="Token to reduce probability (auto-detected if not provided)")
    parser.add_argument("--wanted-token", help="Token to increase probability (optional)")
    parser.add_argument("--token-file", help="JSON file containing glitch tokens (optional)")
    parser.add_argument("--include-normal-tokens", action="store_true",
                       help="Include normal ASCII tokens from model vocabulary")
    parser.add_argument("--comprehensive-search", action="store_true",
                       help="Perform comprehensive search through all vocabulary tokens (slower but thorough)")
    parser.add_argument("--ascii-only", action="store_true",
                       help="Filter to ASCII-only tokens")
    parser.add_argument("--population-size", type=int, default=50, help="Population size")
    parser.add_argument("--generations", type=int, default=100, help="Maximum generations")
    parser.add_argument("--mutation-rate", type=float, default=0.1, help="Mutation rate")
    parser.add_argument("--crossover-rate", type=float, default=0.7, help="Crossover rate")
    parser.add_argument("--elite-size", type=int, default=5, help="Elite size")
    parser.add_argument("--max-tokens", type=int, default=3, help="Maximum tokens per individual")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top results to display")
    parser.add_argument("--early-stopping-threshold", type=float, default=0.999,
                       help="Early stopping threshold for fitness")
    parser.add_argument("--baseline-seeding-ratio", type=float, default=0.7,
                       help="Fraction of population to seed with baseline guidance")
    parser.add_argument("--no-baseline-seeding", action="store_true",
                       help="Disable baseline-guided population seeding")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable caching of comprehensive search results")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear comprehensive search cache before running")

    args = parser.parse_args()

    # Create analyzer
    analyzer = GeneticProbabilityReducer(
        model_name=args.model_name,
        base_text=args.base_text,
        target_token=args.target_token,
        wanted_token=args.wanted_token
    )

    # Set GA parameters
    analyzer.population_size = args.population_size
    analyzer.max_generations = args.generations
    analyzer.mutation_rate = args.mutation_rate
    analyzer.crossover_rate = args.crossover_rate
    analyzer.elite_size = args.elite_size
    analyzer.max_tokens_per_individual = args.max_tokens
    analyzer.early_stopping_threshold = args.early_stopping_threshold

    # Set baseline seeding parameters
    if args.no_baseline_seeding:
        analyzer.use_baseline_seeding = False
    analyzer.baseline_seeding_ratio = max(0.0, min(1.0, args.baseline_seeding_ratio))

    try:
        # Handle cache clearing if requested
        if args.clear_cache:
            cache_dir = Path("cache/comprehensive_search")
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                print(f"Cleared comprehensive search cache: {cache_dir}")

        # Load model and tokenizer
        analyzer.load_model()

        # Load tokens
        analyzer.load_glitch_tokens(
            token_file=args.token_file,
            ascii_only=args.ascii_only,
            include_normal_tokens=args.include_normal_tokens,
            comprehensive_search=args.comprehensive_search
        )

        # Set cache preferences if specified
        if hasattr(args, 'no_cache') and args.no_cache:
            analyzer.use_cache = False

        # Run evolution
        final_population = analyzer.run_evolution()

        # Display results
        analyzer.display_results(final_population, top_n=args.top_n)

        # Save results if requested
        if args.output:
            analyzer.save_results(final_population, args.output)

    except Exception as e:
        analyzer.logger.error(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()
