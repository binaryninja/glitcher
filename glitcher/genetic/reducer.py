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
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm



@dataclass
class Individual:
    """Represents an individual in the genetic algorithm population."""
    tokens: List[int]  # 1-3 token IDs
    fitness: float = 0.0  # Probability reduction achieved
    baseline_prob: float = 0.0  # Original probability
    modified_prob: float = 0.0  # Probability after token insertion
    new_top_tokens: List[Tuple[int, float]] = None  # Top 10 tokens after applying evolved combination

    def __str__(self):
        return f"Individual(tokens={self.tokens}, fitness={self.fitness:.4f})"


class GeneticProbabilityReducer:
    """
    Genetic algorithm for evolving token combinations that reduce prediction probabilities.
    """

    def __init__(self, model_name: str, base_text: str, target_token: Optional[str] = None, gui_callback=None):
        """
        Initialize the genetic algorithm.

        Args:
            model_name: HuggingFace model identifier
            base_text: Base text to test probability reduction on
            target_token: Specific token to target (auto-detected if None)
            gui_callback: Optional GUI callback for real-time visualization

        Note:
            Token combination size is configurable via max_tokens_per_individual (default: 3).
            Use --max-tokens CLI argument to adjust this dynamically.
        """
        self.model_name = model_name
        self.base_text = base_text
        self.target_token = target_token
        self.gui_callback = gui_callback

        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Glitch tokens
        self.glitch_tokens: List[int] = []

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

        # Target information
        self.target_token_id: Optional[int] = None
        self.baseline_probability: float = 0.0
        self.initial_top_tokens: List[Tuple[int, float]] = []  # Store initial top 10 tokens with probabilities
        self.token_impact_map: Dict[int, Dict[str, Any]] = {}  # Map of token_id -> impact metrics

        # Baseline-guided seeding configuration
        self.use_baseline_seeding: bool = True  # Enable baseline-guided population seeding by default
        self.baseline_seeding_ratio: float = 0.7  # Fraction of population to seed with baseline guidance

        # Token count configuration
        self.use_exact_token_count: bool = True  # Use exact max_tokens_per_individual (True) or variable 1-N (False)

        # Sequence-aware diversity configuration
        self.use_sequence_aware_diversity: bool = True  # Enable sequence-aware diversity injection
        self.sequence_diversity_ratio: float = 0.6  # Fraction of diversity injection to use sequence-aware strategies

        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        """Load the language model and tokenizer."""
        self.logger.info(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager"  # Avoid SDPA issues
        )
        self.model.eval()

        self.logger.info(f"Model loaded on device: {self.device}")

    def load_glitch_tokens(self, token_file: str, ascii_only: bool = False):
        """
        Load glitch tokens from JSON file.

        Args:
            token_file: Path to JSON file containing glitch tokens
            ascii_only: If True, filter to only include tokens with ASCII-only decoded text
        """
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

            if not self.glitch_tokens:
                raise ValueError("No glitch tokens found in file")

            # Filter to ASCII-only tokens if requested
            if ascii_only:
                original_count = len(self.glitch_tokens)
                self.glitch_tokens = self._filter_ascii_tokens(self.glitch_tokens)
                filtered_count = len(self.glitch_tokens)
                self.logger.info(f"ASCII filtering: {original_count} -> {filtered_count} tokens "
                               f"({original_count - filtered_count} non-ASCII tokens removed)")

                if not self.glitch_tokens:
                    raise ValueError("No ASCII-only glitch tokens found after filtering")

        except Exception as e:
            self.logger.error(f"Error loading glitch tokens: {e}")
            raise

    def _filter_ascii_tokens(self, token_ids: List[int]) -> List[int]:
        """
        Filter token IDs to only include those with ASCII-only decoded text.

        Args:
            token_ids: List of token IDs to filter

        Returns:
            List of token IDs that decode to ASCII-only text
        """
        # Load tokenizer if not already loaded
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.logger.info(f"Loading tokenizer for ASCII filtering: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        ascii_tokens = []
        for token_id in token_ids:
            try:
                # Decode the token
                decoded_text = self.tokenizer.decode([token_id], skip_special_tokens=False)

                # Check if all characters are ASCII (0-127)
                if self._is_ascii_only(decoded_text):
                    ascii_tokens.append(token_id)
                else:
                    self.logger.debug(f"Filtered non-ASCII token {token_id}: '{decoded_text}' "
                                    f"(contains non-ASCII characters)")

            except Exception as e:
                self.logger.warning(f"Error decoding token {token_id}: {e}")
                continue

        return ascii_tokens

    def _is_ascii_only(self, text: str) -> bool:
        """
        Check if text contains only ASCII characters (0-127).

        Args:
            text: Text to check

        Returns:
            True if text contains only ASCII characters, False otherwise
        """
        try:
            # Attempt to encode as ASCII - will raise UnicodeEncodeError if non-ASCII
            text.encode('ascii')
            return True
        except UnicodeEncodeError:
            return False


    def get_baseline_probability(self) -> Tuple[int, float]:
        """
        Get baseline probability for the target token.

        Returns:
            Tuple of (target_token_id, probability)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")

        self.logger.info(f"Getting baseline probability for: {self.base_text}")

        # Tokenize base text
        inputs = self.tokenizer(self.base_text, return_tensors="pt").to(self.device)

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last position logits
            probs = torch.softmax(logits, dim=-1)

        # Capture initial top 10 tokens for comparison
        top_probs, top_indices = torch.topk(probs, 10)
        self.initial_top_tokens = [(int(idx.item()), float(prob.item())) for idx, prob in zip(top_indices, top_probs)]
        self.logger.info(f"Initial top 10 tokens captured: {[(idx, f'{prob:.4f}') for idx, prob in self.initial_top_tokens[:5]]}...")

        if self.target_token:
            # Use specified target token
            target_tokens = self.tokenizer.encode(self.target_token, add_special_tokens=False)
            if target_tokens:
                target_id = target_tokens[0]
                prob = probs[int(target_id)].item()
            else:
                raise ValueError(f"Could not tokenize target: {self.target_token}")
        else:
            # Use most likely token
            target_id = int(torch.argmax(probs).item())
            prob = probs[target_id].item()

        target_text = self.tokenizer.decode([target_id])
        self.logger.info(f"Baseline: '{self.base_text}' → '{target_text}' (ID: {target_id}, prob: {prob:.4f})")

        return target_id, prob

    def baseline_token_impacts(self) -> Dict[int, Dict[str, Any]]:
        """
        Baseline the individual impact of all glitch tokens on target probability.

        Returns:
            Dictionary mapping token_id -> {
                'token_text': str,
                'baseline_prob': float,
                'modified_prob': float,
                'impact': float,
                'reduction_ratio': float,
                'rank_by_impact': int
            }
        """
        if not self.glitch_tokens:
            raise ValueError("Glitch tokens must be loaded first")

        if self.target_token_id is None or self.baseline_probability == 0.0:
            raise ValueError("Baseline probability must be established first")

        self.logger.info(f"Baselining individual impact of {len(self.glitch_tokens)} tokens...")

        impact_results = {}

        # Test each token individually
        for token_id in tqdm(self.glitch_tokens, desc="Testing token impacts"):
            try:
                # Decode token text
                token_text = self.tokenizer.decode([token_id])

                # Create modified text with single token at beginning
                user_string = f"({token_text}): "
                modified_text = user_string + self.base_text

                # Get probability with this token
                inputs = self.tokenizer(modified_text, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits[0, -1, :]
                    probs = torch.softmax(logits, dim=-1)

                modified_prob = probs[self.target_token_id].item()
                impact = self.baseline_probability - modified_prob
                reduction_ratio = impact / self.baseline_probability if self.baseline_probability > 0 else 0.0

                impact_results[token_id] = {
                    'token_text': token_text,
                    'baseline_prob': self.baseline_probability,
                    'modified_prob': modified_prob,
                    'impact': impact,
                    'reduction_ratio': reduction_ratio,
                    'rank_by_impact': 0  # Will be filled after sorting
                }

            except Exception as e:
                self.logger.warning(f"Error testing token {token_id}: {e}")
                impact_results[token_id] = {
                    'token_text': self.tokenizer.decode([token_id]) if token_id < len(self.tokenizer) else f"<UNK:{token_id}>",
                    'baseline_prob': self.baseline_probability,
                    'modified_prob': self.baseline_probability,
                    'impact': 0.0,
                    'reduction_ratio': 0.0,
                    'rank_by_impact': len(self.glitch_tokens)
                }

        # Sort by impact and assign ranks
        sorted_by_impact = sorted(impact_results.items(), key=lambda x: x[1]['impact'], reverse=True)
        for rank, (token_id, data) in enumerate(sorted_by_impact, 1):
            impact_results[token_id]['rank_by_impact'] = rank

        # Store results
        self.token_impact_map = impact_results

        # Log top performers
        self.logger.info("Top 10 individual token impacts:")
        for i, (token_id, data) in enumerate(sorted_by_impact[:10]):
            self.logger.info(f"  {i+1:2d}. Token {token_id:6d} '{data['token_text'][:20]}' "
                           f"-> Impact: {data['impact']:.4f} ({data['reduction_ratio']:.1%})")

        # Log statistics
        positive_impacts = [d['impact'] for d in impact_results.values() if d['impact'] > 0]
        if positive_impacts:
            avg_positive_impact = sum(positive_impacts) / len(positive_impacts)
            max_impact = max(positive_impacts)
            self.logger.info(f"Impact statistics: {len(positive_impacts)}/{len(self.glitch_tokens)} tokens "
                           f"have positive impact (avg: {avg_positive_impact:.4f}, max: {max_impact:.4f})")

        return self.token_impact_map

    def evaluate_fitness(self, individual: Individual) -> float:
        """
        Evaluate fitness of an individual (probability reduction).

        Args:
            individual: Individual to evaluate

        Returns:
            Fitness score (probability reduction)
        """
        # Create modified text with tokens at beginning
        token_texts = [self.tokenizer.decode([tid]) for tid in individual.tokens]
        joined_tokens = "".join(token_texts)
        user_string = f"({joined_tokens}): "
        modified_text=user_string + self.base_text


        try:
            inputs = self.tokenizer(modified_text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)

            modified_prob = probs[self.target_token_id].item()

            # Capture new top 10 tokens after applying evolved combination
            top_probs, top_indices = torch.topk(probs, 10)
            individual.new_top_tokens = [(int(idx.item()), float(prob.item())) for idx, prob in zip(top_indices, top_probs)]

            # Fitness is probability reduction
            fitness = self.baseline_probability - modified_prob

            individual.baseline_prob = self.baseline_probability
            individual.modified_prob = modified_prob
            individual.fitness = fitness

            return fitness

        except Exception as e:
            self.logger.warning(f"Error evaluating individual {individual.tokens}: {e}")
            individual.fitness = -1.0  # Penalty for invalid combinations
            return -1.0

    def create_random_individual(self) -> Individual:
        """Create a random individual with exact or variable token count based on configuration."""
        if self.use_exact_token_count:
            num_tokens = self.max_tokens_per_individual
        else:
            num_tokens = random.randint(1, self.max_tokens_per_individual)
        tokens = random.sample(self.glitch_tokens, num_tokens)
        return Individual(tokens=tokens)

    def create_baseline_guided_individual(self, top_tokens: List[int]) -> Individual:
        """
        Create an individual guided by baseline impact results.

        Args:
            top_tokens: List of token IDs sorted by impact (best first)

        Returns:
            Individual with tokens selected based on impact weights
        """
        if self.use_exact_token_count:
            num_tokens = self.max_tokens_per_individual
        else:
            num_tokens = random.randint(1, self.max_tokens_per_individual)

        # Create weighted selection based on impact scores
        # Use exponential decay to favor top tokens while maintaining some diversity
        weights = []
        available_tokens = top_tokens[:min(len(top_tokens), 50)]  # Use top 50 to maintain diversity

        for i, token_id in enumerate(available_tokens):
            # Exponential decay weight: higher impact = higher weight
            weight = max(0.1, 1.0 * (0.9 ** i))  # Decay factor of 0.9
            weights.append(weight)

        # Select tokens using weighted random sampling without replacement
        selected_tokens = []
        temp_tokens = available_tokens.copy()
        temp_weights = weights.copy()

        for _ in range(num_tokens):
            if not temp_tokens:
                break

            # Weighted random selection
            total_weight = sum(temp_weights)
            if total_weight == 0:
                break

            r = random.uniform(0, total_weight)
            cumulative = 0
            selected_idx = 0

            for i, weight in enumerate(temp_weights):
                cumulative += weight
                if r <= cumulative:
                    selected_idx = i
                    break

            # Add selected token and remove from available pool
            selected_tokens.append(temp_tokens[selected_idx])
            temp_tokens.pop(selected_idx)
            temp_weights.pop(selected_idx)

        return Individual(tokens=selected_tokens)

    def create_elite_seeded_individual(self, top_tokens: List[int], strategy: str = "top_singles") -> Individual:
        """
        Create an individual using elite seeding strategies based on top baseline performers.

        Args:
            top_tokens: List of token IDs sorted by impact (best first)
            strategy: Seeding strategy ("top_singles", "top_pairs", "top_combinations")

        Returns:
            Individual with elite token combinations
        """
        if not top_tokens:
            return self.create_random_individual()

        if strategy == "top_singles":
            if self.use_exact_token_count:
                num_tokens = min(self.max_tokens_per_individual, len(top_tokens))
            else:
                num_tokens = min(1, len(top_tokens))
            selected_tokens = top_tokens[:num_tokens]

        elif strategy == "top_pairs":
            if self.use_exact_token_count:
                num_tokens = min(self.max_tokens_per_individual, len(top_tokens))
            else:
                num_tokens = min(2, len(top_tokens))
            selected_tokens = top_tokens[:num_tokens]

        elif strategy == "top_combinations":
            if self.use_exact_token_count:
                num_tokens = min(self.max_tokens_per_individual, len(top_tokens))
            else:
                max_tokens = min(self.max_tokens_per_individual, len(top_tokens), 4)
                num_tokens = random.randint(2, max_tokens)
            selected_tokens = top_tokens[:num_tokens]

        else:
            # Default behavior based on exact count setting
            if self.use_exact_token_count:
                num_tokens = min(self.max_tokens_per_individual, len(top_tokens))
                selected_tokens = top_tokens[:num_tokens] if num_tokens <= len(top_tokens) else random.sample(top_tokens, num_tokens)
            else:
                max_tokens = min(self.max_tokens_per_individual, len(top_tokens))
                num_tokens = random.randint(1, max_tokens)
                selected_tokens = random.sample(top_tokens[:10], num_tokens)

        return Individual(tokens=selected_tokens)

    def create_initial_population(self) -> List[Individual]:
        """Create initial population using multiple baseline-guided seeding strategies."""
        population = []

        # If we have baseline results and seeding is enabled, use them for intelligent seeding
        if self.token_impact_map and self.use_baseline_seeding:
            # Get top performing tokens sorted by impact
            sorted_tokens = sorted(
                self.token_impact_map.items(),
                key=lambda x: x[1]['impact'],
                reverse=True
            )

            # Extract top tokens with positive impact
            top_tokens = [token_id for token_id, data in sorted_tokens if data['impact'] > 0]

            if top_tokens:
                self.logger.info(f"Seeding population with {len(top_tokens)} top-performing tokens using multiple strategies")

                # Calculate seeding distribution based on baseline_seeding_ratio
                baseline_population_size = int(self.population_size * self.baseline_seeding_ratio)

                # Strategy 1: Elite singles (~15% of baseline population) - Best individual tokens
                elite_singles_count = max(1, int(baseline_population_size * 0.15))
                for i in range(min(elite_singles_count, len(top_tokens))):
                    individual = self.create_elite_seeded_individual(top_tokens, "top_singles")
                    population.append(individual)

                # Strategy 2: Elite pairs (~20% of baseline population) - Best token pairs
                elite_pairs_count = max(1, int(baseline_population_size * 0.20))
                for i in range(elite_pairs_count):
                    individual = self.create_elite_seeded_individual(top_tokens, "top_pairs")
                    population.append(individual)

                # Strategy 3: Elite combinations (~25% of baseline population) - Best token combinations
                elite_combinations_count = max(1, int(baseline_population_size * 0.25))
                for i in range(elite_combinations_count):
                    individual = self.create_elite_seeded_individual(top_tokens, "top_combinations")
                    population.append(individual)

                # Strategy 4: Baseline-guided weighted selection (remaining baseline population)
                current_baseline_count = elite_singles_count + elite_pairs_count + elite_combinations_count
                remaining_baseline_count = baseline_population_size - current_baseline_count
                for i in range(max(0, remaining_baseline_count)):
                    individual = self.create_baseline_guided_individual(top_tokens)
                    population.append(individual)

                # Strategy 5: Random individuals for diversity (remaining population)
                current_count = len(population)
                random_count = self.population_size - current_count
                for _ in range(random_count):
                    individual = self.create_random_individual()
                    population.append(individual)

                self.logger.info(f"Population seeded with: {elite_singles_count} elite singles, "
                               f"{elite_pairs_count} elite pairs, {elite_combinations_count} elite combinations, "
                               f"{remaining_baseline_count} baseline-guided, {random_count} random individuals "
                               f"(baseline ratio: {self.baseline_seeding_ratio:.1%})")
            else:
                # No positive impact tokens, fall back to random
                self.logger.warning("No positive impact tokens found, using random initialization")
                for _ in range(self.population_size):
                    individual = self.create_random_individual()
                    population.append(individual)
        else:
            # No baseline data or seeding disabled, use random initialization
            if not self.use_baseline_seeding:
                self.logger.info("Baseline seeding disabled, using random initialization")
            else:
                self.logger.info("No baseline data available, using random initialization")
            for _ in range(self.population_size):
                individual = self.create_random_individual()
                population.append(individual)

        return population

    def tournament_selection(self, population: List[Individual], tournament_size: int = 2) -> Individual:
        """
        Tournament selection for choosing parents.

        Args:
            population: Current population
            tournament_size: Number of individuals in tournament

        Returns:
            Selected individual
        """
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Create offspring through improved crossover with exact token count maintenance.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Two offspring individuals with exactly max_tokens_per_individual tokens
        """
        if random.random() > self.crossover_rate:
            # No crossover, return copies of parents (already have exact count)
            return Individual(tokens=parent1.tokens.copy()), Individual(tokens=parent2.tokens.copy())

        # Single-point crossover for meaningful recombination
        cut1 = random.randint(1, len(parent1.tokens))
        cut2 = random.randint(1, len(parent2.tokens))

        child1_tokens = parent1.tokens[:cut1] + parent2.tokens[cut2:]
        child2_tokens = parent2.tokens[:cut2] + parent1.tokens[cut1:]

        # Remove duplicates within each child
        child1_tokens = list(dict.fromkeys(child1_tokens))
        child2_tokens = list(dict.fromkeys(child2_tokens))

        # Ensure proper token count based on configuration
        if self.use_exact_token_count:
            child1_tokens = self._ensure_exact_token_count(child1_tokens)
            child2_tokens = self._ensure_exact_token_count(child2_tokens)
        else:
            # For variable count, just ensure we don't exceed max
            child1_tokens = child1_tokens[:self.max_tokens_per_individual]
            child2_tokens = child2_tokens[:self.max_tokens_per_individual]

        return Individual(tokens=child1_tokens), Individual(tokens=child2_tokens)

    def _ensure_exact_token_count(self, tokens: List[int]) -> List[int]:
        """
        Ensure a token list has exactly max_tokens_per_individual tokens.

        Args:
            tokens: Input token list

        Returns:
            Token list with exact count
        """
        if len(tokens) == self.max_tokens_per_individual:
            return tokens
        elif len(tokens) > self.max_tokens_per_individual:
            # Trim to exact count
            return tokens[:self.max_tokens_per_individual]
        else:
            # Pad to exact count with random tokens
            result = tokens.copy()
            available_tokens = [t for t in self.glitch_tokens if t not in result]

            while len(result) < self.max_tokens_per_individual and available_tokens:
                new_token = random.choice(available_tokens)
                result.append(new_token)
                available_tokens.remove(new_token)

            # If we still need more tokens and no unique ones available, allow duplicates
            while len(result) < self.max_tokens_per_individual:
                result.append(random.choice(self.glitch_tokens))

            return result

    def create_sequence_variations(self, individual: Individual, num_variations: int = 3) -> List[Individual]:
        """
        Create sequence variations of an individual by permuting token order.

        Args:
            individual: Source individual to create variations from
            num_variations: Number of sequence variations to generate

        Returns:
            List of individuals with different token sequences
        """
        if not individual.tokens or len(individual.tokens) <= 1:
            return [Individual(tokens=individual.tokens.copy())]

        import itertools
        variations = []

        # Generate all possible permutations
        all_perms = list(itertools.permutations(individual.tokens))

        # If we have fewer permutations than requested, use all
        if len(all_perms) <= num_variations:
            for perm in all_perms:
                variations.append(Individual(tokens=list(perm)))
        else:
            # Sample random permutations
            selected_perms = random.sample(all_perms, num_variations)
            for perm in selected_perms:
                variations.append(Individual(tokens=list(perm)))

        return variations

    def create_sequence_aware_individual(self, top_performers: List[Individual]) -> Individual:
        """
        Create a new individual by combining and reordering tokens from top performers.

        Args:
            top_performers: List of high-fitness individuals to sample from

        Returns:
            Individual with reordered token combination
        """
        if not top_performers:
            return self.create_random_individual()

        # Strategy 1: Take tokens from multiple top performers and reorder
        if len(top_performers) >= 2 and random.random() < 0.6:
            # Collect unique tokens from top performers
            all_tokens = []
            for performer in top_performers[:3]:  # Use top 3
                all_tokens.extend(performer.tokens)

            # Remove duplicates while preserving some order preference
            unique_tokens = []
            seen = set()
            for token in all_tokens:
                if token not in seen:
                    unique_tokens.append(token)
                    seen.add(token)

            # Select tokens up to max count
            if len(unique_tokens) >= self.max_tokens_per_individual:
                selected_tokens = unique_tokens[:self.max_tokens_per_individual]
                # Shuffle to create new sequence
                random.shuffle(selected_tokens)
            else:
                selected_tokens = self._ensure_exact_token_count(unique_tokens)
                random.shuffle(selected_tokens)

            return Individual(tokens=selected_tokens)

        # Strategy 2: Take one top performer and create sequence variation
        else:
            source = random.choice(top_performers)
            variations = self.create_sequence_variations(source, num_variations=1)
            return variations[0] if variations else self.create_random_individual()

    def maintain_diversity(self, population: List[Individual]) -> List[Individual]:
        """
        Remove duplicate individuals and maintain population diversity.

        Args:
            population: Current population

        Returns:
            Population with duplicates removed and diversity preserved
        """
        unique_population = []
        seen_token_sets = set()

        # Keep unique individuals based on their token combinations
        for individual in population:
            # Create a signature for this individual (sorted tokens to handle order variations)
            token_signature = tuple(sorted(individual.tokens))

            if token_signature not in seen_token_sets:
                unique_population.append(individual)
                seen_token_sets.add(token_signature)

        # If we lost too many individuals due to duplicates, fill with random ones
        while len(unique_population) < self.population_size:
            new_individual = self.create_random_individual()
            token_signature = tuple(sorted(new_individual.tokens))

            # Ensure the new individual is also unique
            if token_signature not in seen_token_sets:
                unique_population.append(new_individual)
                seen_token_sets.add(token_signature)
            elif len(unique_population) < self.population_size // 2:
                # If we're really struggling to find unique individuals, allow some duplicates
                unique_population.append(new_individual)

        return unique_population[:self.population_size]

    def calculate_population_diversity(self, population: List[Individual]) -> dict:
        """
        Calculate diversity metrics for the population.

        Args:
            population: Current population

        Returns:
            Dictionary with diversity metrics
        """
        if not population:
            return {"unique_individuals": 0, "avg_tokens_per_individual": 0, "unique_tokens": 0, "diversity_ratio": 0.0}

        # Count unique individuals
        seen_token_sets = set()
        for individual in population:
            token_signature = tuple(sorted(individual.tokens))
            seen_token_sets.add(token_signature)

        unique_individuals = len(seen_token_sets)

        # Calculate average tokens per individual
        total_tokens = sum(len(ind.tokens) for ind in population)
        avg_tokens = total_tokens / len(population) if population else 0

        # Count unique tokens across population
        all_tokens = set()
        for individual in population:
            all_tokens.update(individual.tokens)
        unique_tokens = len(all_tokens)

        # Diversity ratio (unique individuals / total population)
        diversity_ratio = unique_individuals / len(population) if population else 0

        return {
            "unique_individuals": unique_individuals,
            "avg_tokens_per_individual": avg_tokens,
            "unique_tokens": unique_tokens,
            "diversity_ratio": diversity_ratio
        }

    def calculate_adaptive_mutation_rate(self, generation: int) -> float:
        """
        Calculate adaptive mutation rate that decreases over time.

        Args:
            generation: Current generation number

        Returns:
            Adaptive mutation rate for current generation
        """
        if not self.adaptive_mutation:
            return self.mutation_rate

        # Linear decay from initial to final mutation rate
        progress = min(generation / self.max_generations, 1.0)
        adaptive_rate = self.initial_mutation_rate * (1 - progress) + self.final_mutation_rate * progress

        return adaptive_rate

    def mutate(self, individual: Individual, mutation_rate: Optional[float] = None):
        """
        Improved mutation with multiple strategies for better exploration.

        Args:
            individual: Individual to mutate
            mutation_rate: Override mutation rate (uses self.current_mutation_rate if None)
        """
        effective_rate = mutation_rate if mutation_rate is not None else self.current_mutation_rate
        if random.random() > effective_rate:
            return

        # Multi-point mutation: potentially multiple changes per individual
        num_mutations = random.choices([1, 2, 3], weights=[0.7, 0.25, 0.05])[0]

        for _ in range(num_mutations):
            if random.random() > effective_rate:  # Each mutation has independent chance
                continue

            # Choose mutation strategy based on exact token count setting
            if self.use_exact_token_count:
                # Maintain exact token count - only replace and swap
                if len(individual.tokens) == 0:
                    # Should not happen with exact token count, but fallback to adding tokens
                    mutation_type = 'add'
                elif len(individual.tokens) < self.max_tokens_per_individual:
                    # Need to add tokens to reach exact count
                    mutation_type = 'add'
                elif len(individual.tokens) > self.max_tokens_per_individual:
                    # Need to remove tokens to reach exact count
                    mutation_type = 'remove'
                else:
                    # At exact count, only replace, swap, or shuffle to maintain count
                    mutation_type = random.choices(['replace', 'swap', 'shuffle'], weights=[0.6, 0.2, 0.2])[0]
            else:
                # Variable token count - allow all mutation types
                if len(individual.tokens) == 0:
                    mutation_type = 'add'
                elif len(individual.tokens) >= self.max_tokens_per_individual:
                    mutation_type = random.choices(['replace', 'remove', 'swap', 'shuffle'], weights=[0.5, 0.2, 0.1, 0.2])[0]
                else:
                    mutation_type = random.choices(['replace', 'add', 'remove', 'swap', 'shuffle'], weights=[0.3, 0.2, 0.2, 0.1, 0.2])[0]

            if mutation_type == 'replace' and individual.tokens:
                # Replace random token with emphasis on diversity
                idx = random.randint(0, len(individual.tokens) - 1)
                old_token = individual.tokens[idx]

                # Try to find a different token (avoid replacing with same token)
                attempts = 0
                while attempts < 5:
                    new_token = random.choice(self.glitch_tokens)
                    if new_token != old_token and new_token not in individual.tokens:
                        individual.tokens[idx] = new_token
                        break
                    attempts += 1
                else:
                    # Fallback: replace anyway
                    individual.tokens[idx] = random.choice(self.glitch_tokens)

            elif mutation_type == 'add' and len(individual.tokens) < self.max_tokens_per_individual:
                # Add random token with duplicate avoidance to reach exact count
                attempts = 0
                while attempts < 10 and len(individual.tokens) < self.max_tokens_per_individual:
                    new_token = random.choice(self.glitch_tokens)
                    if new_token not in individual.tokens:
                        individual.tokens.append(new_token)
                        break
                    attempts += 1

            elif mutation_type == 'remove' and len(individual.tokens) > (1 if not self.use_exact_token_count else self.max_tokens_per_individual):
                # Remove random token (respect minimum of 1 for variable count)
                idx = random.randint(0, len(individual.tokens) - 1)
                individual.tokens.pop(idx)

            elif mutation_type == 'swap' and len(individual.tokens) >= 2:
                # Swap positions of two tokens (order mutation)
                idx1, idx2 = random.sample(range(len(individual.tokens)), 2)
                individual.tokens[idx1], individual.tokens[idx2] = individual.tokens[idx2], individual.tokens[idx1]

            elif mutation_type == 'shuffle' and len(individual.tokens) >= 2:
                # Shuffle all tokens to explore different sequences
                random.shuffle(individual.tokens)

    def evolve_generation(self, population: List[Individual], generation: int = 0) -> List[Individual]:
        """
        Evolve one generation of the population with improved diversity preservation.

        Args:
            population: Current population
            generation: Current generation number for adaptive parameters

        Returns:
            New population with better diversity
        """
        # Update adaptive mutation rate
        if self.adaptive_mutation:
            self.current_mutation_rate = self.calculate_adaptive_mutation_rate(generation)
        # Ensure population diversity before evolution
        population = self.maintain_diversity(population)

        # Evaluate fitness for all individuals
        for individual in population:
            if individual.fitness == 0.0:  # Not evaluated yet
                self.evaluate_fitness(individual)

        # Sort by fitness (descending)
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Create new population
        new_population = []

        # Elitism - keep best individuals (reduced elite size for more diversity)
        new_population.extend(population[:self.elite_size])

        # Generate offspring
        while len(new_population) < self.population_size:
            # Use tournament selection with some diversity pressure
            parent1 = self.tournament_selection(population, tournament_size=2)  # Reduced tournament size
            parent2 = self.tournament_selection(population, tournament_size=2)

            # Ensure parents are different to promote diversity
            max_attempts = 5
            attempts = 0
            while parent1.tokens == parent2.tokens and attempts < max_attempts:
                parent2 = self.tournament_selection(population, tournament_size=2)
                attempts += 1

            child1, child2 = self.crossover(parent1, parent2)

            self.mutate(child1)
            self.mutate(child2)

            new_population.extend([child1, child2])

        # Trim to exact population size and ensure final diversity
        new_population = new_population[:self.population_size]
        new_population = self.maintain_diversity(new_population)

        return new_population

    def run_evolution(self) -> List[Individual]:
        """
        Run the genetic algorithm evolution.

        Returns:
            Final population sorted by fitness
        """
        self.logger.info("Starting genetic algorithm evolution")

        # Setup baseline
        self.target_token_id, self.baseline_probability = self.get_baseline_probability()

        # Baseline individual token impacts
        self.baseline_token_impacts()

        # Notify GUI callback of evolution start
        if self.gui_callback:
            target_text = self.tokenizer.decode([self.target_token_id]) if self.target_token_id else None
            self.gui_callback.on_evolution_start(
                baseline_prob=self.baseline_probability,
                target_token_id=self.target_token_id,
                target_token_text=target_text,
                initial_top_tokens=self.initial_top_tokens,
                tokenizer=self.tokenizer
            )

        # Create initial population
        population = self.create_initial_population()

        # Log initial diversity
        initial_diversity = self.calculate_population_diversity(population)
        self.logger.info(f"Initial population diversity: {initial_diversity['unique_individuals']}/{len(population)} unique individuals "
                        f"(diversity ratio: {initial_diversity['diversity_ratio']:.3f})")

        best_fitness_history = []
        avg_fitness_history = []
        diversity_history = []
        stagnation_counter = 0
        last_best_fitness = 0.0
        early_stopped = False

        # Evolution loop
        for generation in tqdm(range(self.max_generations), desc="Evolving"):
            population = self.evolve_generation(population, generation)

            # Track statistics
            fitnesses = [ind.fitness for ind in population if ind.fitness > -1.0]
            if fitnesses:
                best_fitness = max(fitnesses)
                avg_fitness = sum(fitnesses) / len(fitnesses)
            else:
                best_fitness = avg_fitness = 0.0

            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            # Track diversity metrics
            diversity_metrics = self.calculate_population_diversity(population)
            diversity_history.append(diversity_metrics)

            # Check for stagnation
            if abs(best_fitness - last_best_fitness) < 1e-6:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                last_best_fitness = best_fitness

            # Log every 10 generations with diversity info
            if generation % 10 == 0:
                mutation_info = f"Mutation rate = {self.current_mutation_rate:.3f}" if self.adaptive_mutation else f"Mutation rate = {self.mutation_rate:.3f}"
                self.logger.info(f"Generation {generation}: Best fitness = {best_fitness:.4f}, "
                               f"Avg fitness = {avg_fitness:.4f}, "
                               f"Diversity = {diversity_metrics['unique_individuals']}/{len(population)} "
                               f"({diversity_metrics['diversity_ratio']:.3f}), "
                               f"Stagnation = {stagnation_counter}, {mutation_info}")

                if stagnation_counter >= 20:
                    self.logger.info(f"⚠️  Population stagnated for {stagnation_counter} generations - aggressive diversity injection!")

                    # More aggressive diversity injection strategy
                    if stagnation_counter >= 50:
                        # Very aggressive: replace 60% of population, including some elites
                        injection_rate = 0.6
                        replace_elites = True
                        # Temporarily boost mutation rate for next few generations
                        self.current_mutation_rate = min(0.8, self.current_mutation_rate * 2.0)
                        self.logger.info(f"🔥 CRITICAL stagnation ({stagnation_counter}gen) - replacing {injection_rate*100:.0f}% of population + boosting mutation to {self.current_mutation_rate:.3f}")
                    elif stagnation_counter >= 30:
                        # Moderate aggressive: replace 40% of population
                        injection_rate = 0.4
                        replace_elites = False
                        self.current_mutation_rate = min(0.6, self.current_mutation_rate * 1.5)
                        self.logger.info(f"⚡ HIGH stagnation ({stagnation_counter}gen) - replacing {injection_rate*100:.0f}% of population + boosting mutation to {self.current_mutation_rate:.3f}")
                    else:
                        # Initial injection: replace 25% of population
                        injection_rate = 0.25
                        replace_elites = False
                        self.logger.info(f"💧 MILD stagnation ({stagnation_counter}gen) - replacing {injection_rate*100:.0f}% of population")

                    num_to_replace = max(2, int(len(population) * injection_rate))

                    # Create diverse new individuals with sequence-aware strategies
                    new_individuals = []

                    # Extract top performers for sequence-aware strategies
                    top_performers = sorted([ind for ind in population if ind.fitness > 0],
                                          key=lambda x: x.fitness, reverse=True)[:5]
                    proven_tokens = set()
                    for performer in top_performers:
                        proven_tokens.update(performer.tokens)

                    # Get existing token combinations (both sorted and exact sequences)
                    existing_combinations = set()
                    existing_sequences = set()
                    for ind in population:
                        existing_combinations.add(tuple(sorted(ind.tokens)))
                        existing_sequences.add(tuple(ind.tokens))

                    for i in range(num_to_replace):
                        max_attempts = 20  # Prevent infinite loops
                        attempt = 0
                        new_ind = None

                        while attempt < max_attempts:
                            # Choose strategy based on sequence diversity configuration
                            if self.use_sequence_aware_diversity and random.random() < self.sequence_diversity_ratio:
                                # Use sequence-aware strategies
                                strategy = i % 4  # 4 sequence-aware strategies

                                if strategy == 0:
                                    # Strategy 1: Sequence variations of top performers
                                    if top_performers:
                                        source = random.choice(top_performers[:3])
                                        variations = self.create_sequence_variations(source, num_variations=3)
                                        candidate = random.choice(variations)
                                    else:
                                        candidate = self.create_random_individual()
                                elif strategy == 1:
                                    # Strategy 2: Sequence-aware combination from multiple top performers
                                    if len(top_performers) >= 2:
                                        candidate = self.create_sequence_aware_individual(top_performers)
                                    else:
                                        candidate = self.create_random_individual()
                                elif strategy == 2:
                                    # Strategy 3: Reverse sequence of best performer
                                    if top_performers:
                                        best = top_performers[0]
                                        reversed_tokens = best.tokens.copy()
                                        reversed_tokens.reverse()
                                        candidate = Individual(tokens=reversed_tokens)
                                    else:
                                        candidate = self.create_random_individual()
                                else:
                                    # Strategy 4: Heavy mutation with sequence focus
                                    if population:
                                        best_current = max(population, key=lambda x: x.fitness)
                                        new_tokens = best_current.tokens.copy()

                                        # Apply mutations while maintaining token count
                                        num_mutations = random.randint(1, min(3, len(new_tokens)))
                                        for _ in range(num_mutations):
                                            if random.random() < 0.8:
                                                # Replace random token
                                                idx = random.randint(0, len(new_tokens) - 1)
                                                new_token = random.choice(self.glitch_tokens)
                                                if new_token not in new_tokens:
                                                    new_tokens[idx] = new_token

                                        # Always shuffle for sequence diversity
                                        random.shuffle(new_tokens)
                                        candidate = Individual(tokens=new_tokens)
                                    else:
                                        candidate = self.create_random_individual()
                            else:
                                # Use traditional diversity strategies
                                candidate = self.create_random_individual()

                            # Check if this sequence is unique (prioritize sequence diversity over just token set diversity)
                            candidate_sequence = tuple(candidate.tokens)
                            candidate_combination = tuple(sorted(candidate.tokens))

                            # Accept if sequence is unique, or if token combination is unique
                            if (candidate_sequence not in existing_sequences or
                                candidate_combination not in existing_combinations):
                                new_ind = candidate
                                existing_combinations.add(candidate_combination)
                                existing_sequences.add(candidate_sequence)
                                break

                            attempt += 1

                        # If we couldn't find a unique individual, use the last candidate
                        if new_ind is None:
                            new_ind = candidate

                        new_individuals.append(new_ind)

                    # Replace individuals in population
                    if replace_elites:
                        # Replace random individuals including some elites
                        indices_to_replace = random.sample(range(len(population)), num_to_replace)
                    else:
                        # Replace worst performers (preserve elites)
                        indices_to_replace = list(range(len(population) - num_to_replace, len(population)))

                    for i, new_ind in enumerate(new_individuals):
                        if i < len(indices_to_replace):
                            population[indices_to_replace[i]] = new_ind

                    # Force re-evaluation of all new individuals
                    for individual in new_individuals:
                        self.evaluate_fitness(individual)

                    # Reset stagnation counter after injection
                    stagnation_counter = 0
                    last_best_fitness = best_fitness

                    self.logger.info(f"✅ Diversity injection complete - {num_to_replace} individuals replaced, stagnation counter reset")

                # Find best individual for detailed logging
                if fitnesses:
                    best_individual = max(population, key=lambda x: x.fitness)
                    best_tokens = [self.tokenizer.decode([token_id]).strip() for token_id in best_individual.tokens]
                    self.logger.info(f"Best tokens = {best_individual.tokens}")

            # Check for early stopping - target probability reduction achieved
            if fitnesses and best_fitness >= (self.baseline_probability * self.early_stopping_threshold):
                reduction_pct = (best_fitness / self.baseline_probability) * 100
                threshold_pct = self.early_stopping_threshold * 100
                self.logger.info(f"🎯 Target probability reduction ({threshold_pct:.1f}%) achieved at generation {generation}!")
                self.logger.info(f"Final reduction: {reduction_pct:.2f}%")
                early_stopped = True

                # Still notify GUI callback before breaking
                if self.gui_callback:
                    best_individual = max(population, key=lambda x: x.fitness)
                    current_prob = getattr(best_individual, 'modified_prob', self.baseline_probability)
                    new_top_tokens = getattr(best_individual, 'new_top_tokens', None)

                    self.gui_callback.on_generation_complete(
                        generation=generation,
                        best_individual=best_individual,
                        avg_fitness=avg_fitness,
                        current_probability=current_prob,
                        tokenizer=self.tokenizer,
                        new_top_tokens=new_top_tokens
                    )

                break

            # Notify GUI callback of generation progress
            if self.gui_callback and fitnesses:
                best_individual = max(population, key=lambda x: x.fitness)
                current_prob = getattr(best_individual, 'modified_prob', self.baseline_probability)

                # Get new top tokens if available
                new_top_tokens = getattr(best_individual, 'new_top_tokens', None)

                self.gui_callback.on_generation_complete(
                    generation=generation,
                    best_individual=best_individual,
                    avg_fitness=avg_fitness,
                    current_probability=current_prob,
                    tokenizer=self.tokenizer,
                    new_top_tokens=new_top_tokens
                )

            # Log progress
            if generation % 10 == 0:
                best_individual = max(population, key=lambda x: x.fitness)
                self.logger.info(
                    f"Generation {generation}: Best fitness = {best_fitness:.4f}, "
                    f"Avg fitness = {avg_fitness:.4f}, "
                    f"Best tokens = {best_individual.tokens}"
                )

        # Final evaluation and sorting
        for individual in population:
            if individual.fitness == 0.0:
                self.evaluate_fitness(individual)

        population.sort(key=lambda x: x.fitness, reverse=True)

        # Notify GUI callback of evolution completion
        if self.gui_callback:
            self.gui_callback.on_evolution_complete(population, self.max_generations)

        if early_stopped:
            self.logger.info("Evolution completed (early stopping triggered)")
        else:
            self.logger.info("Evolution completed")
        return population

    def display_results(self, population: List[Individual], top_n: int = 10):
        """
        Display the best results from evolution.

        Args:
            population: Final population
            top_n: Number of top results to display
        """
        print(f"\n=== Top {top_n} Probability Reducers ===")
        print(f"Base text: '{self.base_text}'")
        print(f"Target token: '{self.tokenizer.decode([self.target_token_id])}' (ID: {self.target_token_id})")
        print(f"Baseline probability: {self.baseline_probability:.4f}")
        print()

        for i, individual in enumerate(population[:top_n]):
            if individual.fitness <= 0:
                continue

            token_texts = [self.tokenizer.decode([tid]) for tid in individual.tokens]
            token_repr = [repr(text) for text in token_texts]

            reduction_pct = (individual.fitness / self.baseline_probability) * 100

            print(f"{i+1:2d}. Tokens: {individual.tokens} → {token_repr}")
            print(f"    Fitness: {individual.fitness:.4f} ({reduction_pct:.1f}% reduction)")
            print(f"    Probability: {self.baseline_probability:.4f} → {individual.modified_prob:.4f}")
            print()

    def save_results(self, population: List[Individual], output_file: str):
        """
        Save results to JSON file.

        Args:
            population: Final population
            output_file: Output file path
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be loaded first")

        results = {
            'model_name': self.model_name,
            'base_text': self.base_text,
            'target_token_id': self.target_token_id,
            'target_token_text': self.tokenizer.decode([self.target_token_id]) if self.target_token_id else None,
            'baseline_probability': self.baseline_probability,
            'ga_parameters': {
                'population_size': self.population_size,
                'max_generations': self.max_generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'max_tokens_per_individual': self.max_tokens_per_individual
            },
            'results': []
        }

        for individual in population:
            token_texts = [self.tokenizer.decode([token_id]) for token_id in individual.tokens]
            results['results'].append({
                'tokens': individual.tokens,
                'token_texts': token_texts,
                'fitness': individual.fitness,
                'baseline_prob': individual.baseline_prob,
                'modified_prob': individual.modified_prob,
                'reduction_percentage': ((individual.baseline_prob - individual.modified_prob) / individual.baseline_prob * 100) if individual.baseline_prob > 0 else 0
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results saved to: {output_file}")

    def get_token_impact_map(self) -> Dict[int, Dict[str, Any]]:
        """
        Get the token impact baseline map.

        Returns:
            Dictionary mapping token_id -> impact metrics
        """
        return self.token_impact_map

    def save_token_impact_results(self, output_file: str = "token_impact_baseline.json"):
        """
        Save token impact baseline results to JSON file.

        Args:
            output_file: Output file path
        """
        if not self.token_impact_map:
            self.logger.warning("No token impact data to save. Run baseline_token_impacts() first.")
            return

        # Prepare results for JSON serialization
        results = {
            'metadata': {
                'model_name': self.model_name,
                'base_text': self.base_text,
                'target_token_id': self.target_token_id,
                'target_token_text': self.tokenizer.decode([self.target_token_id]) if self.target_token_id else None,
                'baseline_probability': self.baseline_probability,
                'total_tokens_tested': len(self.token_impact_map),
                'positive_impact_tokens': len([d for d in self.token_impact_map.values() if d['impact'] > 0])
            },
            'token_impacts': []
        }

        # Sort by impact and add to results
        sorted_impacts = sorted(self.token_impact_map.items(), key=lambda x: x[1]['impact'], reverse=True)
        for token_id, data in sorted_impacts:
            results['token_impacts'].append({
                'token_id': token_id,
                'token_text': data['token_text'],
                'baseline_prob': data['baseline_prob'],
                'modified_prob': data['modified_prob'],
                'impact': data['impact'],
                'reduction_ratio': data['reduction_ratio'],
                'rank_by_impact': data['rank_by_impact']
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Token impact results saved to: {output_file}")

    def display_token_impact_results(self, top_n: int = 10, bottom_n: int = 5):
        """
        Display token impact baseline results.

        Args:
            top_n: Number of top performers to show
            bottom_n: Number of bottom performers to show
        """
        if not self.token_impact_map:
            self.logger.warning("No token impact data to display. Run baseline_token_impacts() first.")
            return

        sorted_impacts = sorted(self.token_impact_map.items(), key=lambda x: x[1]['impact'], reverse=True)

        print(f"\n=== Token Impact Baseline Results ===")
        print(f"Base text: '{self.base_text}'")
        print(f"Target token: '{self.tokenizer.decode([self.target_token_id])}' (ID: {self.target_token_id})")
        print(f"Baseline probability: {self.baseline_probability:.4f}")
        print(f"Total tokens tested: {len(self.token_impact_map)}")

        positive_impacts = [d['impact'] for d in self.token_impact_map.values() if d['impact'] > 0]
        print(f"Tokens with positive impact: {len(positive_impacts)}/{len(self.token_impact_map)}")

        if positive_impacts:
            avg_positive = sum(positive_impacts) / len(positive_impacts)
            max_impact = max(positive_impacts)
            print(f"Average positive impact: {avg_positive:.4f}")
            print(f"Maximum impact: {max_impact:.4f}")

        print(f"\n🏆 Top {top_n} Most Effective Tokens:")
        for i, (token_id, data) in enumerate(sorted_impacts[:top_n]):
            print(f"  {i+1:2d}. Token {token_id:6d} '{data['token_text'][:30]:<30}' "
                  f"Impact: {data['impact']:7.4f} ({data['reduction_ratio']:6.1%}) "
                  f"Prob: {data['baseline_prob']:.4f} → {data['modified_prob']:.4f}")

        if bottom_n > 0 and len(sorted_impacts) > top_n:
            print(f"\n⬇️  Bottom {bottom_n} Least Effective Tokens:")
            for i, (token_id, data) in enumerate(sorted_impacts[-bottom_n:], len(sorted_impacts) - bottom_n + 1):
                print(f"  {i:2d}. Token {token_id:6d} '{data['token_text'][:30]:<30}' "
                      f"Impact: {data['impact']:7.4f} ({data['reduction_ratio']:6.1%}) "
                      f"Prob: {data['baseline_prob']:.4f} → {data['modified_prob']:.4f}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Genetic Algorithm for Breeding Probability Reducer Token Combinations"
    )
    parser.add_argument(
        "model_name",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--base-text",
        default="The quick brown",
        help="Base text to test probability reduction on"
    )
    parser.add_argument(
        "--target-token",
        help="Specific token to target (auto-detected if not provided)"
    )
    parser.add_argument(
        "--token-file",
        default="email_llams321.json",
        help="JSON file containing glitch tokens"
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=50,
        help="Population size for genetic algorithm"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="Maximum number of generations"
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.1,
        help="Mutation rate (0.0-1.0)"
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.7,
        help="Crossover rate (0.0-1.0)"
    )
    parser.add_argument(
        "--elite-size",
        type=int,
        default=5,
        help="Number of elite individuals to preserve each generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3,
        help="Maximum tokens per individual combination (1-10 recommended, default: 3). Higher values explore more complex combinations but may have diminishing returns."
    )
    parser.add_argument(
        "--output",
        help="Output file for results (JSON format)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top results to display"
    )
    parser.add_argument(
        "--ascii-only",
        action="store_true",
        help="Filter tokens to only include those with ASCII-only decoded text"
    )
    parser.add_argument(
        "--adaptive-mutation",
        action="store_true",
        help="Use adaptive mutation rate that starts high (0.8) and decreases to low (0.1) over generations"
    )
    parser.add_argument(
        "--initial-mutation-rate",
        type=float,
        default=0.8,
        help="Initial mutation rate for adaptive mutation (default: 0.8)"
    )
    parser.add_argument(
        "--final-mutation-rate",
        type=float,
        default=0.1,
        help="Final mutation rate for adaptive mutation (default: 0.1)"
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only run token impact baseline analysis without genetic algorithm evolution"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip token impact baseline analysis and go straight to genetic algorithm"
    )
    parser.add_argument(
        "--baseline-output",
        default="token_impact_baseline.json",
        help="Output file for token impact baseline results (default: token_impact_baseline.json)"
    )
    parser.add_argument(
        "--baseline-top-n",
        type=int,
        default=10,
        help="Number of top tokens to display in baseline results (default: 10)"
    )
    parser.add_argument(
        "--baseline-seeding",
        action="store_true",
        default=True,
        help="Use baseline results to intelligently seed initial population (default: enabled)"
    )
    parser.add_argument(
        "--no-baseline-seeding",
        action="store_true",
        help="Disable baseline-guided population seeding, use random initialization only"
    )
    parser.add_argument(
        "--baseline-seeding-ratio",
        type=float,
        default=0.7,
        help="Fraction of population to seed with baseline guidance (0.0-1.0, default: 0.7)"
    )
    parser.add_argument(
        "--exact-token-count",
        action="store_true",
        default=True,
        help="Use exact max_tokens count for all individuals (default: enabled)"
    )
    parser.add_argument(
        "--variable-token-count",
        action="store_true",
        help="Allow variable token count (1 to max_tokens) for individuals"
    )
    parser.add_argument(
        "--sequence-aware-diversity",
        action="store_true",
        default=True,
        help="Enable sequence-aware diversity injection (default: enabled)"
    )
    parser.add_argument(
        "--no-sequence-diversity",
        action="store_true",
        help="Disable sequence-aware diversity injection, use traditional diversity only"
    )
    parser.add_argument(
        "--sequence-diversity-ratio",
        type=float,
        default=0.6,
        help="Fraction of diversity injection to use sequence-aware strategies (0.0-1.0, default: 0.6)"
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = GeneticProbabilityReducer(
        model_name=args.model_name,
        base_text=args.base_text,
        target_token=args.target_token
    )

    # Set GA parameters
    analyzer.population_size = args.population_size
    analyzer.max_generations = args.generations
    analyzer.mutation_rate = args.mutation_rate
    analyzer.crossover_rate = args.crossover_rate
    analyzer.elite_size = args.elite_size
    analyzer.max_tokens_per_individual = args.max_tokens

    # Set adaptive mutation parameters
    analyzer.adaptive_mutation = args.adaptive_mutation
    analyzer.initial_mutation_rate = args.initial_mutation_rate
    analyzer.final_mutation_rate = args.final_mutation_rate
    if args.adaptive_mutation:
        analyzer.current_mutation_rate = args.initial_mutation_rate

    # Set baseline seeding parameters
    if args.no_baseline_seeding:
        analyzer.use_baseline_seeding = False
    else:
        analyzer.use_baseline_seeding = args.baseline_seeding
    analyzer.baseline_seeding_ratio = max(0.0, min(1.0, args.baseline_seeding_ratio))

    # Set token count behavior
    if args.variable_token_count:
        analyzer.use_exact_token_count = False
    else:
        analyzer.use_exact_token_count = args.exact_token_count

    # Set sequence-aware diversity behavior
    if args.no_sequence_diversity:
        analyzer.use_sequence_aware_diversity = False
    else:
        analyzer.use_sequence_aware_diversity = args.sequence_aware_diversity
    analyzer.sequence_diversity_ratio = max(0.0, min(1.0, args.sequence_diversity_ratio))

    try:
        # Load model and tokenizer
        analyzer.load_model()
        analyzer.load_glitch_tokens(args.token_file, ascii_only=args.ascii_only)

        if args.baseline_only:
            # Only run token impact baseline analysis
            analyzer.target_token_id, analyzer.baseline_probability = analyzer.get_baseline_probability()
            analyzer.baseline_token_impacts()
            analyzer.display_token_impact_results(top_n=args.baseline_top_n)
            analyzer.save_token_impact_results(args.baseline_output)
        else:
            # Run evolution (which includes baseline unless skipped)
            if args.skip_baseline:
                # Temporarily disable baseline in run_evolution
                original_run_evolution = analyzer.run_evolution
                def run_evolution_no_baseline():
                    analyzer.target_token_id, analyzer.baseline_probability = analyzer.get_baseline_probability()
                    # Skip baseline_token_impacts() call
                    # ... rest of original run_evolution logic
                    return original_run_evolution()

                # For now, just run normal evolution - the baseline call is fast enough
                final_population = analyzer.run_evolution()
            else:
                final_population = analyzer.run_evolution()

            # Display results
            analyzer.display_results(final_population, top_n=args.top_n)

            # Save results if requested
            if args.output:
                analyzer.save_results(final_population, args.output)

            # Save token impact results if baseline was run
            if not args.skip_baseline:
                analyzer.save_token_impact_results(args.baseline_output)

    except Exception as e:
        analyzer.logger.error(f"Error during execution: {e}")
        raise


# CLI integration moved to main cli.py
# if __name__ == "__main__":
#     main()
