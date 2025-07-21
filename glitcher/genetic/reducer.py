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

    def __str__(self):
        return f"Individual(tokens={self.tokens}, fitness={self.fitness:.4f})"


class GeneticProbabilityReducer:
    """
    Genetic algorithm for evolving token combinations that reduce prediction probabilities.

    遗传算法用于进化减少预测概率的标记组合。
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
        self.elite_size = 5
        self.max_tokens_per_individual = 3  # Configurable via CLI --max-tokens

        # Target information
        self.target_token_id: Optional[int] = None
        self.baseline_probability: float = 0.0
        self.initial_top_tokens: List[Tuple[int, float]] = []  # Store initial top 10 tokens with probabilities

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

    def load_glitch_tokens(self, token_file: str):
        """
        Load glitch tokens from JSON file.

        Args:
            token_file: Path to JSON file containing glitch tokens
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

        except Exception as e:
            self.logger.error(f"Error loading glitch tokens: {e}")
            raise

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
        modified_text = "".join(token_texts) + self.base_text

        try:
            inputs = self.tokenizer(modified_text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)

            modified_prob = probs[self.target_token_id].item()

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
        """Create a random individual with 1-N tokens (N = max_tokens_per_individual)."""
        num_tokens = random.randint(1, self.max_tokens_per_individual)
        tokens = random.sample(self.glitch_tokens, num_tokens)
        return Individual(tokens=tokens)

    def create_initial_population(self) -> List[Individual]:
        """Create initial population of random individuals."""
        population = []
        for _ in range(self.population_size):
            individual = self.create_random_individual()
            population.append(individual)
        return population

    def tournament_selection(self, population: List[Individual], tournament_size: int = 3) -> Individual:
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
        Create offspring through crossover.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Two offspring individuals
        """
        if random.random() > self.crossover_rate:
            # No crossover, return copies of parents
            return Individual(tokens=parent1.tokens.copy()), Individual(tokens=parent2.tokens.copy())

        # Combine tokens from both parents
        all_tokens = parent1.tokens + parent2.tokens

        # Create two offspring with different token combinations
        max_len = min(len(all_tokens), self.max_tokens_per_individual)

        if len(all_tokens) <= self.max_tokens_per_individual:
            # Use all tokens if not too many
            child1_tokens = all_tokens
            child2_tokens = all_tokens
        else:
            # Random selection from combined tokens
            child1_tokens = random.sample(all_tokens, max_len)
            child2_tokens = random.sample(all_tokens, max_len)

        return Individual(tokens=child1_tokens), Individual(tokens=child2_tokens)

    def mutate(self, individual: Individual):
        """
        Mutate an individual by changing tokens.

        Args:
            individual: Individual to mutate
        """
        if random.random() > self.mutation_rate:
            return

        mutation_type = random.choice(['replace', 'add', 'remove'])

        if mutation_type == 'replace' and individual.tokens:
            # Replace random token
            idx = random.randint(0, len(individual.tokens) - 1)
            individual.tokens[idx] = random.choice(self.glitch_tokens)

        elif mutation_type == 'add' and len(individual.tokens) < self.max_tokens_per_individual:
            # Add random token
            new_token = random.choice(self.glitch_tokens)
            if new_token not in individual.tokens:  # Avoid duplicates
                individual.tokens.append(new_token)

        elif mutation_type == 'remove' and len(individual.tokens) > 1:
            # Remove random token
            idx = random.randint(0, len(individual.tokens) - 1)
            individual.tokens.pop(idx)

    def evolve_generation(self, population: List[Individual]) -> List[Individual]:
        """
        Evolve one generation of the population.

        Args:
            population: Current population

        Returns:
            New population
        """
        # Evaluate fitness for all individuals
        for individual in population:
            if individual.fitness == 0.0:  # Not evaluated yet
                self.evaluate_fitness(individual)

        # Sort by fitness (descending)
        population.sort(key=lambda x: x.fitness, reverse=True)

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

        # Trim to exact population size
        return new_population[:self.population_size]

    def run_evolution(self) -> List[Individual]:
        """
        Run the genetic algorithm evolution.

        Returns:
            Final population sorted by fitness
        """
        self.logger.info("Starting genetic algorithm evolution")

        # Setup baseline
        self.target_token_id, self.baseline_probability = self.get_baseline_probability()

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

        best_fitness_history = []
        avg_fitness_history = []

        # Evolution loop
        for generation in tqdm(range(self.max_generations), desc="Evolving"):
            population = self.evolve_generation(population)

            # Track statistics
            fitnesses = [ind.fitness for ind in population if ind.fitness > -1.0]
            if fitnesses:
                best_fitness = max(fitnesses)
                avg_fitness = sum(fitnesses) / len(fitnesses)
            else:
                best_fitness = avg_fitness = 0.0

            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            # Notify GUI callback of generation progress
            if self.gui_callback and fitnesses:
                best_individual = max(population, key=lambda x: x.fitness)
                current_prob = self.baseline_probability * (1 - best_individual.fitness) if best_individual.fitness > 0 else self.baseline_probability
                self.gui_callback.on_generation_complete(
                    generation=generation,
                    best_individual=best_individual,
                    avg_fitness=avg_fitness,
                    current_probability=current_prob,
                    tokenizer=self.tokenizer
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

    try:
        # Load model and tokens
        analyzer.load_model()
        analyzer.load_glitch_tokens(args.token_file)

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


# CLI integration moved to main cli.py
# if __name__ == "__main__":
#     main()
