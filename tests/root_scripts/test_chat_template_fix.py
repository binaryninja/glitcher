#!/usr/bin/env python3
"""
Test script to verify chat template implementation in genetic algorithm.

This script tests the new chat template functionality to ensure:
1. Instruct models use proper conversation formatting
2. Base models use direct text completion
3. Probabilities are measured at the correct positions
4. The system prompt guides the model appropriately
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
from pathlib import Path
import sys

# Add the glitcher module to path
sys.path.append(str(Path(__file__).parent))

from glitcher.genetic.reducer import GeneticProbabilityReducer


def test_chat_template_formatting(model_path: str):
    """Test chat template formatting for different model types."""

    print(f"{'='*80}")
    print(f"TESTING CHAT TEMPLATE FORMATTING")
    print(f"{'='*80}")
    print(f"Model: {model_path}")

    # Create genetic reducer instance
    ga = GeneticProbabilityReducer(
        model_name=model_path,
        base_text="The quick brown",
        target_token="fox"
    )

    # Load model
    print("\nLoading model...")
    ga.load_model()

    # Test base text formatting
    print(f"\n{'-'*60}")
    print("BASE TEXT FORMATTING")
    print(f"{'-'*60}")

    base_text = "The quick brown"
    formatted_base = ga._format_input_for_model(base_text)

    print(f"Original text: '{base_text}'")
    print(f"Formatted text:")
    print(f"'{formatted_base}'")
    print(f"Is instruct model: {ga.is_instruct_model}")

    # Test with glitch tokens
    print(f"\n{'-'*60}")
    print("GLITCH TOKEN FORMATTING")
    print(f"{'-'*60}")

    # Sample glitch tokens
    glitch_tokens = [115614, 118508, 121267]
    token_texts = [ga.tokenizer.decode([tid]) for tid in glitch_tokens]
    joined_tokens = "".join(token_texts)

    print(f"Glitch tokens: {glitch_tokens}")
    print(f"Decoded tokens: {token_texts}")
    print(f"Joined tokens: '{joined_tokens}'")

    # Format according to model type
    if ga.is_instruct_model:
        modified_text = f"{joined_tokens} {base_text}".strip()
    else:
        modified_text = f"({joined_tokens}): {base_text}"

    formatted_modified = ga._format_input_for_model(modified_text)

    print(f"Modified text: '{modified_text}'")
    print(f"Formatted modified text:")
    print(f"'{formatted_modified}'")

    return ga, formatted_base, formatted_modified


def test_probability_measurement(ga, formatted_base: str, formatted_modified: str):
    """Test probability measurement at correct positions."""

    print(f"\n{'='*80}")
    print(f"TESTING PROBABILITY MEASUREMENT")
    print(f"{'='*80}")

    # Test baseline probability
    print(f"\n{'-'*60}")
    print("BASELINE PROBABILITY")
    print(f"{'-'*60}")

    target_id, target_prob, wanted_id, wanted_prob = ga.get_baseline_probability()

    print(f"Target token ID: {target_id}")
    print(f"Target token: '{ga.tokenizer.decode([target_id]) if target_id else 'None'}'")
    print(f"Target probability: {target_prob:.8f}")

    # Test modified probability
    print(f"\n{'-'*60}")
    print("MODIFIED PROBABILITY (with glitch tokens)")
    print(f"{'-'*60}")

    # Tokenize modified input
    inputs = ga.tokenizer(formatted_modified, return_tensors="pt").to(ga.device)

    with torch.no_grad():
        outputs = ga.model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last position
        probs = torch.softmax(logits, dim=-1)

    if target_id is not None:
        modified_prob = probs[target_id].item()
        print(f"Modified probability: {modified_prob:.8f}")

        # Calculate impact
        prob_change = modified_prob - target_prob
        prob_ratio = modified_prob / target_prob if target_prob > 0 else float('inf')
        reduction = (target_prob - modified_prob) / target_prob if target_prob > 0 else 0

        print(f"Probability change: {prob_change:+.8f}")
        print(f"Probability ratio: {prob_ratio:.6f}")
        print(f"Probability reduction: {reduction:.4f} ({reduction*100:.2f}%)")

    # Show top 5 predictions for both cases
    print(f"\n{'-'*40}")
    print("TOP 5 BASELINE PREDICTIONS")
    print(f"{'-'*40}")

    baseline_inputs = ga.tokenizer(formatted_base, return_tensors="pt").to(ga.device)
    with torch.no_grad():
        baseline_outputs = ga.model(**baseline_inputs)
        baseline_logits = baseline_outputs.logits[0, -1, :]
        baseline_probs = torch.softmax(baseline_logits, dim=-1)

    top_probs, top_indices = torch.topk(baseline_probs, 5)
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
        token_str = ga.tokenizer.decode([idx.item()])
        print(f"  {i}. '{token_str}' (ID: {idx.item()}) - {prob.item():.6f}")

    print(f"\n{'-'*40}")
    print("TOP 5 MODIFIED PREDICTIONS")
    print(f"{'-'*40}")

    top_probs, top_indices = torch.topk(probs, 5)
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
        token_str = ga.tokenizer.decode([idx.item()])
        print(f"  {i}. '{token_str}' (ID: {idx.item()}) - {prob.item():.6f}")


def test_genetic_evolution_sample(ga):
    """Test a few generations of genetic evolution to verify it works."""

    print(f"\n{'='*80}")
    print(f"TESTING GENETIC EVOLUTION SAMPLE")
    print(f"{'='*80}")

    # Load some sample glitch tokens
    try:
        ga.load_glitch_tokens(token_file="glitch_tokens.json", ascii_only=True)
        print(f"Loaded {len(ga.available_tokens)} glitch tokens")
    except Exception as e:
        print(f"Warning: Could not load glitch tokens: {e}")
        # Use some sample tokens
        ga.available_tokens = [115614, 118508, 121267, 112063, 126357, 89472]
        print(f"Using sample tokens: {ga.available_tokens}")

    # Set up basic parameters
    ga.population_size = 10
    ga.max_generations = 3
    ga.max_tokens_per_individual = 2

    # Get baseline
    target_id, target_prob, wanted_id, wanted_prob = ga.get_baseline_probability()
    ga.target_token_id = target_id
    ga.baseline_target_probability = target_prob
    ga.wanted_token_id = wanted_id
    ga.baseline_wanted_probability = wanted_prob or 0.0

    print(f"Running {ga.max_generations} generations with population size {ga.population_size}")

    # Create initial population
    population = ga.create_initial_population()

    # Evaluate initial population
    for individual in population:
        individual.fitness = ga.evaluate_fitness(individual)

    # Sort by fitness
    population.sort(key=lambda x: x.fitness, reverse=True)

    print(f"\nBest individual from initial population:")
    best = population[0]
    token_texts = [ga.tokenizer.decode([tid]) for tid in best.tokens]
    print(f"  Tokens: {best.tokens} ({token_texts})")
    print(f"  Fitness: {best.fitness:.6f}")
    print(f"  Baseline prob: {best.baseline_prob:.6f}")
    print(f"  Modified prob: {best.modified_prob:.6f}")
    print(f"  Reduction: {best.target_reduction:.6f}")

    # Run a few generations
    for generation in range(ga.max_generations):
        print(f"\nGeneration {generation + 1}:")

        # Create new population
        new_population = []

        # Keep elite
        elite_size = min(2, len(population))
        new_population.extend(population[:elite_size])

        # Fill rest with crossover and mutation
        while len(new_population) < ga.population_size:
            parent1 = ga.tournament_selection(population)
            parent2 = ga.tournament_selection(population)
            child = ga.crossover(parent1, parent2)
            child = ga.mutate(child)
            child.fitness = ga.evaluate_fitness(child)
            new_population.append(child)

        population = new_population
        population.sort(key=lambda x: x.fitness, reverse=True)

        best = population[0]
        token_texts = [ga.tokenizer.decode([tid]) for tid in best.tokens]
        print(f"  Best: {best.tokens} ({token_texts}) - Fitness: {best.fitness:.6f}")

    print(f"\nFinal best individual:")
    best = population[0]
    token_texts = [ga.tokenizer.decode([tid]) for tid in best.tokens]
    print(f"  Tokens: {best.tokens} ({token_texts})")
    print(f"  Fitness: {best.fitness:.6f}")
    print(f"  Target reduction: {best.target_reduction:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Test chat template implementation")
    parser.add_argument("model_path", help="Path to the model to test")
    parser.add_argument("--save-results", help="Save test results to JSON file")
    parser.add_argument("--test-evolution", action="store_true", help="Test genetic evolution")

    args = parser.parse_args()

    print(f"Testing chat template implementation with model: {args.model_path}")

    try:
        # Test chat template formatting
        ga, formatted_base, formatted_modified = test_chat_template_formatting(args.model_path)

        # Test probability measurement
        test_probability_measurement(ga, formatted_base, formatted_modified)

        # Test genetic evolution if requested
        if args.test_evolution:
            test_genetic_evolution_sample(ga)

        print(f"\n{'='*80}")
        print("TEST COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")

        # Summary
        print(f"\nSummary:")
        print(f"  Model: {args.model_path}")
        print(f"  Is instruct model: {ga.is_instruct_model}")
        print(f"  Uses chat template: {ga.is_instruct_model and hasattr(ga.tokenizer, 'chat_template') and ga.tokenizer.chat_template is not None}")
        print(f"  System prompt set: {ga.system_prompt is not None}")

        if args.save_results:
            results = {
                'model_path': args.model_path,
                'is_instruct_model': ga.is_instruct_model,
                'has_chat_template': hasattr(ga.tokenizer, 'chat_template') and ga.tokenizer.chat_template is not None,
                'system_prompt': ga.system_prompt,
                'formatted_base': formatted_base,
                'formatted_modified': formatted_modified,
                'baseline_target_prob': ga.baseline_target_probability,
                'target_token_id': ga.target_token_id,
                'target_token': ga.tokenizer.decode([ga.target_token_id]) if ga.target_token_id else None
            }

            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.save_results}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
