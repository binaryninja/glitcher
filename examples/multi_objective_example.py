#!/usr/bin/env python3
"""
Multi-Objective Genetic Algorithm Example

This script demonstrates the new multi-objective genetic algorithm functionality
that can simultaneously:
1. Reduce probability of a target token
2. Increase probability of a wanted token
3. Use both glitch tokens and normal ASCII tokens from model vocabulary

Author: Claude
Date: 2024
"""

import argparse
import json
import logging
from pathlib import Path

from glitcher.genetic import GeneticProbabilityReducer


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def run_single_objective_example(model_name: str):
    """
    Example 1: Single objective (traditional) - only reduce target token probability.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Objective (Target Reduction Only)")
    print("="*80)

    ga = GeneticProbabilityReducer(
        model_name=model_name,
        base_text="The quick brown",
        target_token="fox"  # Try to reduce probability of "fox"
    )

    # Configure parameters
    ga.population_size = 30
    ga.max_generations = 50
    ga.max_tokens_per_individual = 3

    # Load model and tokens
    ga.load_model()
    ga.load_glitch_tokens(
        token_file="glitch_tokens.json",
        ascii_only=True,
        include_normal_tokens=True  # Include normal vocabulary tokens
    )

    # Run evolution
    print(f"Running single-objective optimization...")
    print(f"Target: Reduce probability of 'fox' after 'The quick brown'")

    final_population = ga.run_evolution()

    # Display results
    ga.display_results(final_population, top_n=5)

    # Save results
    ga.save_results(final_population, "single_objective_results.json")

    return final_population[0] if final_population else None


def run_multi_objective_example(model_name: str):
    """
    Example 2: Multi-objective - reduce target token AND increase wanted token.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Multi-Objective (Target Reduction + Wanted Increase)")
    print("="*80)

    ga = GeneticProbabilityReducer(
        model_name=model_name,
        base_text="The weather is",
        target_token="nice",     # Try to reduce probability of "nice"
        wanted_token="terrible"  # Try to increase probability of "terrible"
    )

    # Configure parameters
    ga.population_size = 40
    ga.max_generations = 60
    ga.max_tokens_per_individual = 4

    # Load model and tokens
    ga.load_model()
    ga.load_glitch_tokens(
        token_file="glitch_tokens.json",
        ascii_only=True,
        include_normal_tokens=True
    )

    # Run evolution
    print(f"Running multi-objective optimization...")
    print(f"Target: Reduce 'nice' AND increase 'terrible' after 'The weather is'")

    final_population = ga.run_evolution()

    # Display results
    ga.display_results(final_population, top_n=5)

    # Save results
    ga.save_results(final_population, "multi_objective_results.json")

    return final_population[0] if final_population else None


def run_vocabulary_only_example(model_name: str):
    """
    Example 3: Use only normal vocabulary tokens (no glitch tokens).
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Normal Vocabulary Tokens Only")
    print("="*80)

    ga = GeneticProbabilityReducer(
        model_name=model_name,
        base_text="I am feeling",
        target_token="happy",
        wanted_token="sad"
    )

    # Configure parameters
    ga.population_size = 25
    ga.max_generations = 40
    ga.max_tokens_per_individual = 2

    # Load model and tokens
    ga.load_model()
    ga.load_glitch_tokens(
        token_file=None,  # No glitch tokens file
        ascii_only=True,
        include_normal_tokens=True  # Only use normal vocabulary
    )

    # Run evolution
    print(f"Running optimization with normal vocabulary tokens only...")
    print(f"Target: Reduce 'happy' AND increase 'sad' after 'I am feeling'")
    print(f"Token pool: {len(ga.available_tokens)} normal ASCII tokens from model vocabulary")

    final_population = ga.run_evolution()

    # Display results
    ga.display_results(final_population, top_n=5)

    # Save results
    ga.save_results(final_population, "vocabulary_only_results.json")

    return final_population[0] if final_population else None


def run_sentiment_manipulation_example(model_name: str):
    """
    Example 4: Sentiment manipulation scenario.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Sentiment Manipulation")
    print("="*80)

    scenarios = [
        {
            "base_text": "This movie is",
            "target_token": "amazing",
            "wanted_token": "awful",
            "description": "Movie review manipulation"
        },
        {
            "base_text": "The food tastes",
            "target_token": "delicious",
            "wanted_token": "horrible",
            "description": "Food review manipulation"
        },
        {
            "base_text": "The service was",
            "target_token": "excellent",
            "wanted_token": "terrible",
            "description": "Service review manipulation"
        }
    ]

    results = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i}: {scenario['description']} ---")

        ga = GeneticProbabilityReducer(
            model_name=model_name,
            base_text=scenario["base_text"],
            target_token=scenario["target_token"],
            wanted_token=scenario["wanted_token"]
        )

        # Configure parameters for quick testing
        ga.population_size = 20
        ga.max_generations = 30
        ga.max_tokens_per_individual = 3

        # Load model and tokens
        ga.load_model()
        ga.load_glitch_tokens(
            token_file="glitch_tokens.json",
            ascii_only=True,
            include_normal_tokens=True
        )

        print(f"Base text: '{scenario['base_text']}'")
        print(f"Reduce: '{scenario['target_token']}' | Increase: '{scenario['wanted_token']}'")

        # Run evolution
        final_population = ga.run_evolution()

        if final_population:
            best = final_population[0]
            results.append({
                "scenario": scenario["description"],
                "base_text": scenario["base_text"],
                "target_token": scenario["target_token"],
                "wanted_token": scenario["wanted_token"],
                "best_tokens": best.tokens,
                "best_fitness": best.fitness,
                "target_reduction": best.target_reduction,
                "wanted_increase": best.wanted_increase
            })

            # Show quick summary
            token_texts = [ga.tokenizer.decode([tid]) for tid in best.tokens]
            print(f"Best result: {best.tokens} → {token_texts}")
            print(f"Combined fitness: {best.fitness:.4f}")

        # Save individual results
        output_file = f"sentiment_scenario_{i}_results.json"
        ga.save_results(final_population, output_file)

    # Save combined results
    with open("sentiment_manipulation_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def analyze_token_sources(model_name: str):
    """
    Example 5: Analyze the effectiveness of different token sources.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Token Source Analysis")
    print("="*80)

    base_text = "The computer program"
    target_token = "works"
    wanted_token = "crashes"

    configs = [
        {
            "name": "Glitch Tokens Only",
            "glitch_file": "glitch_tokens.json",
            "include_normal": False
        },
        {
            "name": "Normal Tokens Only",
            "glitch_file": None,
            "include_normal": True
        },
        {
            "name": "Combined Tokens",
            "glitch_file": "glitch_tokens.json",
            "include_normal": True
        }
    ]

    results = []

    for config in configs:
        print(f"\n--- Testing: {config['name']} ---")

        ga = GeneticProbabilityReducer(
            model_name=model_name,
            base_text=base_text,
            target_token=target_token,
            wanted_token=wanted_token
        )

        # Configure parameters
        ga.population_size = 25
        ga.max_generations = 30
        ga.max_tokens_per_individual = 3

        # Load model and tokens
        ga.load_model()
        ga.load_glitch_tokens(
            token_file=config["glitch_file"],
            ascii_only=True,
            include_normal_tokens=config["include_normal"]
        )

        print(f"Available tokens: {len(ga.available_tokens)}")
        print(f"Glitch tokens: {len(ga.glitch_tokens)}")
        print(f"Normal tokens: {len(ga.ascii_tokens)}")

        # Run evolution
        final_population = ga.run_evolution()

        if final_population:
            best = final_population[0]
            results.append({
                "config_name": config["name"],
                "total_tokens": len(ga.available_tokens),
                "glitch_tokens": len(ga.glitch_tokens),
                "normal_tokens": len(ga.ascii_tokens),
                "best_fitness": best.fitness,
                "target_reduction": best.target_reduction,
                "wanted_increase": best.wanted_increase,
                "best_token_ids": best.tokens
            })

    # Display comparison
    print("\n" + "="*60)
    print("TOKEN SOURCE COMPARISON")
    print("="*60)

    for result in results:
        print(f"\n{result['config_name']}:")
        print(f"  Token pool size: {result['total_tokens']}")
        print(f"  Best fitness: {result['best_fitness']:.4f}")
        print(f"  Target reduction: {result['target_reduction']:.4f}")
        print(f"  Wanted increase: {result['wanted_increase']:.4f}")

    # Save analysis
    with open("token_source_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    """Main function to run all examples."""
    parser = argparse.ArgumentParser(
        description="Multi-Objective Genetic Algorithm Examples"
    )
    parser.add_argument(
        "model_name",
        help="HuggingFace model identifier (e.g., meta-llama/Llama-3.2-1B-Instruct)"
    )
    parser.add_argument(
        "--example", type=str, choices=["single", "multi", "vocab", "sentiment", "analysis", "all"],
        default="all",
        help="Which example to run (default: all)"
    )
    parser.add_argument(
        "--glitch-tokens", type=str, default="glitch_tokens.json",
        help="Path to glitch tokens file"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    print("🧬 Multi-Objective Genetic Algorithm Examples")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Glitch tokens file: {args.glitch_tokens}")

    # Check if glitch tokens file exists
    if args.glitch_tokens and not Path(args.glitch_tokens).exists():
        print(f"⚠️  Warning: Glitch tokens file not found: {args.glitch_tokens}")
        print("Some examples will run with normal vocabulary tokens only.")

    try:
        if args.example == "single" or args.example == "all":
            run_single_objective_example(args.model_name)

        if args.example == "multi" or args.example == "all":
            run_multi_objective_example(args.model_name)

        if args.example == "vocab" or args.example == "all":
            run_vocabulary_only_example(args.model_name)

        if args.example == "sentiment" or args.example == "all":
            run_sentiment_manipulation_example(args.model_name)

        if args.example == "analysis" or args.example == "all":
            analyze_token_sources(args.model_name)

        print("\n" + "="*80)
        print("✅ All examples completed successfully!")
        print("Check the generated JSON files for detailed results.")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
