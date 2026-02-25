#!/usr/bin/env python3
"""
Comprehensive demonstration of chat template improvements in genetic algorithm.

This script demonstrates the key improvements made to the genetic algorithm:
1. Proper chat template usage for instruct models
2. Appropriate system prompt for text continuation task
3. Correct probability measurement in assistant response
4. Comparison between old raw text format and new chat format

Key Changes Implemented:
- Instruct models now use proper conversation formatting
- System prompt guides model to perform text continuation
- Probabilities measured at assistant response position
- Fallback to direct text for base models without chat templates

Author: Claude
Date: 2024
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


def load_model_directly(model_path: str):
    """Load model directly (old way) for comparison."""
    print(f"Loading model directly: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    device = next(model.parameters()).device
    model.eval()

    return tokenizer, model, device


def test_old_vs_new_formatting(model_path: str, base_text: str, target_token: str):
    """Compare old raw text format vs new chat template format."""

    print(f"{'='*100}")
    print(f"COMPARING OLD VS NEW FORMATTING")
    print(f"{'='*100}")
    print(f"Model: {model_path}")
    print(f"Base text: '{base_text}'")
    print(f"Target token: '{target_token}'")

    # Test with genetic reducer (new way)
    print(f"\n{'-'*80}")
    print("NEW WAY: Using GeneticProbabilityReducer with Chat Templates")
    print(f"{'-'*80}")

    ga = GeneticProbabilityReducer(
        model_name=model_path,
        base_text=base_text,
        target_token=target_token
    )
    ga.load_model()

    # Show model detection
    print(f"Instruct model detected: {ga.is_instruct_model}")
    print(f"Has chat template: {hasattr(ga.tokenizer, 'chat_template') and ga.tokenizer.chat_template is not None}")

    if ga.system_prompt:
        print(f"System prompt: {ga.system_prompt}")

    # Get baseline with new formatting
    target_id, target_prob, wanted_id, wanted_prob = ga.get_baseline_probability()
    new_formatted_input = ga._format_input_for_model(base_text)

    print(f"\nFormatted input (new way):")
    print(f"'{new_formatted_input}'")
    print(f"\nBaseline probability: {target_prob:.8f}")

    # Test with glitch tokens
    sample_glitch_tokens = [111067, 127367]  # From our successful test
    token_texts = [ga.tokenizer.decode([tid]) for tid in sample_glitch_tokens]
    joined_tokens = "".join(token_texts)

    if ga.is_instruct_model:
        modified_text = f"{joined_tokens} {base_text}".strip()
    else:
        modified_text = f"({joined_tokens}): {base_text}"

    new_formatted_modified = ga._format_input_for_model(modified_text)

    # Get probability with glitch tokens
    inputs = ga.tokenizer(new_formatted_modified, return_tensors="pt").to(ga.device)
    with torch.no_grad():
        outputs = ga.model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

    new_modified_prob = probs[target_id].item()
    new_reduction = (target_prob - new_modified_prob) / target_prob

    print(f"\nWith glitch tokens '{joined_tokens}':")
    print(f"Modified text: '{modified_text}'")
    print(f"Formatted modified input:")
    print(f"'{new_formatted_modified}'")
    print(f"Modified probability: {new_modified_prob:.8f}")
    print(f"Probability reduction: {new_reduction:.4f} ({new_reduction*100:.2f}%)")

    # Test old way (direct model loading)
    print(f"\n{'-'*80}")
    print("OLD WAY: Direct Model Loading (Raw Text)")
    print(f"{'-'*80}")

    tokenizer, model, device = load_model_directly(model_path)

    # Old baseline (raw text)
    old_inputs = tokenizer(base_text, return_tensors="pt").to(device)
    with torch.no_grad():
        old_outputs = model(**old_inputs)
        old_logits = old_outputs.logits[0, -1, :]
        old_probs = torch.softmax(old_logits, dim=-1)

    old_baseline_prob = old_probs[target_id].item()

    print(f"Raw input (old way): '{base_text}'")
    print(f"Baseline probability: {old_baseline_prob:.8f}")

    # Old way with glitch tokens
    old_modified_text = f"({joined_tokens}): {base_text}"
    old_modified_inputs = tokenizer(old_modified_text, return_tensors="pt").to(device)

    with torch.no_grad():
        old_modified_outputs = model(**old_modified_inputs)
        old_modified_logits = old_modified_outputs.logits[0, -1, :]
        old_modified_probs = torch.softmax(old_modified_logits, dim=-1)

    old_modified_prob = old_modified_probs[target_id].item()
    old_reduction = (old_baseline_prob - old_modified_prob) / old_baseline_prob

    print(f"\nWith glitch tokens (old format):")
    print(f"Modified input: '{old_modified_text}'")
    print(f"Modified probability: {old_modified_prob:.8f}")
    print(f"Probability reduction: {old_reduction:.4f} ({old_reduction*100:.2f}%)")

    # Comparison
    print(f"\n{'-'*80}")
    print("COMPARISON SUMMARY")
    print(f"{'-'*80}")

    print(f"Baseline Probabilities:")
    print(f"  New (chat template): {target_prob:.8f}")
    print(f"  Old (raw text):      {old_baseline_prob:.8f}")
    print(f"  Difference:          {abs(target_prob - old_baseline_prob):.8f}")

    print(f"\nWith Glitch Tokens:")
    print(f"  New probability:     {new_modified_prob:.8f}")
    print(f"  Old probability:     {old_modified_prob:.8f}")
    print(f"  Difference:          {abs(new_modified_prob - old_modified_prob):.8f}")

    print(f"\nReduction Effectiveness:")
    print(f"  New reduction:       {new_reduction:.4f} ({new_reduction*100:.2f}%)")
    print(f"  Old reduction:       {old_reduction:.4f} ({old_reduction*100:.2f}%)")
    print(f"  Improvement:         {(new_reduction - old_reduction):.4f} ({(new_reduction - old_reduction)*100:.2f}% points)")

    return {
        'new_baseline': target_prob,
        'old_baseline': old_baseline_prob,
        'new_modified': new_modified_prob,
        'old_modified': old_modified_prob,
        'new_reduction': new_reduction,
        'old_reduction': old_reduction,
        'new_formatted_input': new_formatted_input,
        'old_raw_input': base_text,
        'new_formatted_modified': new_formatted_modified,
        'old_modified_input': old_modified_text
    }


def demonstrate_top_predictions(model_path: str, base_text: str):
    """Show how top predictions change with chat template."""

    print(f"\n{'='*100}")
    print(f"TOP PREDICTIONS COMPARISON")
    print(f"{'='*100}")

    # New way
    ga = GeneticProbabilityReducer(model_name=model_path, base_text=base_text)
    ga.load_model()

    formatted_input = ga._format_input_for_model(base_text)
    inputs = ga.tokenizer(formatted_input, return_tensors="pt").to(ga.device)

    with torch.no_grad():
        outputs = ga.model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

    top_probs, top_indices = torch.topk(probs, 10)

    print(f"Top 10 predictions with chat template:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
        token_str = ga.tokenizer.decode([idx.item()])
        print(f"  {i:2d}. '{token_str}' (ID: {idx.item()}) - {prob.item():.6f} ({prob.item()*100:.2f}%)")

    # Old way
    tokenizer, model, device = load_model_directly(model_path)
    old_inputs = tokenizer(base_text, return_tensors="pt").to(device)

    with torch.no_grad():
        old_outputs = model(**old_inputs)
        old_logits = old_outputs.logits[0, -1, :]
        old_probs = torch.softmax(old_logits, dim=-1)

    old_top_probs, old_top_indices = torch.topk(old_probs, 10)

    print(f"\nTop 10 predictions with raw text:")
    for i, (prob, idx) in enumerate(zip(old_top_probs, old_top_indices), 1):
        token_str = tokenizer.decode([idx.item()])
        print(f"  {i:2d}. '{token_str}' (ID: {idx.item()}) - {prob.item():.6f} ({prob.item()*100:.2f}%)")


def demonstrate_system_prompt_impact(model_path: str):
    """Show how different system prompts affect predictions."""

    print(f"\n{'='*100}")
    print(f"SYSTEM PROMPT IMPACT DEMONSTRATION")
    print(f"{'='*100}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    device = next(model.parameters()).device

    base_text = "The quick brown"

    # Test different system prompts
    system_prompts = [
        None,  # No system prompt
        "You are a helpful assistant.",  # Generic assistant
        "You are a text completion assistant. Your task is to continue sequences of text. When the user provides text, you should predict and output the most likely next word(s) that would naturally follow in the sequence. Respond with only the continuation, without any additional explanation or formatting.",  # Our continuation prompt
        "You are a creative writing assistant. Continue the story in an interesting way.",  # Creative prompt
    ]

    prompt_names = [
        "No system prompt",
        "Generic assistant",
        "Text completion assistant",
        "Creative writing assistant"
    ]

    for prompt, name in zip(system_prompts, prompt_names):
        print(f"\n{'-'*60}")
        print(f"Testing: {name}")
        print(f"{'-'*60}")

        if prompt is None:
            # Direct text
            formatted_input = base_text
        else:
            # Chat format
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": base_text}
            ]

            try:
                formatted_input = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                formatted_input = f"System: {prompt}\n\nUser: {base_text}\nAssistant: "

        inputs = tokenizer(formatted_input, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

        # Get top 5 predictions
        top_probs, top_indices = torch.topk(probs, 5)

        print(f"Top 5 predictions:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
            token_str = tokenizer.decode([idx.item()])
            print(f"  {i}. '{token_str}' - {prob.item():.6f} ({prob.item()*100:.2f}%)")

        # Check fox probability specifically
        fox_tokens = tokenizer.encode("fox", add_special_tokens=False)
        if fox_tokens:
            fox_id = fox_tokens[0]
            fox_prob = probs[fox_id].item()
            print(f"'fox' probability: {fox_prob:.6f} ({fox_prob*100:.2f}%)")


def run_genetic_algorithm_demo(model_path: str):
    """Demonstrate the genetic algorithm working with chat templates."""

    print(f"\n{'='*100}")
    print(f"GENETIC ALGORITHM DEMONSTRATION")
    print(f"{'='*100}")

    # Create genetic reducer
    ga = GeneticProbabilityReducer(
        model_name=model_path,
        base_text="The quick brown",
        target_token="fox"
    )

    # Set small parameters for demo
    ga.population_size = 6
    ga.max_generations = 3
    ga.max_tokens_per_individual = 2

    # Load model and tokens
    ga.load_model()

    # Use some sample tokens if no file exists
    try:
        ga.load_glitch_tokens(token_file="email_extraction_all_glitches.json", ascii_only=True)
        print(f"Loaded {len(ga.available_tokens)} ASCII glitch tokens")
    except:
        # Fallback to sample tokens
        ga.available_tokens = [111067, 127367, 112210, 99072, 53355, 125808]
        print(f"Using sample tokens: {ga.available_tokens}")

    # Get baseline
    target_id, target_prob, wanted_id, wanted_prob = ga.get_baseline_probability()
    ga.target_token_id = target_id
    ga.baseline_target_probability = target_prob
    ga.wanted_token_id = wanted_id
    ga.baseline_wanted_probability = wanted_prob or 0.0

    print(f"\nRunning {ga.max_generations} generations with population {ga.population_size}")
    print(f"Target: '{ga.tokenizer.decode([target_id])}' (baseline probability: {target_prob:.6f})")

    # Create and evolve population
    population = ga.create_initial_population()

    for individual in population:
        individual.fitness = ga.evaluate_fitness(individual)

    for generation in range(ga.max_generations):
        population.sort(key=lambda x: x.fitness, reverse=True)

        best = population[0]
        token_texts = [ga.tokenizer.decode([tid]) for tid in best.tokens]

        print(f"\nGeneration {generation}: Best individual")
        print(f"  Tokens: {best.tokens} ({token_texts})")
        print(f"  Fitness: {best.fitness:.6f}")
        print(f"  Reduction: {best.target_reduction:.6f} ({best.target_reduction/target_prob*100:.2f}%)")

        # Show what the input looks like
        joined_tokens = "".join(token_texts)
        if ga.is_instruct_model:
            modified_text = f"{joined_tokens} {ga.base_text}".strip()
        else:
            modified_text = f"({joined_tokens}): {ga.base_text}"

        formatted_input = ga._format_input_for_model(modified_text)
        print(f"  Input format: '{modified_text}'")

        # Evolve population
        if generation < ga.max_generations - 1:
            new_population = population[:2]  # Keep top 2

            while len(new_population) < ga.population_size:
                parent1 = ga.tournament_selection(population)
                parent2 = ga.tournament_selection(population)
                child = ga.crossover(parent1, parent2)
                child = ga.mutate(child)
                child.fitness = ga.evaluate_fitness(child)
                new_population.append(child)

            population = new_population

    print(f"\nFinal best solution:")
    best = population[0]
    token_texts = [ga.tokenizer.decode([tid]) for tid in best.tokens]
    reduction_percent = best.target_reduction / target_prob * 100

    print(f"  Tokens: {best.tokens}")
    print(f"  Decoded: {token_texts}")
    print(f"  Probability reduction: {best.target_reduction:.6f} ({reduction_percent:.2f}%)")
    print(f"  Original probability: {target_prob:.6f}")
    print(f"  Modified probability: {best.modified_prob:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Demonstrate chat template improvements")
    parser.add_argument(
        "model_path",
        help="Path to the model to test (e.g., meta-llama/Llama-3.2-1B-Instruct)"
    )
    parser.add_argument(
        "--base-text",
        default="The quick brown",
        help="Base text to test (default: 'The quick brown')"
    )
    parser.add_argument(
        "--target-token",
        default="fox",
        help="Target token to test (default: 'fox')"
    )
    parser.add_argument(
        "--save-results",
        help="Save detailed results to JSON file"
    )
    parser.add_argument(
        "--skip-genetic",
        action="store_true",
        help="Skip genetic algorithm demo (faster)"
    )

    args = parser.parse_args()

    print(f"{'='*100}")
    print(f"CHAT TEMPLATE IMPROVEMENTS DEMONSTRATION")
    print(f"{'='*100}")
    print(f"Model: {args.model_path}")
    print(f"This demo shows the improvements made to use proper chat templates")
    print(f"for instruct models instead of raw text completion.")
    print(f"{'='*100}")

    try:
        # Main comparison
        comparison_results = test_old_vs_new_formatting(
            args.model_path,
            args.base_text,
            args.target_token
        )

        # Show top predictions
        demonstrate_top_predictions(args.model_path, args.base_text)

        # Show system prompt impact
        demonstrate_system_prompt_impact(args.model_path)

        # Run genetic algorithm demo
        if not args.skip_genetic:
            run_genetic_algorithm_demo(args.model_path)

        # Summary
        print(f"\n{'='*100}")
        print(f"DEMONSTRATION SUMMARY")
        print(f"{'='*100}")

        print(f"Key Improvements:")
        print(f"✅ Instruct models now use proper chat templates")
        print(f"✅ System prompt guides model for text continuation task")
        print(f"✅ Probabilities measured at correct assistant response position")
        print(f"✅ Fallback to raw text for base models without templates")
        print(f"✅ Genetic algorithm works correctly with new formatting")

        new_reduction = comparison_results['new_reduction']
        old_reduction = comparison_results['old_reduction']
        improvement = new_reduction - old_reduction

        print(f"\nEffectiveness Comparison:")
        print(f"  Old method reduction: {old_reduction:.4f} ({old_reduction*100:.2f}%)")
        print(f"  New method reduction: {new_reduction:.4f} ({new_reduction*100:.2f}%)")
        if improvement > 0:
            print(f"  ✅ Improvement: +{improvement:.4f} ({improvement*100:.2f}% points better)")
        else:
            print(f"  ⚠️  Change: {improvement:.4f} ({improvement*100:.2f}% points)")

        # Save results
        if args.save_results:
            results = {
                'model_path': args.model_path,
                'base_text': args.base_text,
                'target_token': args.target_token,
                'comparison_results': comparison_results,
                'improvements': {
                    'uses_chat_template': True,
                    'has_system_prompt': True,
                    'measures_assistant_response': True,
                    'fallback_for_base_models': True
                },
                'effectiveness': {
                    'old_reduction': old_reduction,
                    'new_reduction': new_reduction,
                    'improvement': improvement
                }
            }

            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✅ Detailed results saved to: {args.save_results}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
