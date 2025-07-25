#!/usr/bin/env python3
"""
Test script with proper tokenization context for "The quick brown fox"
This script addresses the tokenization issues discovered in our investigation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json

def load_model(model_path):
    """Load the tokenizer and model"""
    print(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    device = next(model.parameters()).device
    print(f"Model loaded on {device}")
    return tokenizer, model, device

def get_next_token_probability(tokenizer, model, device, context_tokens, target_token_text):
    """Get probability of a specific next token given context tokens"""

    # Tokenize the target to get its ID
    target_tokens = tokenizer.encode(target_token_text, add_special_tokens=False)
    if len(target_tokens) != 1:
        # Handle multi-token targets by taking the first token
        print(f"Warning: '{target_token_text}' tokenizes to {len(target_tokens)} tokens: {target_tokens}")
        print(f"Using first token: {target_tokens[0]}")

    target_token_id = target_tokens[0]

    # Get model predictions
    input_ids = torch.tensor([context_tokens], device=device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # Last position logits
        probs = torch.softmax(logits, dim=-1)

    target_prob = probs[target_token_id].item()
    rank = (probs > probs[target_token_id]).sum().item() + 1

    # Get top 5 for context
    top_probs, top_indices = torch.topk(probs, 5)
    top_completions = []
    for prob, idx in zip(top_probs, top_indices):
        token_str = tokenizer.decode([idx.item()])
        top_completions.append((token_str, prob.item()))

    return {
        'target_token': target_token_text,
        'target_token_id': target_token_id,
        'probability': target_prob,
        'rank': rank,
        'top_completions': top_completions
    }

def test_context_variations(tokenizer, model, device, genetic_tokens):
    """Test different context variations for 'The quick brown' -> 'fox'"""

    print(f"\n{'='*80}")
    print("TESTING PROPER TOKENIZATION CONTEXTS")
    print(f"{'='*80}")

    # Different context variations to test
    contexts = [
        "The quick brown",
        "the quick brown",
        "The quick brown fox",  # Complete phrase
        "the quick brown fox"   # Complete phrase lowercase
    ]

    target_tokens = ["fox", " fox"]

    results = {}

    for context in contexts:
        print(f"\n{'-'*60}")
        print(f"Context: '{context}'")
        print(f"{'-'*60}")

        # Analyze tokenization
        context_tokens = tokenizer.encode(context, add_special_tokens=False)
        context_token_strings = [tokenizer.decode([t]) for t in context_tokens]
        print(f"Tokenization: {context_tokens} -> {context_token_strings}")

        results[context] = {}

        for target in target_tokens:
            if "fox" in context.lower() and target == "fox":
                # Skip testing "fox" after contexts that already contain "fox"
                continue

            print(f"\nTesting target: '{target}'")

            # Baseline test
            baseline = get_next_token_probability(tokenizer, model, device, context_tokens, target)
            print(f"BASELINE - Probability: {baseline['probability']:.6f} ({baseline['probability']*100:.2f}%), Rank: {baseline['rank']}")

            # Test with genetic tokens prefix
            prefixed_tokens = genetic_tokens + context_tokens
            genetic = get_next_token_probability(tokenizer, model, device, prefixed_tokens, target)
            print(f"GENETIC  - Probability: {genetic['probability']:.6f} ({genetic['probability']*100:.2f}%), Rank: {genetic['rank']}")

            # Calculate impact
            if baseline['probability'] > 0:
                ratio = genetic['probability'] / baseline['probability']
                change = genetic['probability'] - baseline['probability']
                print(f"IMPACT   - Ratio: {ratio:.2f}x, Change: {change:+.6f}")
            else:
                print(f"IMPACT   - Baseline was 0, genetic: {genetic['probability']:.6f}")

            # Store results
            results[context][target] = {
                'baseline': baseline,
                'genetic': genetic
            }

            # Show top completions for context
            print(f"Top 5 baseline completions:")
            for i, (token, prob) in enumerate(baseline['top_completions'], 1):
                print(f"  {i}. '{token}' - {prob:.6f} ({prob*100:.2f}%)")

    return results

def test_famous_phrase_completion(tokenizer, model, device, genetic_tokens):
    """Test the exact famous phrase scenario properly"""

    print(f"\n{'='*80}")
    print("TESTING FAMOUS PHRASE COMPLETION PROPERLY")
    print(f"{'='*80}")

    # The exact scenario: what comes after "The quick brown"
    context = "The quick brown"
    context_tokens = tokenizer.encode(context, add_special_tokens=False)

    print(f"Context: '{context}'")
    print(f"Context tokens: {context_tokens}")
    print(f"Context token strings: {[tokenizer.decode([t]) for t in context_tokens]}")

    # Test both " fox" and "fox" variants
    fox_variants = [" fox", "fox", " Fox", "Fox"]

    print(f"\n{'-'*40}")
    print("BASELINE PREDICTIONS")
    print(f"{'-'*40}")

    baseline_results = {}
    input_ids = torch.tensor([context_tokens], device=device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

    # Get top 10 predictions
    top_probs, top_indices = torch.topk(probs, 10)
    print("Top 10 completions:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
        token_str = tokenizer.decode([idx.item()])
        print(f"  {i:2d}. '{token_str}' (ID: {idx.item()}) - {prob.item():.6f} ({prob.item()*100:.2f}%)")

    # Check fox variants
    print(f"\nFox variant probabilities:")
    for variant in fox_variants:
        variant_tokens = tokenizer.encode(variant, add_special_tokens=False)
        if len(variant_tokens) == 1:
            token_id = variant_tokens[0]
            prob = probs[token_id].item()
            rank = (probs > probs[token_id]).sum().item() + 1
            baseline_results[variant] = {'prob': prob, 'rank': rank, 'token_id': token_id}
            print(f"  '{variant}' (ID: {token_id}) - {prob:.6f} ({prob*100:.2f}%) - Rank: {rank}")

    # Test with genetic tokens
    print(f"\n{'-'*40}")
    print("WITH GENETIC TOKENS PREFIX")
    print(f"{'-'*40}")

    genetic_context = genetic_tokens + context_tokens
    input_ids = torch.tensor([genetic_context], device=device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

    # Get top 10 predictions with genetic prefix
    top_probs, top_indices = torch.topk(probs, 10)
    print("Top 10 completions with genetic prefix:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
        token_str = tokenizer.decode([idx.item()])
        print(f"  {i:2d}. '{token_str}' (ID: {idx.item()}) - {prob.item():.6f} ({prob.item()*100:.2f}%)")

    # Check fox variants with genetic prefix
    print(f"\nFox variant probabilities with genetic prefix:")
    genetic_results = {}
    for variant in fox_variants:
        if variant in baseline_results:
            token_id = baseline_results[variant]['token_id']
            prob = probs[token_id].item()
            rank = (probs > probs[token_id]).sum().item() + 1
            genetic_results[variant] = {'prob': prob, 'rank': rank}

            baseline_prob = baseline_results[variant]['prob']
            ratio = prob / baseline_prob if baseline_prob > 0 else float('inf')
            change = prob - baseline_prob

            print(f"  '{variant}' (ID: {token_id}) - {prob:.6f} ({prob*100:.2f}%) - Rank: {rank}")
            print(f"    Impact: {ratio:.2f}x ratio, {change:+.6f} change")

    return baseline_results, genetic_results

def main():
    parser = argparse.ArgumentParser(description="Test proper tokenization context for 'The quick brown fox'")
    parser.add_argument("model_path", help="Path to the model")
    parser.add_argument("--save-results", help="Save results to JSON file")

    args = parser.parse_args()

    # The genetic algorithm tokens
    genetic_tokens = [115614, 118508, 121267, 112063, 118508, 126357]

    # Load model
    tokenizer, model, device = load_model(args.model_path)

    print(f"\nGenetic tokens: {genetic_tokens}")
    genetic_token_strings = [tokenizer.decode([t]) for t in genetic_tokens]
    print(f"Genetic token strings: {genetic_token_strings}")

    # Run tests
    context_results = test_context_variations(tokenizer, model, device, genetic_tokens)
    baseline_fox, genetic_fox = test_famous_phrase_completion(tokenizer, model, device, genetic_tokens)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    print("\nKey findings for 'The quick brown' -> ' fox':")
    if ' fox' in baseline_fox and ' fox' in genetic_fox:
        baseline_prob = baseline_fox[' fox']['prob']
        genetic_prob = genetic_fox[' fox']['prob']
        ratio = genetic_prob / baseline_prob if baseline_prob > 0 else float('inf')

        print(f"  Baseline probability: {baseline_prob:.6f} ({baseline_prob*100:.2f}%)")
        print(f"  Genetic probability:  {genetic_prob:.6f} ({genetic_prob*100:.2f}%)")
        print(f"  Impact ratio: {ratio:.2f}x")

        if ratio > 1:
            print(f"  -> Genetic tokens INCREASE ' fox' probability by {ratio:.1f}x")
        else:
            print(f"  -> Genetic tokens DECREASE ' fox' probability by {1/ratio:.1f}x")

    # Save results
    if args.save_results:
        results = {
            'model_path': args.model_path,
            'genetic_tokens': genetic_tokens,
            'genetic_token_strings': genetic_token_strings,
            'context_variations': context_results,
            'famous_phrase_baseline': baseline_fox,
            'famous_phrase_genetic': genetic_fox
        }

        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.save_results}")

if __name__ == "__main__":
    main()
