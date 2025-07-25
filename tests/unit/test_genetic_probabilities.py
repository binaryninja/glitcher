#!/usr/bin/env python3
"""
Test script for genetic algorithm token probability impacts
Tests the specific token sequence found by the genetic algorithm
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

def calculate_sequence_probability(tokenizer, model, device, token_ids, target_text="Hello world"):
    """Calculate probability of target text after given token sequence"""

    # Encode the target text
    target_tokens = tokenizer.encode(target_text, add_special_tokens=False)

    # Create input sequence: prefix tokens + target tokens
    input_ids = torch.tensor([token_ids + target_tokens], device=device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

        # Calculate probabilities for each target token
        log_probs = []
        total_log_prob = 0.0

        print(f"\nSequence analysis for target: '{target_text}'")
        print("=" * 60)

        for i, target_token_id in enumerate(target_tokens):
            # Position in sequence (after prefix tokens)
            pos = len(token_ids) + i - 1

            if pos >= 0 and pos < logits.shape[1]:
                token_logits = logits[0, pos, :]
                token_probs = torch.softmax(token_logits, dim=-1)

                target_prob = token_probs[target_token_id].item()
                target_log_prob = torch.log(token_probs[target_token_id]).item()

                log_probs.append(target_log_prob)
                total_log_prob += target_log_prob

                target_token_str = tokenizer.decode([target_token_id])
                print(f"Token {i+1}: '{target_token_str}' (ID: {target_token_id}) - Prob: {target_prob:.6f}")

        # Calculate overall probability
        overall_prob = torch.exp(torch.tensor(total_log_prob)).item()
        print(f"\nOverall sequence probability: {overall_prob:.8f}")
        print(f"Log probability: {total_log_prob:.6f}")

    return overall_prob, total_log_prob

def test_baseline_probability(tokenizer, model, device, target_text="Hello world"):
    """Test probability without any prefix tokens"""
    print(f"\n{'='*60}")
    print("BASELINE TEST (no prefix tokens)")
    print(f"{'='*60}")

    return calculate_sequence_probability(tokenizer, model, device, [], target_text)

def test_genetic_tokens_probability(tokenizer, model, device, genetic_tokens, target_text="Hello world"):
    """Test probability with genetic algorithm tokens as prefix"""
    print(f"\n{'='*60}")
    print("GENETIC TOKENS TEST")
    print(f"{'='*60}")

    # Decode and display the genetic tokens
    genetic_token_strings = [tokenizer.decode([tid]) for tid in genetic_tokens]
    print("Genetic token sequence:")
    for i, (tid, token_str) in enumerate(zip(genetic_tokens, genetic_token_strings)):
        print(f"  {i+1}. Token ID {tid}: '{token_str}'")

    return calculate_sequence_probability(tokenizer, model, device, genetic_tokens, target_text)

def test_individual_tokens(tokenizer, model, device, genetic_tokens, target_text="Hello world"):
    """Test each genetic token individually"""
    print(f"\n{'='*60}")
    print("INDIVIDUAL TOKEN TESTS")
    print(f"{'='*60}")

    results = []
    for i, token_id in enumerate(genetic_tokens):
        token_str = tokenizer.decode([token_id])
        print(f"\nTesting single token {i+1}: '{token_str}' (ID: {token_id})")
        prob, log_prob = calculate_sequence_probability(tokenizer, model, device, [token_id], target_text)
        results.append({
            'token_id': token_id,
            'token_str': token_str,
            'probability': prob,
            'log_probability': log_prob
        })

    return results

def main():
    parser = argparse.ArgumentParser(description="Test genetic algorithm token probability impacts")
    parser.add_argument("model_path", help="Path to the model")
    parser.add_argument("--target-text", default="Hello world", help="Target text to test probability for")
    parser.add_argument("--save-results", help="Save results to JSON file")

    args = parser.parse_args()

    # The best token sequence from genetic algorithm
    genetic_tokens = [115614, 118508, 121267, 112063, 118508, 126357]

    # Load model
    tokenizer, model, device = load_model(args.model_path)

    print(f"\nTarget text: '{args.target_text}'")
    print(f"Genetic token sequence: {genetic_tokens}")

    # Test baseline probability
    baseline_prob, baseline_log_prob = test_baseline_probability(tokenizer, model, device, args.target_text)

    # Test with genetic tokens
    genetic_prob, genetic_log_prob = test_genetic_tokens_probability(tokenizer, model, device, genetic_tokens, args.target_text)

    # Test individual tokens
    individual_results = test_individual_tokens(tokenizer, model, device, genetic_tokens, args.target_text)

    # Calculate impact
    prob_reduction = (baseline_prob - genetic_prob) / baseline_prob if baseline_prob > 0 else 0
    prob_ratio = genetic_prob / baseline_prob if baseline_prob > 0 else 0

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Target text: '{args.target_text}'")
    print(f"Baseline probability: {baseline_prob:.8f}")
    print(f"Genetic tokens probability: {genetic_prob:.8f}")
    print(f"Probability reduction: {prob_reduction:.4f} ({prob_reduction*100:.2f}%)")
    print(f"Probability ratio: {prob_ratio:.6f}")
    print(f"Log probability change: {genetic_log_prob - baseline_log_prob:.6f}")

    # Prepare results
    results = {
        'model_path': args.model_path,
        'target_text': args.target_text,
        'genetic_tokens': genetic_tokens,
        'genetic_token_strings': [tokenizer.decode([tid]) for tid in genetic_tokens],
        'baseline': {
            'probability': baseline_prob,
            'log_probability': baseline_log_prob
        },
        'genetic_sequence': {
            'probability': genetic_prob,
            'log_probability': genetic_log_prob
        },
        'impact': {
            'probability_reduction': prob_reduction,
            'probability_ratio': prob_ratio,
            'log_probability_change': genetic_log_prob - baseline_log_prob
        },
        'individual_tokens': individual_results
    }

    # Save results if requested
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.save_results}")

    # Test with different target texts
    print(f"\n{'='*60}")
    print("TESTING WITH DIFFERENT TARGET TEXTS")
    print(f"{'='*60}")

    test_texts = [
        "The quick brown fox",
        "I am a helpful assistant",
        "Hello, how are you?",
        "What is the meaning of life?",
        "Please help me with"
    ]

    for test_text in test_texts:
        print(f"\nTesting with: '{test_text}'")
        baseline_p, _ = test_baseline_probability(tokenizer, model, device, test_text)
        genetic_p, _ = test_genetic_tokens_probability(tokenizer, model, device, genetic_tokens, test_text)

        if baseline_p > 0:
            reduction = (baseline_p - genetic_p) / baseline_p
            ratio = genetic_p / baseline_p
            print(f"  Baseline: {baseline_p:.8f}")
            print(f"  Genetic:  {genetic_p:.8f}")
            print(f"  Reduction: {reduction:.4f} ({reduction*100:.2f}%)")
            print(f"  Ratio: {ratio:.6f}")

if __name__ == "__main__":
    main()
