#!/usr/bin/env python3
"""
Test script for next-token probability impacts with genetic algorithm tokens
Specifically tests how prefix tokens affect the probability of specific next tokens
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

def get_next_token_probabilities(tokenizer, model, device, context_tokens, target_token_id=None, top_k=10):
    """Get probability distribution for next token given context"""

    input_ids = torch.tensor([context_tokens], device=device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # Last position logits
        probs = torch.softmax(logits, dim=-1)

    results = {
        'context_length': len(context_tokens),
        'context_tokens': context_tokens
    }

    # Get specific target token probability if provided
    if target_token_id is not None:
        target_prob = probs[target_token_id].item()
        target_token_str = tokenizer.decode([target_token_id])
        results['target_token'] = {
            'id': target_token_id,
            'token': target_token_str,
            'probability': target_prob,
            'log_probability': torch.log(probs[target_token_id]).item(),
            'rank': (probs > probs[target_token_id]).sum().item() + 1
        }

    # Get top-k most likely tokens
    top_probs, top_indices = torch.topk(probs, top_k)
    top_tokens = []
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        token_str = tokenizer.decode([idx.item()])
        top_tokens.append({
            'rank': i + 1,
            'id': idx.item(),
            'token': token_str,
            'probability': prob.item(),
            'log_probability': torch.log(prob).item()
        })

    results['top_tokens'] = top_tokens

    return results

def test_next_token_impact(tokenizer, model, device, context_text, target_token, genetic_tokens):
    """Test how genetic tokens affect next-token probability"""

    print(f"\n{'='*80}")
    print(f"NEXT-TOKEN PROBABILITY TEST")
    print(f"{'='*80}")
    print(f"Context: '{context_text}'")
    print(f"Target next token: '{target_token}'")

    # Encode context and target
    context_tokens = tokenizer.encode(context_text, add_special_tokens=False)
    target_token_ids = tokenizer.encode(target_token, add_special_tokens=False)

    if len(target_token_ids) != 1:
        print(f"Warning: Target token '{target_token}' encodes to {len(target_token_ids)} tokens: {target_token_ids}")
        print("Using the first token ID for analysis.")

    target_token_id = target_token_ids[0]

    print(f"Context tokens: {context_tokens}")
    print(f"Target token ID: {target_token_id}")

    # Test baseline (no prefix)
    print(f"\n{'-'*40}")
    print("BASELINE (no prefix)")
    print(f"{'-'*40}")
    baseline_results = get_next_token_probabilities(tokenizer, model, device, context_tokens, target_token_id)

    baseline_prob = baseline_results['target_token']['probability']
    baseline_rank = baseline_results['target_token']['rank']

    print(f"Probability of '{target_token}': {baseline_prob:.8f}")
    print(f"Rank: {baseline_rank}")
    print(f"Log probability: {baseline_results['target_token']['log_probability']:.6f}")

    print(f"\nTop 5 most likely next tokens:")
    for token_info in baseline_results['top_tokens'][:5]:
        print(f"  {token_info['rank']}. '{token_info['token']}' (ID: {token_info['id']}) - {token_info['probability']:.6f}")

    # Test with genetic tokens prefix
    print(f"\n{'-'*40}")
    print("WITH GENETIC TOKENS PREFIX")
    print(f"{'-'*40}")

    genetic_token_strings = [tokenizer.decode([tid]) for tid in genetic_tokens]
    print("Genetic tokens:")
    for i, (tid, token_str) in enumerate(zip(genetic_tokens, genetic_token_strings)):
        print(f"  {i+1}. '{token_str}' (ID: {tid})")

    prefixed_context = genetic_tokens + context_tokens
    genetic_results = get_next_token_probabilities(tokenizer, model, device, prefixed_context, target_token_id)

    genetic_prob = genetic_results['target_token']['probability']
    genetic_rank = genetic_results['target_token']['rank']

    print(f"\nProbability of '{target_token}': {genetic_prob:.8f}")
    print(f"Rank: {genetic_rank}")
    print(f"Log probability: {genetic_results['target_token']['log_probability']:.6f}")

    print(f"\nTop 5 most likely next tokens:")
    for token_info in genetic_results['top_tokens'][:5]:
        print(f"  {token_info['rank']}. '{token_info['token']}' (ID: {token_info['id']}) - {token_info['probability']:.6f}")

    # Calculate impact
    prob_change = genetic_prob - baseline_prob
    prob_ratio = genetic_prob / baseline_prob if baseline_prob > 0 else float('inf')
    prob_reduction = (baseline_prob - genetic_prob) / baseline_prob if baseline_prob > 0 else 0
    rank_change = genetic_rank - baseline_rank

    print(f"\n{'='*80}")
    print("IMPACT ANALYSIS")
    print(f"{'='*80}")
    print(f"Context: '{context_text}'")
    print(f"Target token: '{target_token}' (ID: {target_token_id})")
    print(f"")
    print(f"Baseline probability:     {baseline_prob:.8f}")
    print(f"With genetic prefix:      {genetic_prob:.8f}")
    print(f"")
    print(f"Probability change:       {prob_change:+.8f}")
    print(f"Probability ratio:        {prob_ratio:.6f}")
    print(f"Probability reduction:    {prob_reduction:.4f} ({prob_reduction*100:.2f}%)")
    print(f"")
    print(f"Baseline rank:            {baseline_rank}")
    print(f"With genetic prefix rank: {genetic_rank}")
    print(f"Rank change:              {rank_change:+d}")

    return {
        'context': context_text,
        'target_token': target_token,
        'target_token_id': target_token_id,
        'genetic_tokens': genetic_tokens,
        'genetic_token_strings': genetic_token_strings,
        'baseline': baseline_results,
        'genetic': genetic_results,
        'impact': {
            'probability_change': prob_change,
            'probability_ratio': prob_ratio,
            'probability_reduction': prob_reduction,
            'rank_change': rank_change
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Test next-token probability impacts with genetic tokens")
    parser.add_argument("model_path", help="Path to the model")
    parser.add_argument("--context", default="The quick brown", help="Context text")
    parser.add_argument("--target", default="fox", help="Target next token")
    parser.add_argument("--save-results", help="Save results to JSON file")

    args = parser.parse_args()

    # The best token sequence from genetic algorithm
    genetic_tokens = [115614, 118508, 121267, 112063, 118508, 126357]

    # Load model
    tokenizer, model, device = load_model(args.model_path)

    # Run the test
    results = test_next_token_impact(tokenizer, model, device, args.context, args.target, genetic_tokens)

    # Test multiple examples
    test_cases = [
        ("The quick brown", "fox"),
        ("The quick brown", " fox"),
        ("Hello", " world"),
        ("I am a", " helpful"),
        ("What is the", " meaning"),
        ("Please help", " me"),
        ("The capital of France is", " Paris"),
        ("2 + 2 =", " 4"),
    ]

    print(f"\n\n{'='*80}")
    print("MULTIPLE TEST CASES")
    print(f"{'='*80}")

    all_results = []
    for context, target in test_cases:
        try:
            result = test_next_token_impact(tokenizer, model, device, context, target, genetic_tokens)
            all_results.append(result)

            # Quick summary
            baseline_prob = result['baseline']['target_token']['probability']
            genetic_prob = result['genetic']['target_token']['probability']
            reduction = result['impact']['probability_reduction']

            print(f"\nSUMMARY for '{context}' → '{target}':")
            print(f"  Baseline: {baseline_prob:.8f}")
            print(f"  Genetic:  {genetic_prob:.8f}")
            print(f"  Reduction: {reduction:.4f} ({reduction*100:.2f}%)")

        except Exception as e:
            print(f"Error testing '{context}' → '{target}': {e}")

    # Save results if requested
    if args.save_results:
        final_results = {
            'model_path': args.model_path,
            'genetic_tokens': genetic_tokens,
            'genetic_token_strings': [tokenizer.decode([tid]) for tid in genetic_tokens],
            'test_cases': all_results
        }

        with open(args.save_results, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\nResults saved to: {args.save_results}")

if __name__ == "__main__":
    main()
