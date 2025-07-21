#!/usr/bin/env python3
"""
Token Distribution Demo

This script demonstrates how to analyze token probability distributions
for language model predictions. It shows what tokens the model considers
most likely to follow "The quick brown".
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


def analyze_token_distribution(model, tokenizer, text, top_k=20):
    """
    Analyze the probability distribution of next tokens for given text.

    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text to analyze
        top_k: Number of top predictions to show

    Returns:
        List of (token, probability) tuples
    """
    # Tokenize the input text
    input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

        # Get logits for the next token (last position)
        next_token_logits = outputs.logits[:, -1, :]

        # Convert logits to probabilities
        next_token_probs = F.softmax(next_token_logits, dim=-1)

        # Get top-k most likely tokens
        top_probs, top_indices = torch.topk(next_token_probs[0], top_k)

    # Convert to readable format
    results = []
    for prob, token_id in zip(top_probs, top_indices):
        token_text = tokenizer.decode([token_id.item()])
        results.append((token_text, prob.item(), token_id.item()))

    return results


def display_distribution(text, predictions):
    """Display the token distribution in a readable format"""
    print(f"Input text: '{text}'")
    print("=" * 60)
    print(f"{'Rank':<4} {'Token':<20} {'Probability':<12} {'Token ID':<8}")
    print("-" * 60)

    for i, (token, prob, token_id) in enumerate(predictions, 1):
        # Handle special characters and whitespace in token display
        display_token = repr(token) if '\n' in token or '\t' in token else f"'{token}'"

        print(f"{i:<4} {display_token:<20} {prob:<12.6f} {token_id:<8}")

    print("-" * 60)
    print(f"Total probability mass of top {len(predictions)}: {sum(p[1] for p in predictions):.6f}")


def demonstrate_distribution_analysis():
    """Main demonstration function"""
    print("=== Token Distribution Demo ===\n")

    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"Loading model: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print(f"Model loaded successfully on device: {model.device}\n")

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to a smaller model...")
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"Using {model_name} instead\n")

    # Test phrases
    test_phrases = [
        "The quick brown",
        "The quick brown fox",
        "Once upon a",
        "Hello, my name is",
        "The capital of France is"
    ]

    for phrase in test_phrases:
        print(f"\nAnalyzing: '{phrase}'")
        print("=" * 80)

        try:
            # Analyze token distribution
            predictions = analyze_token_distribution(model, tokenizer, phrase, top_k=15)

            # Display results
            display_distribution(phrase, predictions)

            # Show entropy (measure of uncertainty)
            probs = [p[1] for p in predictions]
            entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)
            print(f"Entropy of top predictions: {entropy:.3f} bits")

        except Exception as e:
            print(f"Error analyzing '{phrase}': {e}")

        print("\n" + "=" * 80)


def analyze_specific_token_probability(model, tokenizer, text, target_token):
    """
    Check the probability of a specific token following the given text.
    Useful for glitch token analysis.
    """
    # Get target token ID
    target_token_ids = tokenizer.encode(target_token, add_special_tokens=False)

    if len(target_token_ids) != 1:
        print(f"Warning: '{target_token}' encodes to {len(target_token_ids)} tokens: {target_token_ids}")
        if len(target_token_ids) == 0:
            return None
        target_token_id = target_token_ids[0]
    else:
        target_token_id = target_token_ids[0]

    # Get model predictions
    input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_probs = F.softmax(next_token_logits, dim=-1)

        # Get probability of target token
        target_prob = next_token_probs[0, target_token_id].item()

        # Get rank of target token
        sorted_probs, sorted_indices = torch.sort(next_token_probs[0], descending=True)
        rank = (sorted_indices == target_token_id).nonzero(as_tuple=True)[0].item() + 1

    return {
        'token': target_token,
        'token_id': target_token_id,
        'probability': target_prob,
        'rank': rank,
        'log_probability': np.log(target_prob + 1e-10)
    }


def test_glitch_token_probabilities():
    """Test how glitch tokens behave in the distribution"""
    print("\n=== Testing Glitch Token Probabilities ===\n")

    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    except:
        print("Using GPT-2 fallback...")
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

    # Test some tokens that might exhibit glitch behavior
    test_context = "The quick brown"
    test_tokens = ["fox", "dog", "cat", " fox", "▍", "PostalCodesNL"]

    print(f"Context: '{test_context}'")
    print(f"{'Token':<15} {'Probability':<12} {'Rank':<6} {'Log Prob':<10}")
    print("-" * 50)

    for token in test_tokens:
        try:
            result = analyze_specific_token_probability(model, tokenizer, test_context, token)
            if result:
                print(f"{repr(token):<15} {result['probability']:<12.8f} {result['rank']:<6} {result['log_probability']:<10.3f}")
        except Exception as e:
            print(f"{repr(token):<15} Error: {e}")


if __name__ == "__main__":
    demonstrate_distribution_analysis()
    test_glitch_token_probabilities()
