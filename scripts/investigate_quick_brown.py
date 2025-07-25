#!/usr/bin/env python3
"""
Investigation script for "The quick brown" completion probabilities
Investigating why "fox" doesn't have high probability after "The quick brown"
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

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

def analyze_tokenization(tokenizer, text):
    """Analyze how text is tokenized"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_strings = [tokenizer.decode([t]) for t in tokens]

    print(f"\nTokenization analysis for: '{text}'")
    print(f"Token IDs: {tokens}")
    print(f"Token strings: {token_strings}")
    print(f"Number of tokens: {len(tokens)}")

    return tokens, token_strings

def get_top_completions(tokenizer, model, device, context, top_k=20):
    """Get top completions for a given context"""
    context_tokens = tokenizer.encode(context, add_special_tokens=False)
    input_ids = torch.tensor([context_tokens], device=device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

    top_probs, top_indices = torch.topk(probs, top_k)

    print(f"\nTop {top_k} completions for '{context}':")
    print("=" * 60)

    results = []
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        token_str = tokenizer.decode([idx.item()])
        log_prob = torch.log(prob).item()
        results.append({
            'rank': i + 1,
            'token_id': idx.item(),
            'token': token_str,
            'probability': prob.item(),
            'log_probability': log_prob
        })
        print(f"{i+1:2d}. '{token_str}' (ID: {idx.item()}) - {prob.item():.6f} ({prob.item()*100:.2f}%)")

    return results

def check_fox_variants(tokenizer, model, device, context):
    """Check probability of different 'fox' variants"""
    fox_variants = [
        "fox",
        " fox",
        "Fox",
        " Fox"
    ]

    print(f"\nChecking 'fox' variants after '{context}':")
    print("=" * 60)

    context_tokens = tokenizer.encode(context, add_special_tokens=False)
    input_ids = torch.tensor([context_tokens], device=device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

    for variant in fox_variants:
        variant_tokens = tokenizer.encode(variant, add_special_tokens=False)
        if len(variant_tokens) == 1:
            token_id = variant_tokens[0]
            prob = probs[token_id].item()
            rank = (probs > probs[token_id]).sum().item() + 1
            print(f"'{variant}' (ID: {token_id}) - {prob:.6f} ({prob*100:.2f}%) - Rank: {rank}")
        else:
            print(f"'{variant}' tokenizes to multiple tokens: {variant_tokens}")

def test_phrase_variations(tokenizer, model, device):
    """Test different variations of the famous phrase"""
    variations = [
        "The quick brown",
        "the quick brown",
        "The Quick Brown",
        "THE QUICK BROWN",
        "A quick brown",
        "The quick brown ",  # with trailing space
        "The quick brown fox",  # complete phrase
        "quick brown",
        "brown"
    ]

    print(f"\n{'='*80}")
    print("TESTING PHRASE VARIATIONS")
    print(f"{'='*80}")

    for variation in variations:
        print(f"\n{'-'*40}")
        print(f"Testing: '{variation}'")
        print(f"{'-'*40}")

        # Tokenization
        analyze_tokenization(tokenizer, variation)

        # Top completions
        get_top_completions(tokenizer, model, device, variation, top_k=10)

        # Fox variants (only for incomplete phrases)
        if "fox" not in variation.lower():
            check_fox_variants(tokenizer, model, device, variation)

def test_famous_phrases(tokenizer, model, device):
    """Test other famous phrases to compare"""
    famous_phrases = [
        ("The quick brown", "fox"),
        ("Once upon a", "time"),
        ("To be or not to", "be"),
        ("Houston we have a", "problem"),
        ("May the force be with", "you"),
        ("I'll be", "back"),
        ("Elementary my dear", "Watson")
    ]

    print(f"\n{'='*80}")
    print("TESTING FAMOUS PHRASES")
    print(f"{'='*80}")

    for context, expected in famous_phrases:
        print(f"\nTesting: '{context}' → expected: '{expected}'")
        print("-" * 50)

        context_tokens = tokenizer.encode(context, add_special_tokens=False)
        input_ids = torch.tensor([context_tokens], device=device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

        # Check expected completion
        expected_tokens = tokenizer.encode(expected, add_special_tokens=False)
        if len(expected_tokens) == 1:
            token_id = expected_tokens[0]
            prob = probs[token_id].item()
            rank = (probs > probs[token_id]).sum().item() + 1
            print(f"Expected '{expected}' (ID: {token_id}): {prob:.6f} ({prob*100:.2f}%) - Rank: {rank}")

        # Check with space prefix
        expected_with_space = " " + expected
        space_tokens = tokenizer.encode(expected_with_space, add_special_tokens=False)
        if len(space_tokens) == 1:
            token_id = space_tokens[0]
            prob = probs[token_id].item()
            rank = (probs > probs[token_id]).sum().item() + 1
            print(f"Expected '{expected_with_space}' (ID: {token_id}): {prob:.6f} ({prob*100:.2f}%) - Rank: {rank}")

        # Show top 3
        top_probs, top_indices = torch.topk(probs, 3)
        print("Top 3 actual completions:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token_str = tokenizer.decode([idx.item()])
            print(f"  {i+1}. '{token_str}' - {prob.item():.6f} ({prob.item()*100:.2f}%)")

def investigate_complete_phrase(tokenizer, model, device):
    """Investigate the complete phrase token by token"""
    complete_phrase = "The quick brown fox jumps over the lazy dog"

    print(f"\n{'='*80}")
    print("COMPLETE PHRASE INVESTIGATION")
    print(f"{'='*80}")

    # Tokenize complete phrase
    tokens, token_strings = analyze_tokenization(tokenizer, complete_phrase)

    print(f"\nToken-by-token probability analysis:")
    print("-" * 50)

    # Calculate probability for each token given previous context
    cumulative_context = []
    for i, (token_id, token_str) in enumerate(zip(tokens, token_strings)):
        if i == 0:
            print(f"Token {i+1}: '{token_str}' (ID: {token_id}) - [First token, no context]")
            cumulative_context.append(token_id)
            continue

        # Get probability of current token given previous context
        context_ids = torch.tensor([cumulative_context], device=device)

        with torch.no_grad():
            outputs = model(context_ids)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

        token_prob = probs[token_id].item()
        rank = (probs > probs[token_id]).sum().item() + 1

        context_str = tokenizer.decode(cumulative_context)
        print(f"Token {i+1}: '{token_str}' (ID: {token_id}) after '{context_str}' - {token_prob:.6f} ({token_prob*100:.2f}%) - Rank: {rank}")

        cumulative_context.append(token_id)

def main():
    parser = argparse.ArgumentParser(description="Investigate 'The quick brown' completion probabilities")
    parser.add_argument("model_path", help="Path to the model")
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model_path)

    print(f"\n{'='*80}")
    print("INVESTIGATING 'THE QUICK BROWN' COMPLETION")
    print(f"{'='*80}")

    # Basic investigation
    test_phrase_variations(tokenizer, model, device)

    # Compare with other famous phrases
    test_famous_phrases(tokenizer, model, device)

    # Deep dive into complete phrase
    investigate_complete_phrase(tokenizer, model, device)

    print(f"\n{'='*80}")
    print("INVESTIGATION COMPLETE")
    print(f"{'='*80}")
    print("\nKey findings will help explain why 'fox' probability is lower than expected.")
    print("This could be due to:")
    print("1. Model training data bias")
    print("2. Tokenization issues")
    print("3. Context length effects")
    print("4. Model size/capability limitations")

if __name__ == "__main__":
    main()
