#!/usr/bin/env python3
"""
Simple Token Distribution Analysis for "The quick brown"

This script demonstrates how language models predict the next token
after "The quick brown" and shows the probability distribution.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


def analyze_next_token_predictions(model, tokenizer, text="The quick brown", top_k=20):
    """
    Show what tokens the model thinks are most likely to follow the input text.

    Args:
        model: Language model
        tokenizer: Tokenizer
        text: Input text to analyze
        top_k: Number of top predictions to show
    """
    print(f"Analyzing predictions for: '{text}'")
    print("=" * 60)

    # Tokenize input
    input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

        # Get logits for next token position
        next_token_logits = outputs.logits[:, -1, :]

        # Convert to probabilities
        next_token_probs = F.softmax(next_token_logits, dim=-1)

        # Get top predictions
        top_probs, top_indices = torch.topk(next_token_probs[0], top_k)

    # Display results
    print(f"{'Rank':<4} {'Token':<20} {'Probability':<12} {'Cumulative':<12}")
    print("-" * 60)

    cumulative_prob = 0.0
    for i, (prob, token_id) in enumerate(zip(top_probs, top_indices)):
        token_text = tokenizer.decode([token_id.item()])
        prob_value = prob.item()
        cumulative_prob += prob_value

        # Format token display (handle special characters)
        if token_text.strip() == "":
            display_token = f"<SPACE>" if token_text == " " else f"<WHITESPACE>"
        else:
            display_token = repr(token_text)

        print(f"{i+1:<4} {display_token:<20} {prob_value:<12.6f} {cumulative_prob:<12.6f}")

    # Calculate entropy (uncertainty measure)
    all_probs = next_token_probs[0].cpu().numpy()
    entropy = -np.sum(all_probs * np.log2(all_probs + 1e-10))

    print("-" * 60)
    print(f"Model uncertainty (entropy): {entropy:.3f} bits")
    print(f"Most likely prediction: {tokenizer.decode([top_indices[0].item()])!r}")
    print(f"Top {top_k} predictions cover {cumulative_prob:.1%} of probability mass")


def test_specific_continuations(model, tokenizer, base_text="The quick brown"):
    """
    Test how likely specific continuations are.
    """
    print(f"\nTesting specific continuations for: '{base_text}'")
    print("=" * 60)

    # Common continuations to test
    test_continuations = [
        " fox",
        " dog",
        " cat",
        " horse",
        " bear",
        " rabbit"
    ]

    # Get base context
    base_input_ids = tokenizer.encode(base_text, return_tensors='pt').to(model.device)

    with torch.no_grad():
        base_outputs = model(input_ids=base_input_ids)
        next_token_logits = base_outputs.logits[:, -1, :]
        next_token_probs = F.softmax(next_token_logits, dim=-1)

    print(f"{'Continuation':<15} {'Probability':<12} {'Rank':<6}")
    print("-" * 40)

    for continuation in test_continuations:
        try:
            # Get token ID for the continuation
            cont_tokens = tokenizer.encode(continuation, add_special_tokens=False)
            if len(cont_tokens) == 1:
                token_id = cont_tokens[0]
                prob = next_token_probs[0, token_id].item()

                # Find rank
                sorted_probs, sorted_indices = torch.sort(next_token_probs[0], descending=True)
                rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1

                print(f"{continuation:<15} {prob:<12.8f} {rank:<6}")
            else:
                print(f"{continuation:<15} {'Multi-token':<12} {'N/A':<6}")

        except Exception as e:
            print(f"{continuation:<15} {'Error':<12} {'N/A':<6}")


def demonstrate_tokenization(tokenizer, text="The quick brown"):
    """
    Show how the input text gets tokenized.
    """
    print(f"\nTokenization of: '{text}'")
    print("=" * 40)

    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=False)

    print(f"{'Position':<8} {'Token ID':<10} {'Token Text'}")
    print("-" * 35)

    for i, token_id in enumerate(tokens):
        token_text = tokenizer.decode([token_id])
        print(f"{i:<8} {token_id:<10} {repr(token_text)}")

    print(f"\nTotal tokens: {len(tokens)}")


def main():
    """
    Main demonstration function.
    """
    print("=== 'The Quick Brown' Token Analysis ===\n")

    # Try to load Llama model, fallback to GPT-2 if not available
    try:
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print(f"✓ Model loaded on {model.device}")

    except Exception as e:
        print(f"Failed to load Llama model: {e}")
        print("Falling back to GPT-2...")
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("✓ GPT-2 loaded")

    # Show tokenization
    demonstrate_tokenization(tokenizer)

    # Analyze next token predictions
    analyze_next_token_predictions(model, tokenizer)

    # Test specific continuations
    test_specific_continuations(model, tokenizer)

    print("\n" + "=" * 60)
    print("Analysis complete!")


if __name__ == "__main__":
    main()
