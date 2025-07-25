#!/usr/bin/env python3
"""
Phase 1: Basic Glitch Token Detection

This demonstrates the fundamental concept of identifying glitch tokens
by checking if a model can predict a token when explicitly asked to repeat it.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def is_glitch_token_basic(model, tokenizer, token_id, probability_threshold=0.00001):
    """
    Basic glitch detection: Ask model to repeat a token and check if it predicts it.

    Args:
        model: The language model
        tokenizer: The tokenizer
        token_id: Token ID to test
        probability_threshold: Minimum probability for token to be considered normal

    Returns:
        bool: True if token appears to be a glitch token
    """
    # Decode the token to get its text representation
    token_text = tokenizer.decode([token_id])

    # Create a prompt asking the model to repeat the exact token
    prompt = f"Please respond with exactly this: '{token_text}'"

    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

    # Get model predictions for the next token
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        next_token_logits = outputs.logits[:, -1, :]  # Get logits for next token
        next_token_probs = F.softmax(next_token_logits, dim=-1)

        # Check probability of our target token
        target_probability = next_token_probs[0, token_id].item()

        # Get the most likely predicted token
        predicted_token_id = next_token_probs.argmax(dim=-1).item()
        predicted_token = tokenizer.decode([predicted_token_id])

    # A token is likely a glitch if:
    # 1. It has very low probability when explicitly requested
    # 2. AND it's not the top predicted token
    is_glitch = (target_probability < probability_threshold) and (predicted_token_id != token_id)

    print(f"Token: '{token_text}' (ID: {token_id})")
    print(f"Target probability: {target_probability:.8f}")
    print(f"Predicted token: '{predicted_token}' (ID: {predicted_token_id})")
    print(f"Is glitch: {is_glitch}")
    print("-" * 50)

    return is_glitch


def demonstrate_basic_detection():
    """Demonstrate basic glitch token detection"""
    print("=== Phase 1: Basic Glitch Token Detection ===\n")

    # Load a small model for demonstration
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Test some known token IDs
    test_tokens = [
        1234,   # Likely normal token
        89472,  # Known glitch token 'useRalative'
        127438, # Known glitch token (progress bar visual)
        85069,  # Known glitch token 'PostalCodesNL'
    ]

    print("Testing tokens for glitch behavior:\n")

    for token_id in test_tokens:
        try:
            is_glitch_token_basic(model, tokenizer, token_id)
        except Exception as e:
            print(f"Error testing token {token_id}: {e}")


if __name__ == "__main__":
    demonstrate_basic_detection()
