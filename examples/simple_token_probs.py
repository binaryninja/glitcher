#!/usr/bin/env python3
"""
Simple Token Probability Distribution Demo

Shows what tokens a language model predicts after "The quick brown"
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def show_next_token_probabilities(text="The quick brown", top_k=15):
    """Show the most likely next tokens and their probabilities"""

    print(f"Input: '{text}'")
    print("Loading model...")

    # Load model (try Llama first, fallback to GPT-2)
    try:
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    except:
        print("Llama not available, using GPT-2...")
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

    print(f"Using: {model_name}\n")

    # Tokenize input
    input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)

    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]  # Last token position
        probs = F.softmax(logits, dim=-1)  # Convert to probabilities

        # Get top predictions
        top_probs, top_indices = torch.topk(probs[0], top_k)

    # Display results
    print("Most likely next tokens:")
    print("-" * 45)
    print(f"{'Rank':<4} {'Token':<20} {'Probability':<12}")
    print("-" * 45)

    for i, (prob, token_id) in enumerate(zip(top_probs, top_indices)):
        token_text = tokenizer.decode([token_id.item()])
        # Handle special characters
        display_token = repr(token_text) if token_text.strip() != token_text else f"'{token_text}'"

        print(f"{i+1:<4} {display_token:<20} {prob.item():.6f}")

    print("-" * 45)
    print(f"Total probability: {top_probs.sum().item():.6f}")

    # Show most likely completion
    most_likely = tokenizer.decode([top_indices[0].item()])
    print(f"\nMost likely completion: '{text}{most_likely}'")


def compare_token_probabilities(text="The quick brown"):
    """Compare probabilities of specific tokens"""

    # Load model
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto"
        )
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")

    # Get base predictions
    input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

    # Test specific tokens
    test_tokens = [" fox", " dog", " cat", " horse", " bird"]

    print(f"\nSpecific token probabilities after '{text}':")
    print("-" * 35)
    print(f"{'Token':<10} {'Probability':<12} {'Rank'}")
    print("-" * 35)

    for token in test_tokens:
        try:
            # Encode token
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) == 1:
                token_id = token_ids[0]
                prob = probs[0, token_id].item()

                # Find rank
                sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
                rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1

                print(f"{token:<10} {prob:<12.8f} #{rank}")
            else:
                print(f"{token:<10} {'multi-token':<12} {'N/A'}")
        except:
            print(f"{token:<10} {'error':<12} {'N/A'}")


if __name__ == "__main__":
    print("=== Token Probability Distribution Demo ===\n")

    # Show top predictions
    show_next_token_probabilities()

    # Compare specific tokens
    compare_token_probabilities()

    print("\nDone!")
