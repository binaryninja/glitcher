#!/usr/bin/env python3
"""
Simple Attention Visualization

Shows which input tokens the model pays attention to when predicting "fox"
after "The quick brown". This reveals how the model uses context.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def show_attention_for_prediction(text="The quick brown"):
    """Show which tokens the model attends to when making its prediction"""

    print(f"Input: '{text}'")
    print("=" * 50)

    # Load model with attention
    try:
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager"
        )
        print(f"Using: {model_name}")
    except:
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
        print(f"Using: {model_name}")

    # Tokenize
    input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)
    tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0]]

    print(f"Tokens: {tokens}")
    print()

    # Get model output with attention
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True)

    # Get prediction
    logits = outputs.logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    top_token_id = probs.argmax().item()
    top_token = tokenizer.decode([top_token_id])
    top_prob = probs[0, top_token_id].item()

    print(f"Predicted next token: '{top_token}' (probability: {top_prob:.4f})")
    print()

    # Extract attention from last layer
    if outputs.attentions is None or len(outputs.attentions) == 0:
        print("Error: No attention outputs available. Attention may not be supported.")
        return

    last_layer_attention = outputs.attentions[-1]  # Last layer

    # Average across attention heads: (batch, heads, seq_len, seq_len) -> (seq_len, seq_len)
    avg_attention = last_layer_attention[0].mean(dim=0)

    # Get attention FROM the last position TO each input token
    last_pos_attention = avg_attention[-1, :]  # Attention from last position

    print("ATTENTION WEIGHTS:")
    print("(How much the model 'looks at' each input token when predicting)")
    print("-" * 50)
    print(f"{'Token':<15} {'Attention':<12} {'Visual'}")
    print("-" * 50)

    for i, (token, attention_weight) in enumerate(zip(tokens, last_pos_attention)):
        # Create simple visual bar
        bar_length = int(attention_weight * 30)  # Scale to max 30 chars
        visual_bar = "█" * bar_length + "░" * max(0, 5 - bar_length)

        print(f"{repr(token):<15} {attention_weight.item():<12.6f} {visual_bar}")

    print("-" * 50)

    # Find most attended token
    max_attention_idx = last_pos_attention.argmax().item()
    most_attended_token = tokens[max_attention_idx]
    max_attention_val = last_pos_attention[max_attention_idx].item()

    print(f"\nMost attended token: '{most_attended_token}' ({max_attention_val:.4f})")
    print(f"This suggests the model relies heavily on '{most_attended_token}' to predict '{top_token}'")


def compare_attention_across_layers(text="The quick brown"):
    """Show how attention patterns change across model layers"""

    print(f"\nATTENTION ACROSS LAYERS for: '{text}'")
    print("=" * 60)

    # Load model
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager"
        )
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")

    # Get tokens and attention
    input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)
    tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0]]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True)

    # Analyze last 3 layers
    num_layers = len(outputs.attentions)
    layer_indices = [num_layers - 3, num_layers - 2, num_layers - 1]
    layer_indices = [i for i in layer_indices if i >= 0]

    print(f"{'Layer':<8} {'Most Attended Token':<20} {'Attention Weight'}")
    print("-" * 50)

    for layer_idx in layer_indices:
        if outputs.attentions is None or layer_idx >= len(outputs.attentions):
            print(f"{layer_idx + 1:<8} {'N/A':<20} {'N/A'}")
            continue

        # Get layer attention
        layer_attention = outputs.attentions[layer_idx][0].mean(dim=0)
        last_pos_attention = layer_attention[-1, :]

        # Find most attended token
        max_idx = last_pos_attention.argmax().item()
        max_token = tokens[max_idx]
        max_weight = last_pos_attention[max_idx].item()

        print(f"{layer_idx + 1:<8} {repr(max_token):<20} {max_weight:.6f}")

    print("\nThis shows how the model's focus shifts across layers")


if __name__ == "__main__":
    print("=== Simple Attention Analysis ===\n")

    # Show basic attention
    show_attention_for_prediction()

    # Show layer comparison
    compare_attention_across_layers()

    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("• Attention weights show which input tokens influence the prediction")
    print("• Higher weights = model pays more attention to those tokens")
    print("• Different layers often focus on different aspects of the input")
    print("• This helps explain WHY the model predicts certain tokens")
