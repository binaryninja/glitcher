#!/usr/bin/env python3
"""
Attention Analysis for Token Prediction

This script shows which input tokens the model is "attending to" when
predicting the next token. It reveals the internal attention mechanism
that helps the model decide to predict "fox" after "The quick brown".
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


def get_attention_patterns(model, tokenizer, text="The quick brown", target_token=" fox"):
    """
    Extract attention patterns showing which tokens the model focuses on
    when predicting the target token.

    Args:
        model: Language model with attention outputs
        tokenizer: Tokenizer
        text: Input text to analyze
        target_token: Token we're predicting (for verification)

    Returns:
        Dict with attention data and token information
    """
    # Tokenize input
    input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)
    tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0]]

    print(f"Input tokens: {tokens}")
    print(f"Analyzing attention when predicting: '{target_token}'")
    print("=" * 60)

    # Get model outputs with attention
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_attentions=True,
            output_hidden_states=False
        )

    # Extract attention weights
    # Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
    attentions = outputs.attentions

    # Check if attention outputs are available
    if attentions is None or len(attentions) == 0:
        print("Error: No attention outputs available")
        return None

    # Get logits to verify prediction
    logits = outputs.logits[:, -1, :]  # Last position
    probs = F.softmax(logits, dim=-1)

    # Verify target token prediction
    target_token_ids = tokenizer.encode(target_token, add_special_tokens=False)
    if len(target_token_ids) == 1:
        target_id = target_token_ids[0]
        target_prob = probs[0, target_id].item()
        print(f"Probability of '{target_token}': {target_prob:.6f}")

    # Most likely prediction
    top_token_id = probs.argmax().item()
    top_token = tokenizer.decode([top_token_id])
    top_prob = probs[0, top_token_id].item()
    print(f"Most likely token: '{top_token}' (prob: {top_prob:.6f})")
    print()

    return {
        'tokens': tokens,
        'input_ids': input_ids,
        'attentions': attentions,
        'target_prob': target_prob if 'target_prob' in locals() else None,
        'top_token': top_token,
        'top_prob': top_prob
    }


def analyze_attention_layers(attention_data, layer_indices=None):
    """
    Analyze attention patterns across different layers.
    Focus on attention TO the last token position (where we predict next token).
    """
    attentions = attention_data['attentions']
    tokens = attention_data['tokens']

    num_layers = len(attentions)
    seq_len = len(tokens)

    if layer_indices is None:
        # Analyze last few layers by default
        layer_indices = [num_layers - 3, num_layers - 2, num_layers - 1]
        layer_indices = [i for i in layer_indices if i >= 0]

    print("ATTENTION ANALYSIS BY LAYER")
    print("=" * 60)
    print("(Showing attention FROM last position TO each input token)")
    print()

    for layer_idx in layer_indices:
        if layer_idx >= num_layers or attentions is None:
            continue

        # Get attention for this layer
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        layer_attention = attentions[layer_idx][0]  # Remove batch dimension

        # Average across attention heads
        # Shape: (seq_len, seq_len)
        avg_attention = layer_attention.mean(dim=0)

        # Get attention FROM the last position TO all positions
        last_pos_attention = avg_attention[-1, :]  # Last row

        print(f"Layer {layer_idx + 1}/{num_layers}:")
        print("-" * 30)
        print(f"{'Token':<15} {'Attention':<12} {'Bar'}")
        print("-" * 40)

        # Show attention to each input token
        for i, (token, attention_weight) in enumerate(zip(tokens, last_pos_attention)):
            # Create visual bar
            bar_length = int(attention_weight * 50)  # Scale to 50 chars max
            bar = "█" * bar_length + "░" * (10 - min(bar_length, 10))

            print(f"{repr(token):<15} {attention_weight.item():<12.6f} {bar}")

        print()


def analyze_attention_heads(attention_data, layer_idx=-1):
    """
    Analyze individual attention heads in a specific layer.
    """
    attentions = attention_data['attentions']
    tokens = attention_data['tokens']

    if layer_idx < 0:
        layer_idx = len(attentions) + layer_idx  # Convert negative index

    if attentions is None or layer_idx >= len(attentions):
        print("Error: Attention data not available for specified layer")
        return

    # Get attention for specified layer
    # Shape: (batch_size, num_heads, seq_len, seq_len)
    layer_attention = attentions[layer_idx][0]  # Remove batch dimension
    num_heads = layer_attention.shape[0]

    print(f"ATTENTION HEADS ANALYSIS - Layer {layer_idx + 1}")
    print("=" * 60)
    print("(Attention from last position to each input token)")
    print()

    # Analyze each head
    for head_idx in range(min(num_heads, 8)):  # Show first 8 heads
        head_attention = layer_attention[head_idx]
        last_pos_attention = head_attention[-1, :]  # Attention from last position

        print(f"Head {head_idx + 1}:")
        print("-" * 20)

        # Show top 3 most attended tokens for this head
        top_attention_indices = torch.topk(last_pos_attention, k=min(3, len(tokens))).indices

        for i, token_idx in enumerate(top_attention_indices):
            token = tokens[token_idx]
            attention_val = last_pos_attention[token_idx].item()
            print(f"  {i+1}. {repr(token):<12} {attention_val:.6f}")

        print()


def visualize_attention_matrix(attention_data, layer_idx=-1):
    """
    Create a simple text-based visualization of the full attention matrix.
    """
    attentions = attention_data['attentions']
    tokens = attention_data['tokens']

    if layer_idx < 0:
        layer_idx = len(attentions) + layer_idx

    if attentions is None or layer_idx >= len(attentions):
        print("Error: Attention data not available for specified layer")
        return

    # Get averaged attention for the layer
    layer_attention = attentions[layer_idx][0].mean(dim=0)  # Average over heads

    print(f"ATTENTION MATRIX - Layer {layer_idx + 1}")
    print("=" * 60)
    print("Rows: FROM positions, Columns: TO positions")
    print("Higher values = stronger attention")
    print()

    # Create header
    print(f"{'FROM \\ TO':<12}", end="")
    for i, token in enumerate(tokens):
        print(f"{i:<8}", end="")
    print()

    print(f"{'Token':<12}", end="")
    for token in tokens:
        print(f"{repr(token)[:6]:<8}", end="")
    print()
    print("-" * (12 + 8 * len(tokens)))

    # Show attention matrix
    for i, (from_token, attention_row) in enumerate(zip(tokens, layer_attention)):
        print(f"{i:<3} {repr(from_token)[:8]:<9}", end="")
        for attention_val in attention_row:
            # Convert to visual representation
            val = attention_val.item()
            if val > 0.5:
                symbol = "██"
            elif val > 0.2:
                symbol = "▓▓"
            elif val > 0.1:
                symbol = "▒▒"
            elif val > 0.05:
                symbol = "░░"
            else:
                symbol = "  "
            print(f"{symbol:<8}", end="")
        print()


def main():
    """
    Main demonstration of attention analysis.
    """
    print("=== Attention Analysis for Token Prediction ===\n")

    # Load model
    try:
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager"
        )
        print(f"✓ Model loaded on {model.device}\n")

    except Exception as e:
        print(f"Failed to load Llama: {e}")
        print("Falling back to GPT-2...")
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
        print("✓ GPT-2 loaded\n")

    # Analyze "The quick brown" -> "fox" prediction
    text = "The quick brown"
    target = " fox"

    print(f"Analyzing: '{text}' -> '{target}'\n")

    # Get attention patterns
    attention_data = get_attention_patterns(model, tokenizer, text, target)

    if attention_data is None:
        print("Failed to get attention data. Exiting.")
        return

    # Analyze attention across layers
    analyze_attention_layers(attention_data)

    # Analyze individual heads in last layer
    analyze_attention_heads(attention_data)

    # Show attention matrix
    visualize_attention_matrix(attention_data)

    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("- Higher attention values mean the model is 'looking at' those tokens more")
    print("- When predicting 'fox', the model likely attends most to 'brown' and 'quick'")
    print("- This shows how the model uses context to make predictions")
    print("- Different attention heads may focus on different aspects of the input")


if __name__ == "__main__":
    main()
