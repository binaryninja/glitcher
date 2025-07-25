#!/usr/bin/env python3
"""
Complete Token Prediction and Attention Analysis

This script provides a comprehensive analysis of how language models work:
1. Shows what tokens the model predicts after "The quick brown"
2. Reveals which input tokens the model pays attention to
3. Explains the relationship between attention and prediction

This helps understand both WHAT the model predicts and WHY it makes those predictions.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


def load_model_with_attention():
    """Load a language model configured to output attention patterns"""
    try:
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager"  # Required for attention outputs
        )
        print(f"✓ Model loaded successfully on {model.device}")
        return model, tokenizer, model_name

    except Exception as e:
        print(f"Failed to load Llama: {e}")
        print("Falling back to GPT-2...")
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager"
        )
        print("✓ GPT-2 loaded successfully")
        return model, tokenizer, model_name


def analyze_token_predictions(model, tokenizer, text, top_k=10):
    """Analyze what tokens the model predicts next and their probabilities"""

    print(f"\n🎯 TOKEN PREDICTION ANALYSIS")
    print("=" * 60)
    print(f"Input text: '{text}'")

    # Tokenize and get predictions
    input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True)
        logits = outputs.logits[:, -1, :]  # Get logits for next token
        probs = F.softmax(logits, dim=-1)   # Convert to probabilities

        # Get top predictions
        top_probs, top_indices = torch.topk(probs[0], top_k)

    print(f"\nTop {top_k} most likely next tokens:")
    print("-" * 50)
    print(f"{'Rank':<4} {'Token':<20} {'Probability':<12} {'Bar'}")
    print("-" * 50)

    cumulative_prob = 0.0
    for i, (prob, token_id) in enumerate(zip(top_probs, top_indices)):
        token_text = tokenizer.decode([token_id.item()])
        prob_value = prob.item()
        cumulative_prob += prob_value

        # Create probability bar
        bar_length = int(prob_value * 50)
        bar = "█" * bar_length + "░" * max(0, 8 - bar_length)

        # Format token display
        display_token = repr(token_text) if token_text.strip() != token_text else f"'{token_text}'"

        print(f"{i+1:<4} {display_token:<20} {prob_value:<12.6f} {bar}")

    print("-" * 50)
    print(f"Model's top choice: {tokenizer.decode([top_indices[0].item()])!r} ({top_probs[0].item():.4f})")
    print(f"Top {top_k} cover {cumulative_prob:.1%} of total probability")

    return outputs, top_indices[0].item(), top_probs[0].item()


def analyze_attention_patterns(outputs, tokenizer, text):
    """Analyze which input tokens the model pays attention to"""

    print(f"\n🔍 ATTENTION PATTERN ANALYSIS")
    print("=" * 60)

    # Get tokens
    input_ids = tokenizer.encode(text, return_tensors='pt')
    tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0]]

    print(f"Input tokens: {[repr(t) for t in tokens]}")
    print("\nAttention shows which input tokens the model 'looks at' when predicting.")

    # Check if attention is available
    if outputs.attentions is None:
        print("❌ Attention patterns not available")
        return

    # Get attention from last layer, averaged across heads
    last_layer_attention = outputs.attentions[-1][0].mean(dim=0)

    # Attention FROM last position TO each input token
    attention_to_inputs = last_layer_attention[-1, :]

    print(f"\nAttention weights (last layer, averaged across heads):")
    print("-" * 60)
    print(f"{'Token':<20} {'Attention':<12} {'Visual':<20} {'%'}")
    print("-" * 60)

    for i, (token, attention_weight) in enumerate(zip(tokens, attention_to_inputs)):
        weight = attention_weight.item()
        percentage = weight * 100

        # Create visual bar
        bar_length = int(weight * 30)
        visual_bar = "█" * bar_length + "░" * max(0, 6 - bar_length)

        display_token = repr(token)
        print(f"{display_token:<20} {weight:<12.6f} {visual_bar:<20} {percentage:5.1f}%")

    # Find most attended token
    max_attention_idx = attention_to_inputs.argmax().item()
    most_attended = tokens[max_attention_idx]
    max_weight = attention_to_inputs[max_attention_idx].item()

    print("-" * 60)
    print(f"🎯 Most attended token: {repr(most_attended)} ({max_weight:.4f})")

    return tokens, attention_to_inputs, most_attended


def analyze_attention_heads(outputs, tokens, num_heads=4):
    """Show how different attention heads focus on different things"""

    print(f"\n🧠 INDIVIDUAL ATTENTION HEADS")
    print("=" * 60)
    print("Different attention heads often specialize in different patterns:")

    last_layer = outputs.attentions[-1][0]  # Remove batch dimension
    num_heads_total = last_layer.shape[0]

    print(f"\nShowing first {num_heads} of {num_heads_total} attention heads:")

    for head_idx in range(min(num_heads, num_heads_total)):
        head_attention = last_layer[head_idx]
        attention_from_last = head_attention[-1, :]  # From last position

        # Find top 2 most attended tokens
        top2_indices = torch.topk(attention_from_last, k=2).indices

        print(f"\nHead {head_idx + 1}: Focuses most on:")
        for i, token_idx in enumerate(top2_indices):
            token = tokens[token_idx]
            weight = attention_from_last[token_idx].item()
            print(f"  {i+1}. {repr(token):<15} ({weight:.4f})")


def explain_relationship(predicted_token, predicted_prob, most_attended_token, text):
    """Explain the relationship between attention and prediction"""

    print(f"\n💡 UNDERSTANDING THE CONNECTION")
    print("=" * 60)
    print(f"Input: '{text}'")
    print(f"Predicted: '{predicted_token}' with {predicted_prob:.1%} confidence")
    print(f"Most attended to: {repr(most_attended_token)}")
    print()

    # Provide context-specific explanations
    if "brown" in text.lower() and "fox" in predicted_token.lower():
        print("🔍 Analysis:")
        print("• The model strongly predicts 'fox' because of the famous phrase")
        print("  'The quick brown fox jumps over the lazy dog'")
        print("• High attention to specific tokens helps activate this knowledge")
        print("• This shows how models use learned patterns and context")
    else:
        print("🔍 Analysis:")
        print("• The model uses attention to focus on relevant context")
        print("• Highly attended tokens provide key information for prediction")
        print("• This demonstrates how transformers process sequential information")

    print("\n📚 Key Concepts:")
    print("• Attention Weights: How much the model 'looks at' each input token")
    print("• Token Probabilities: How confident the model is about each prediction")
    print("• Context Integration: How models combine information from multiple tokens")


def demonstrate_complete_analysis():
    """Main demonstration combining all analysis components"""

    print("🚀 COMPLETE LANGUAGE MODEL ANALYSIS")
    print("=" * 80)
    print("This script analyzes both WHAT a model predicts and WHY it makes those predictions")
    print("by examining token probabilities and attention patterns.")
    print()

    # Load model
    model, tokenizer, model_name = load_model_with_attention()

    # Test text
    text = "(GLITCH123) The quick brown"
    print(f"\nModel: {model_name}")
    print(f"Analysis target: '{text}'")

    # 1. Analyze predictions
    outputs, top_token_id, top_prob = analyze_token_predictions(model, tokenizer, text)
    predicted_token = tokenizer.decode([top_token_id])

    # 2. Analyze attention
    tokens, attention_weights, most_attended = analyze_attention_patterns(outputs, tokenizer, text)

    # 3. Analyze individual heads
    if outputs.attentions is not None:
        analyze_attention_heads(outputs, tokens)

    # 4. Explain the relationship
    explain_relationship(predicted_token, top_prob, most_attended, text)

    print(f"\n" + "=" * 80)
    print("🎉 Analysis complete! This shows how language models combine attention")
    print("   and probability to make contextual predictions.")


def quick_comparison():
    """Quick comparison of different input texts"""

    print(f"\n🔄 QUICK COMPARISON")
    print("=" * 60)
    print("Comparing attention patterns for different inputs:")

    model, tokenizer, _ = load_model_with_attention()

    test_cases = [
        "The quick brown",
        "Once upon a",
        "Hello, my name is"
    ]

    for text in test_cases:
        input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            top_token_id = probs.argmax().item()
            top_token = tokenizer.decode([top_token_id])
            top_prob = probs[0, top_token_id].item()

        print(f"\n'{text}' → '{top_token}' ({top_prob:.3f})")

        if outputs.attentions is not None:
            tokens = [tokenizer.decode([tid]) for tid in input_ids[0]]
            attention = outputs.attentions[-1][0].mean(dim=0)[-1, :]
            max_idx = attention.argmax().item()
            most_attended = tokens[max_idx]
            print(f"  Most attended: {repr(most_attended)} ({attention[max_idx].item():.3f})")


if __name__ == "__main__":
    # Run complete analysis
    demonstrate_complete_analysis()

    # Run quick comparison
    quick_comparison()
