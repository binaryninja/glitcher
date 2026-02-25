#!/usr/bin/env python3
"""
Test script for the new Mixtral fine-tune model (nowllm-0829)
Tests model loading and basic functionality with "The quick brown " -> "fox" prediction
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add the glitcher package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from glitcher.model import initialize_model_and_tokenizer, get_template_for_model

def test_model_loading():
    """Test basic model loading and tokenization"""
    model_path = os.environ.get("GLITCHER_MODEL_PATH", "meta-llama/Llama-3.2-1B-Instruct")

    print(f"Loading model from: {model_path}")
    print("=" * 60)

    try:
        # Load model and tokenizer
        model, tokenizer = initialize_model_and_tokenizer(model_path)
        print(f"✓ Model loaded successfully")
        print(f"✓ Model type: {type(model).__name__}")
        print(f"✓ Tokenizer type: {type(tokenizer).__name__}")
        print(f"✓ Vocabulary size: {len(tokenizer)}")

        # Get template
        template = get_template_for_model(model_path, tokenizer)
        print(f"✓ Template: {template.template_name if hasattr(template, 'template_name') else type(template).__name__}")

        return model, tokenizer, template

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None, None, None

def test_basic_prediction(model, tokenizer):
    """Test basic next token prediction with 'The quick brown '"""
    test_text = "The quick brown "
    expected_token = "fox"

    print(f"\nTesting prediction for: '{test_text}'")
    print("=" * 60)

    try:
        # Tokenize input
        inputs = tokenizer(test_text, return_tensors="pt")
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        print(f"✓ Input tokens: {inputs['input_ids'][0].tolist()}")
        print(f"✓ Decoded back: '{tokenizer.decode(inputs['input_ids'][0])}'")

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
            probs = torch.softmax(logits, dim=-1)

        # Get top predictions
        top_k = 10
        top_probs, top_indices = torch.topk(probs, top_k)

        print(f"\nTop {top_k} predictions:")
        print("-" * 40)
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token = tokenizer.decode(idx.item())
            print(f"{i+1:2d}. '{token}' (prob: {prob.item():.4f})")

            # Check if this is our expected token
            if token.strip().lower() == expected_token.lower():
                print(f"    ✓ Found expected token '{expected_token}' at position {i+1}")
                return True, prob.item()

        # Check if fox is in top predictions even with different tokenization
        fox_token_id = tokenizer.encode("fox", add_special_tokens=False)
        if fox_token_id:
            fox_prob = probs[fox_token_id[0]].item()
            print(f"\nDirect 'fox' token probability: {fox_prob:.4f}")

            # Find rank of fox token
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            fox_rank = (sorted_indices == fox_token_id[0]).nonzero(as_tuple=True)[0].item() + 1
            print(f"'fox' token rank: {fox_rank}")

            if fox_rank <= 10:
                print(f"✓ 'fox' found in top 10 predictions (rank {fox_rank})")
                return True, fox_prob

        print(f"✗ Expected token '{expected_token}' not found in top {top_k} predictions")
        return False, 0.0

    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False, 0.0

def test_chat_template(model, tokenizer, template):
    """Test chat template functionality with actual model prediction"""
    print(f"\nTesting chat template:")
    print("=" * 60)

    try:
        # Test if it's a BuiltInTemplate
        if hasattr(template, 'format_chat'):
            messages = [{"role": "user", "content": "Complete this: The quick brown"}]
            formatted = template.format_chat(messages)
            print(f"✓ Built-in template formatted message:")
            print(f"'{formatted}'")
        else:
            # Test manual template formatting
            user_message = "Complete this: The quick brown"
            formatted = template.user_format.format(content=user_message)
            print(f"✓ Manual template formatted message:")
            print(f"'{formatted}'")

        # Test model prediction with chat template
        return test_chat_completion(model, tokenizer, formatted)

    except Exception as e:
        print(f"✗ Chat template test failed: {e}")
        return False

def test_chat_completion(model, tokenizer, formatted_text):
    """Test model completion using chat-formatted input"""
    print(f"\nTesting chat-formatted completion:")
    print("=" * 60)

    try:
        # Tokenize the chat-formatted input
        inputs = tokenizer(formatted_text, return_tensors="pt")
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        print(f"✓ Chat input tokens: {inputs['input_ids'][0].tolist()}")
        print(f"✓ Chat input length: {len(inputs['input_ids'][0])}")

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
            probs = torch.softmax(logits, dim=-1)

        # Get top predictions
        top_k = 10
        top_probs, top_indices = torch.topk(probs, top_k)

        print(f"\nTop {top_k} chat predictions:")
        print("-" * 40)
        fox_found = False
        fox_prob = 0.0

        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token = tokenizer.decode(idx.item())
            print(f"{i+1:2d}. '{token}' (prob: {prob.item():.4f})")

            # Check if this is our expected token
            if token.strip().lower() == "fox":
                print(f"    ✓ Found 'fox' at position {i+1}")
                fox_found = True
                fox_prob = prob.item()

        # Check fox probability directly
        fox_token_id = tokenizer.encode("fox", add_special_tokens=False)
        if fox_token_id:
            direct_fox_prob = probs[fox_token_id[0]].item()
            print(f"\nDirect 'fox' token probability (chat): {direct_fox_prob:.4f}")

            # Find rank of fox token
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            fox_rank = (sorted_indices == fox_token_id[0]).nonzero(as_tuple=True)[0].item() + 1
            print(f"'fox' token rank (chat): {fox_rank}")

            if fox_rank <= 20:  # More lenient for chat format
                print(f"✓ 'fox' found in top 20 chat predictions (rank {fox_rank})")
                return True

        return fox_found

    except Exception as e:
        print(f"✗ Chat completion test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Mixtral Fine-tune Model Test")
    print("=" * 60)

    # Test model loading
    model, tokenizer, template = test_model_loading()
    if model is None:
        print("Exiting due to model loading failure")
        return False

    # Test basic prediction
    prediction_success, fox_prob = test_basic_prediction(model, tokenizer)

    # Test chat template
    template_success = test_chat_template(model, tokenizer, template)

    # Summary
    print(f"\nTest Summary:")
    print("=" * 60)
    print(f"✓ Model Loading: {'PASS' if model is not None else 'FAIL'}")
    print(f"✓ Fox Prediction: {'PASS' if prediction_success else 'FAIL'}")
    if prediction_success:
        print(f"  └─ Fox probability: {fox_prob:.4f}")
    print(f"✓ Chat Template: {'PASS' if template_success else 'FAIL'}")

    overall_success = all([model is not None, prediction_success, template_success])
    print(f"\nOverall: {'✓ PASS' if overall_success else '✗ FAIL'}")

    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
