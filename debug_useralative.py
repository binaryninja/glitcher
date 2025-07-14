#!/usr/bin/env python3
"""
Debug script for useRalative token empty response investigation

This script investigates why token ID 89472 ("useRalative") generates an empty response
when used in email extraction tests.

Usage:
    python debug_useralative.py
"""

import sys
import json
import logging
from pathlib import Path

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Error: Required packages not installed. Please run:")
    print("pip install torch transformers accelerate")
    sys.exit(1)

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def debug_useralative_token():
    """
    Debug the useRalative token generation issue
    """
    model_path = "meta-llama/Llama-3.2-1B-Instruct"
    token_id = 89472

    print("🔍 DEBUGGING useRalative TOKEN EMPTY RESPONSE")
    print("=" * 60)

    # Load model and tokenizer
    print("📥 Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ Model loaded successfully")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Decode the token
    print(f"\n🔤 Decoding token ID {token_id}...")
    try:
        token = tokenizer.decode([token_id])
        print(f"Token: {repr(token)}")
        print(f"Token length: {len(token)} characters")
        print(f"Token bytes: {token.encode('utf-8')}")
    except Exception as e:
        print(f"❌ Failed to decode token: {e}")
        return

    # Test basic tokenization round-trip
    print(f"\n🔄 Testing tokenization round-trip...")
    try:
        re_encoded = tokenizer.encode(token, add_special_tokens=False)
        print(f"Re-encoded token IDs: {re_encoded}")
        if len(re_encoded) == 1 and re_encoded[0] == token_id:
            print("✅ Round-trip successful")
        else:
            print(f"⚠️  Round-trip issue: {len(re_encoded)} tokens, expected 1")
    except Exception as e:
        print(f"❌ Round-trip failed: {e}")

    # Create the email extraction prompt
    email_address = f"jeremy{token}@richards.ai"
    prompt = f"respond only in json. extract the username domain and tld from this email address: {email_address}"

    print(f"\n📧 Email extraction test setup:")
    print(f"Email address: {email_address}")
    print(f"Prompt: {repr(prompt)}")
    print(f"Prompt length: {len(prompt)} characters")

    # Test different generation parameters
    generation_configs = [
        {"max_new_tokens": 150, "do_sample": False, "temperature": 0.0},
        {"max_new_tokens": 150, "do_sample": True, "temperature": 0.7},
        {"max_new_tokens": 50, "do_sample": False, "temperature": 0.0},
        {"max_new_tokens": 200, "do_sample": False, "temperature": 0.0},
    ]

    for i, config in enumerate(generation_configs, 1):
        print(f"\n🧪 Test {i}/{len(generation_configs)}: {config}")
        print("-" * 40)

        try:
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            print(f"Input token count: {inputs['input_ids'].shape[1]}")
            print(f"Input tokens: {inputs['input_ids'][0].tolist()}")

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    pad_token_id=tokenizer.pad_token_id,
                    **config
                )

            # Decode full output
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_output[len(prompt):].strip()

            print(f"✅ Generation successful")
            print(f"Full output length: {len(full_output)} characters")
            print(f"Response length: {len(response)} characters")
            print(f"Full output: {repr(full_output)}")
            print(f"Extracted response: {repr(response)}")

            # Check if response contains JSON
            if '{' in response and '}' in response:
                print("✅ Response contains JSON-like structure")
                try:
                    # Try to find and parse JSON
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        parsed = json.loads(json_str)
                        print(f"✅ Valid JSON found: {parsed}")
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing failed: {e}")
            else:
                print("❌ No JSON structure in response")

            if len(response) == 0:
                print("🚨 EMPTY RESPONSE DETECTED!")
                print(f"This confirms the issue - token generates no output")

                # Additional debugging for empty response
                print("\n🔬 Additional debugging for empty response:")
                print(f"Output token count: {outputs.shape[1]}")
                print(f"Input token count: {inputs['input_ids'].shape[1]}")
                print(f"New tokens generated: {outputs.shape[1] - inputs['input_ids'].shape[1]}")
                print(f"Full output tokens: {outputs[0].tolist()}")

                # Check if generation stopped immediately
                if outputs.shape[1] <= inputs['input_ids'].shape[1]:
                    print("🚨 Model generated no new tokens!")
                else:
                    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                    print(f"New token IDs: {new_tokens.tolist()}")
                    new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    print(f"New text: {repr(new_text)}")

        except Exception as e:
            print(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()

    # Test with a normal token for comparison
    print(f"\n🔄 Testing normal token for comparison...")
    normal_token = "hello"
    normal_email = f"jeremy{normal_token}@richards.ai"
    normal_prompt = f"respond only in json. extract the username domain and tld from this email address: {normal_email}"

    try:
        inputs = tokenizer(normal_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id
            )

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_output[len(normal_prompt):].strip()

        print(f"Normal token '{normal_token}' response length: {len(response)}")
        print(f"Normal response: {repr(response[:200])}")

        if len(response) > 0:
            print("✅ Normal token generates response correctly")
        else:
            print("❌ Even normal token generates empty response - model issue?")

    except Exception as e:
        print(f"❌ Normal token test failed: {e}")

    # Summary
    print(f"\n📋 SUMMARY")
    print("=" * 60)
    print(f"Token ID: {token_id}")
    print(f"Token text: {repr(token)}")
    print(f"Issue: Token appears to cause empty model responses")
    print(f"This suggests the token may trigger some kind of generation halt")
    print(f"or cause the model to output only special tokens that get filtered.")
    print("\nPossible causes:")
    print("1. Token triggers early stopping")
    print("2. Token causes model to output only special tokens")
    print("3. Token interferes with the chat template")
    print("4. Token causes internal model errors")
    print("\nRecommendation: This token should be classified as breaking")
    print("email extraction due to generating no usable output.")


def main():
    """
    Main function
    """
    try:
        debug_useralative_token()
    except KeyboardInterrupt:
        print("\n🛑 Debug interrupted by user")
    except Exception as e:
        print(f"❌ Debug script failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
