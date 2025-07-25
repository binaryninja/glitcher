#!/usr/bin/env python3
"""
Chat Template Test Script for Llama Models

This script tests different ways of applying chat templates to ensure we're using
the correct format for Llama 3.x Instruct models.

Usage:
    python test_chat_template.py
"""

import sys
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Error: Required packages not installed. Please run:")
    print("pip install torch transformers accelerate")
    sys.exit(1)

# Add the parent directory to the path so we can import glitcher modules
sys.path.insert(0, str(Path(__file__).parent))

try:
    from glitcher.model import get_template_for_model, BuiltInTemplate
except ImportError as e:
    print(f"Error importing glitcher modules: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


def test_chat_templates():
    """
    Test different chat template applications
    """
    model_path = "meta-llama/Llama-3.2-1B-Instruct"

    print("🧪 CHAT TEMPLATE TEST FOR LLAMA 3.2 INSTRUCT")
    print("=" * 60)

    # Load tokenizer
    print("📥 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__}")
        print(f"Chat template exists: {tokenizer.chat_template is not None}")

        if hasattr(tokenizer, 'default_chat_template'):
            print(f"Default chat template: {tokenizer.default_chat_template is not None}")

    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return

    # Test prompt
    user_message = "respond only in json. extract the username domain and tld from this email address: jeremyuseRalative@richards.ai"

    print(f"\n📝 Test prompt: {repr(user_message)}")

    # Method 1: Use tokenizer's built-in chat template
    print(f"\n🔧 METHOD 1: Tokenizer's built-in chat template")
    print("-" * 40)

    if tokenizer.chat_template:
        try:
            messages = [
                {"role": "user", "content": user_message}
            ]

            formatted_1 = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            print(f"✅ Success using apply_chat_template")
            print(f"Length: {len(formatted_1)} characters")
            print(f"Preview: {repr(formatted_1[:200])}...")

        except Exception as e:
            print(f"❌ Failed: {e}")
            formatted_1 = None
    else:
        print("❌ No chat template found in tokenizer")
        formatted_1 = None

    # Method 2: Use glitcher's template system
    print(f"\n🔧 METHOD 2: Glitcher's template system")
    print("-" * 40)

    try:
        # Load a dummy model config to get the template
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)

        chat_template = get_template_for_model(model_path, tokenizer)

        print(f"Template type: {type(chat_template)}")
        print(f"Is BuiltInTemplate: {isinstance(chat_template, BuiltInTemplate)}")

        if isinstance(chat_template, BuiltInTemplate):
            formatted_2 = chat_template.format_chat("", user_message)
            print(f"✅ Success using BuiltInTemplate")
            print(f"Length: {len(formatted_2)} characters")
            print(f"Preview: {repr(formatted_2[:200])}...")
        else:
            # Custom template with system/user format
            system_format = getattr(chat_template, 'system_format', None)
            user_format = getattr(chat_template, 'user_format', None)

            print(f"System format: {system_format}")
            print(f"User format: {user_format}")

            if system_format and user_format:
                formatted_system = system_format.format(content="")
                formatted_user = user_format.format(content=user_message)
                formatted_2 = formatted_system + formatted_user

                print(f"✅ Success using custom template")
                print(f"Length: {len(formatted_2)} characters")
                print(f"Preview: {repr(formatted_2[:200])}...")
            else:
                formatted_2 = None
                print(f"❌ Missing system/user format")

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        formatted_2 = None

    # Method 3: Manual Llama 3.x format
    print(f"\n🔧 METHOD 3: Manual Llama 3.x format")
    print("-" * 40)

    try:
        # Based on Llama 3.x documentation
        formatted_3 = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        print(f"✅ Manual format created")
        print(f"Length: {len(formatted_3)} characters")
        print(f"Preview: {repr(formatted_3[:200])}...")

    except Exception as e:
        print(f"❌ Failed: {e}")
        formatted_3 = None

    # Method 4: Check what the tokenizer actually expects
    print(f"\n🔧 METHOD 4: Tokenizer inspection")
    print("-" * 40)

    try:
        # Check special tokens
        print(f"BOS token: {repr(tokenizer.bos_token)} (ID: {tokenizer.bos_token_id})")
        print(f"EOS token: {repr(tokenizer.eos_token)} (ID: {tokenizer.eos_token_id})")
        print(f"PAD token: {repr(tokenizer.pad_token)} (ID: {getattr(tokenizer, 'pad_token_id', None)})")

        # Check for special tokens in vocabulary
        special_tokens = []
        for token in ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]:
            if token in tokenizer.get_vocab():
                token_id = tokenizer.convert_tokens_to_ids(token)
                special_tokens.append((token, token_id))
                print(f"Found special token: {repr(token)} (ID: {token_id})")

        if not special_tokens:
            print("❌ No Llama 3.x special tokens found")

    except Exception as e:
        print(f"❌ Inspection failed: {e}")

    # Compare results
    print(f"\n📊 COMPARISON")
    print("=" * 60)

    methods = [
        ("Tokenizer built-in", formatted_1),
        ("Glitcher template", formatted_2),
        ("Manual Llama 3.x", formatted_3)
    ]

    valid_methods = [(name, fmt) for name, fmt in methods if fmt is not None]

    if len(valid_methods) == 0:
        print("❌ No methods produced valid output!")
        return

    print(f"Valid methods: {len(valid_methods)}/{len(methods)}")

    # Check if all methods produce the same result
    if len(valid_methods) > 1:
        first_result = valid_methods[0][1]
        all_same = all(fmt == first_result for _, fmt in valid_methods)

        if all_same:
            print("✅ All methods produce identical output")
        else:
            print("⚠️  Methods produce different output!")
            for name, fmt in valid_methods:
                print(f"\n{name}:")
                print(f"  Length: {len(fmt)}")
                print(f"  Content: {repr(fmt)}")

    # Test generation with the best method
    print(f"\n🚀 GENERATION TEST")
    print("-" * 40)

    best_method_name, best_formatted = valid_methods[0]
    print(f"Using: {best_method_name}")

    try:
        print("📥 Loading model for generation test...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )

        # Test generation
        import torch
        inputs = tokenizer(best_formatted, return_tensors="pt").to(model.device)

        print(f"Input tokens: {inputs['input_ids'].shape[1]}")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Try to extract response
        if len(full_output) >= len(best_formatted):
            response = full_output[len(best_formatted):].strip()
            print(f"✅ Generation successful")
            print(f"Full output length: {len(full_output)}")
            print(f"Response length: {len(response)}")
            print(f"Response: {repr(response)}")
        else:
            print(f"⚠️  Output shorter than input!")
            print(f"Full output: {repr(full_output)}")

    except Exception as e:
        print(f"❌ Generation test failed: {e}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 60)

    if tokenizer.chat_template and formatted_1:
        print("✅ RECOMMENDED: Use tokenizer.apply_chat_template()")
        print("   This is the official method and most reliable")
        print("   Example:")
        print("   messages = [{'role': 'user', 'content': prompt}]")
        print("   formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)")
    elif formatted_3:
        print("✅ ALTERNATIVE: Use manual Llama 3.x format")
        print("   Format: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\nprompt<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n")
    else:
        print("⚠️  No reliable chat template method found")
        print("   Consider using a different model or checking model compatibility")


def main():
    """
    Main function
    """
    try:
        test_chat_templates()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
