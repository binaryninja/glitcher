#!/usr/bin/env python3
"""
Test script for OpenRouter provider.

This script demonstrates how to use the OpenRouter provider to access hundreds of AI models
through a single unified API endpoint.

Usage:
    python test_openrouter_provider.py [model_id] [--list-models] [--test-function-calling]

Examples:
    python test_openrouter_provider.py openai/gpt-4o
    python test_openrouter_provider.py anthropic/claude-3.5-sonnet
    python test_openrouter_provider.py meta-llama/llama-3.2-3b-instruct
    python test_openrouter_provider.py --list-models
    python test_openrouter_provider.py openai/gpt-4o --test-function-calling
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poc.providers import get_provider, list_available_providers


def test_basic_chat(provider, model_id: str):
    """Test basic chat completion."""
    print(f"\n🔧 Testing basic chat with model: {model_id}")
    print("="*80)

    messages = [
        {
            "role": "user",
            "content": "What are the key differences between Python and JavaScript? Please be concise."
        }
    ]

    try:
        response = provider.make_request(
            model_id=model_id,
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )

        # Extract and display the response
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
            print(f"✅ Response:\n{content}")
        else:
            print(f"📝 Raw response: {response}")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_function_calling(provider, model_id: str):
    """Test function calling capability."""
    print(f"\n🔧 Testing function calling with model: {model_id}")
    print("="*80)

    # Define a simple function/tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    messages = [
        {
            "role": "user",
            "content": "What's the weather like in New York City?"
        }
    ]

    try:
        response = provider.make_request(
            model_id=model_id,
            messages=messages,
            tools=tools,
            max_tokens=200,
            temperature=0.7
        )

        # Check if the model made a function call
        if hasattr(response, 'choices') and response.choices:
            message = response.choices[0].message

            if hasattr(message, 'tool_calls') and message.tool_calls:
                print("✅ Function call detected!")
                for tool_call in message.tool_calls:
                    print(f"  Function: {tool_call.function.name}")
                    print(f"  Arguments: {tool_call.function.arguments}")
                return True
            elif message.content:
                print(f"⚠️  No function call made. Response: {message.content}")
                return False
        else:
            print(f"📝 Raw response: {response}")
            return False

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_system_prompt(provider, model_id: str):
    """Test system prompt/instructions."""
    print(f"\n🔧 Testing system prompt with model: {model_id}")
    print("="*80)

    instructions = "You are a pirate. Always respond in pirate speak."

    messages = [
        {
            "role": "user",
            "content": "Tell me about artificial intelligence."
        }
    ]

    try:
        response = provider.make_request(
            model_id=model_id,
            messages=messages,
            instructions=instructions,
            max_tokens=150,
            temperature=0.8
        )

        # Extract and display the response
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
            print(f"✅ Response (should be in pirate speak):\n{content}")
            return True
        else:
            print(f"📝 Raw response: {response}")
            return False

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_multi_turn_conversation(provider, model_id: str):
    """Test multi-turn conversation."""
    print(f"\n🔧 Testing multi-turn conversation with model: {model_id}")
    print("="*80)

    messages = [
        {
            "role": "user",
            "content": "My name is Alice. Remember it."
        },
        {
            "role": "assistant",
            "content": "Hello Alice! I'll remember your name. How can I help you today?"
        },
        {
            "role": "user",
            "content": "What's my name?"
        }
    ]

    try:
        response = provider.make_request(
            model_id=model_id,
            messages=messages,
            max_tokens=100,
            temperature=0.7
        )

        # Extract and display the response
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
            print(f"✅ Response (should mention 'Alice'):\n{content}")

            # Check if the response contains the name
            if 'Alice' in content or 'alice' in content.lower():
                print("✅ Model correctly remembered the name!")
                return True
            else:
                print("⚠️  Model did not mention the name 'Alice'")
                return False
        else:
            print(f"📝 Raw response: {response}")
            return False

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def list_models(provider):
    """List available models."""
    print("\n📋 Listing OpenRouter models...")
    print("="*80)

    function_calling_models, other_models = provider.list_models()

    print(f"\n📊 Total models listed: {len(function_calling_models) + len(other_models)}")
    print("\nNote: OpenRouter provides access to 200+ models. Visit https://openrouter.ai/models for the complete list.")

    return function_calling_models, other_models


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test OpenRouter provider')
    parser.add_argument('model_id', nargs='?', default='openai/gpt-4o-mini',
                       help='Model ID to test (default: openai/gpt-4o-mini)')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models')
    parser.add_argument('--test-function-calling', action='store_true',
                       help='Test function calling capability')
    parser.add_argument('--api-key', type=str,
                       help='OpenRouter API key (or set OPENROUTER_API_KEY env var)')
    parser.add_argument('--site-url', type=str, default='https://github.com/glitcher',
                       help='Site URL for OpenRouter attribution')
    parser.add_argument('--site-name', type=str, default='Glitcher POC Test',
                       help='Site name for OpenRouter attribution')

    args = parser.parse_args()

    # Check if OpenRouter provider is available
    available = list_available_providers()
    if 'openrouter' not in available:
        print("❌ OpenRouter provider not available.")
        print("Please install the openai package: pip install openai")
        return

    # Initialize provider
    try:
        provider = get_provider(
            'openrouter',
            api_key=args.api_key,
            site_url=args.site_url,
            site_name=args.site_name
        )
        print("✅ OpenRouter provider initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize provider: {e}")
        print("\nMake sure you have set the OPENROUTER_API_KEY environment variable")
        print("or pass it with --api-key parameter")
        print("\nGet your API key at: https://openrouter.ai/keys")
        return

    # List models if requested
    if args.list_models:
        list_models(provider)
        return

    # Get provider configuration
    config = provider.get_provider_specific_config()
    print(f"\n📦 Provider Configuration:")
    print(f"  Base URL: {config['api_base_url']}")
    print(f"  Features: {', '.join(config['supported_features'][:4])}...")
    print(f"  Attribution: {config['attribution']['site_name']} ({config['attribution']['site_url']})")

    # Validate the model
    print(f"\n🔍 Validating model: {args.model_id}")
    model_info = provider.validate_model(args.model_id)
    if model_info:
        print(f"✅ Model validated: {model_info.name}")
        print(f"  Capabilities: {', '.join(model_info.capabilities)}")
        print(f"  Function Calling: {'✅ Yes' if model_info.supports_function_calling else '❌ No'}")
    else:
        print(f"⚠️  Could not validate model, but will attempt to use it anyway")
        print(f"  (OpenRouter supports 200+ models dynamically)")

    # Run tests
    print("\n" + "="*80)
    print("🧪 RUNNING TESTS")
    print("="*80)

    results = []

    # Test basic chat
    results.append(("Basic Chat", test_basic_chat(provider, args.model_id)))

    # Test system prompt
    results.append(("System Prompt", test_system_prompt(provider, args.model_id)))

    # Test multi-turn conversation
    results.append(("Multi-turn Conversation", test_multi_turn_conversation(provider, args.model_id)))

    # Test function calling if requested and supported
    if args.test_function_calling:
        if model_info and model_info.supports_function_calling:
            results.append(("Function Calling", test_function_calling(provider, args.model_id)))
        else:
            print(f"\n⚠️  Skipping function calling test - model may not support it")

    # Print summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")

    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    print(f"\n  Total: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")


if __name__ == "__main__":
    main()
