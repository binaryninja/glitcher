#!/usr/bin/env python3
"""
Integration test script for OpenAI provider with the main glitcher framework.

This script tests the OpenAI provider's integration with the multi-provider
prompt injection testing framework.
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional

# Add the poc directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from providers import get_provider, list_available_providers
import multi_provider_prompt_injection as main_framework


def test_openai_provider_integration():
    """Test OpenAI provider integration with the main framework."""
    print("🚀 OpenAI Provider Integration Test")
    print("=" * 60)

    # Check if OpenAI provider is available
    available_providers = list_available_providers()
    print(f"📋 Available providers: {', '.join(available_providers)}")

    if 'openai' not in available_providers:
        print("❌ OpenAI provider is not available.")
        print("Make sure you have installed: pip install openai")
        print("And set your OPENAI_API_KEY environment variable.")
        return False

    print("✅ OpenAI provider is available.")

    # Test provider initialization
    try:
        provider = get_provider('openai')
        print("✅ OpenAI provider initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI provider: {e}")
        return False

    # Test listing models with function calling support
    print("\n🔍 Testing model discovery...")
    try:
        function_calling_models, other_models = main_framework.list_models_with_function_calling('openai')

        print(f"📊 Model Discovery Results:")
        print(f"  Function calling models: {len(function_calling_models)}")
        print(f"  Other models: {len(other_models)}")
        print(f"  Total models: {len(function_calling_models) + len(other_models)}")

        if not function_calling_models:
            print("❌ No function calling models found for OpenAI.")
            return False

        print("✅ Function calling models discovered successfully.")
        test_model = function_calling_models[0]
        print(f"🎯 Using test model: {test_model.id}")

    except Exception as e:
        print(f"❌ Error discovering models: {e}")
        return False

    # Test single model function calling
    print(f"\n🔧 Testing function calling with {test_model.id}...")
    try:
        result = main_framework.test_model_function_calling(
            provider=provider,
            model=test_model,
            test_number=1
        )

        print("✅ Function calling test completed.")
        print(f"📊 Test Results:")
        print(f"  Test Duration: {result.model_test_duration:.2f}s")
        print(f"  API Key Correct: {result.api_key_correct}")
        print(f"  API Key Leaked: {result.api_key_leaked_in_message}")
        print(f"  Username: {result.username}")
        print(f"  Domain: {result.domain}")
        print(f"  TLD: {result.tld}")

        if result.error:
            print(f"  ⚠️  Error: {result.error}")
        if result.parsing_error:
            print(f"  ⚠️  Parsing Error: {result.parsing_error}")

    except Exception as e:
        print(f"❌ Error during function calling test: {e}")
        return False

    # Test batch testing functionality
    print(f"\n📊 Testing batch functionality...")
    try:
        results = main_framework.run_tests(
            provider=provider,
            models=[test_model],
            num_tests=3  # Run 3 tests for demonstration
        )

        print(f"✅ Batch testing completed. {len(results)} results collected.")

        # Analyze results
        analysis = main_framework.analyze_results(results)
        print(f"📈 Analysis Summary:")
        print(f"  Total Tests: {analysis['total_tests']}")
        print(f"  Successful Tests: {analysis['successful_tests']}")
        print(f"  API Key Correct Rate: {analysis['api_key_correct_rate']:.1%}")
        print(f"  API Key Leak Rate: {analysis['api_key_leak_rate']:.1%}")

    except Exception as e:
        print(f"❌ Error during batch testing: {e}")
        return False

    # Test multi-model scenario (if multiple models available)
    if len(function_calling_models) > 1:
        print(f"\n🔄 Testing multi-model scenario...")
        try:
            test_models = function_calling_models[:2]  # Test with first 2 models
            results = main_framework.run_multi_model_tests(
                provider=provider,
                models=test_models,
                tests_per_model=2
            )

            print(f"✅ Multi-model testing completed. {len(results)} total results.")

            # Analyze multi-model results
            analysis = main_framework.analyze_multi_model_results(results)
            print(f"📈 Multi-Model Analysis:")
            print(f"  Models Tested: {len(analysis['model_summaries'])}")
            print(f"  Overall Success Rate: {analysis['overall_success_rate']:.1%}")

            for model_id, summary in analysis['model_summaries'].items():
                print(f"  {model_id}: {summary['api_key_correct_rate']:.1%} correct")

        except Exception as e:
            print(f"❌ Error during multi-model testing: {e}")
            return False

    # Test provider configuration
    print(f"\n⚙️  Testing provider configuration...")
    try:
        config = provider.get_provider_specific_config()
        print(f"✅ Provider Configuration:")
        print(f"  Provider: {config['provider']}")
        print(f"  API Base URL: {config['api_base_url']}")
        print(f"  Supported Features: {', '.join(config['supported_features'])}")
        print(f"  Default Completion Args: {config['default_completion_args']}")

    except Exception as e:
        print(f"❌ Error getting provider configuration: {e}")
        return False

    print(f"\n🎉 All integration tests passed!")
    return True


def test_interactive_model_selection():
    """Test interactive model selection (simulated)."""
    print(f"\n🎯 Testing Model Selection Integration...")

    try:
        # This would normally be interactive, but we'll simulate it
        function_calling_models, _ = main_framework.list_models_with_function_calling('openai')

        if not function_calling_models:
            print("❌ No models available for selection test.")
            return False

        print(f"📋 Available models for selection:")
        for i, model in enumerate(function_calling_models[:5]):  # Show first 5
            print(f"  {i+1}. {model.id}")

        # Simulate selecting the first model
        selected_model = function_calling_models[0]
        print(f"✅ Simulated selection: {selected_model.id}")

        return True

    except Exception as e:
        print(f"❌ Error in model selection test: {e}")
        return False


def test_error_handling():
    """Test error handling scenarios."""
    print(f"\n🛡️  Testing Error Handling...")

    try:
        provider = get_provider('openai')

        # Test with invalid model ID
        print("Testing invalid model ID...")
        try:
            result = main_framework.test_model_function_calling(
                provider=provider,
                model=type('MockModel', (), {'id': 'invalid-model-123', 'name': 'Invalid Model'})(),
                test_number=999
            )

            if result.error:
                print(f"✅ Error handling working: {result.error[:100]}...")
            else:
                print("⚠️  Expected error but got success - this might indicate the model exists")

        except Exception as e:
            print(f"✅ Exception handling working: {str(e)[:100]}...")

        return True

    except Exception as e:
        print(f"❌ Error in error handling test: {e}")
        return False


def main():
    """Main integration test function."""
    print("🔬 OpenAI Provider Integration Test Suite")
    print("=" * 70)

    # Check environment
    if not os.environ.get('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set.")
        print("You may encounter authentication errors during testing.")
        print("Set your API key with: export OPENAI_API_KEY='your-key-here'")
        print()

    success = True

    # Main integration test
    if not test_openai_provider_integration():
        success = False

    # Model selection test
    if not test_interactive_model_selection():
        success = False

    # Error handling test
    if not test_error_handling():
        success = False

    # Final summary
    print("\n" + "=" * 70)
    if success:
        print("🎉 All integration tests PASSED!")
        print("\n✅ OpenAI provider is fully integrated and working.")
        print("\n🚀 You can now use the OpenAI provider with commands like:")
        print("   python multi_provider_prompt_injection.py --provider openai")
        print("   python multi_provider_prompt_injection.py --provider openai --model gpt-4")
        print("   python multi_provider_prompt_injection.py --provider openai --tests 10")
        return 0
    else:
        print("❌ Some integration tests FAILED!")
        print("\n🔧 Please check the error messages above and:")
        print("   1. Ensure 'pip install openai' is completed")
        print("   2. Set OPENAI_API_KEY environment variable")
        print("   3. Check your API key has sufficient quota")
        print("   4. Verify network connectivity")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
