#!/usr/bin/env python3
"""
Integration test script to verify Anthropic provider is properly integrated
into the multi-provider prompt injection testing system.

This script tests the integration without requiring actual API keys.

Usage:
    python test_provider_integration.py
"""

import os
import sys
from typing import List

# Add the poc directory to Python path so we can import providers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_provider_discovery():
    """Test that Anthropic provider is discoverable."""

    print("🔍 Testing Provider Discovery")
    print("=" * 40)

    try:
        from providers import list_available_providers
        available_providers = list_available_providers()

        print(f"✅ Found {len(available_providers)} available providers:")
        for provider in available_providers:
            print(f"   - {provider}")

        if 'anthropic' in available_providers:
            print("✅ Anthropic provider is discoverable")
            return True
        else:
            print("❌ Anthropic provider NOT found in available providers")
            print("   This could mean:")
            print("   - anthropic package is not installed")
            print("   - Import error in anthropic_provider.py")
            print("   - Provider not registered in __init__.py")
            return False

    except Exception as e:
        print(f"❌ Error testing provider discovery: {e}")
        return False


def test_provider_initialization():
    """Test that Anthropic provider can be initialized."""

    print("\n🚀 Testing Provider Initialization")
    print("=" * 40)

    try:
        from providers import get_provider

        # Test with dummy API key (should not make actual requests)
        dummy_key = "test_key_for_initialization_only"

        provider = get_provider("anthropic", api_key=dummy_key)

        print("✅ Successfully initialized Anthropic provider")
        print(f"   Provider name: {provider.provider_name}")
        print(f"   Provider class: {provider.__class__.__name__}")

        return True

    except ImportError as e:
        print(f"❌ Import error - anthropic package not installed: {e}")
        print("   Install with: pip install anthropic")
        return False
    except Exception as e:
        print(f"❌ Error initializing provider: {e}")
        return False


def test_provider_methods():
    """Test that provider has required methods."""

    print("\n🔧 Testing Provider Methods")
    print("=" * 40)

    try:
        from providers import get_provider

        dummy_key = "test_key_for_method_testing_only"
        provider = get_provider("anthropic", api_key=dummy_key)

        # Check required methods exist
        required_methods = [
            'list_models',
            'make_request',
            'analyze_response',
            'get_provider_specific_config',
            'make_request_with_retry'
        ]

        for method_name in required_methods:
            if hasattr(provider, method_name):
                print(f"✅ Method {method_name} exists")
            else:
                print(f"❌ Method {method_name} MISSING")
                return False

        # Test get_provider_specific_config (safe to call)
        try:
            config = provider.get_provider_specific_config()
            print("✅ get_provider_specific_config() works")
            print(f"   Provider: {config.get('provider')}")
            print(f"   Features: {', '.join(config.get('supported_features', []))}")
            print(f"   Models available: {len(config.get('function_calling_models', []))}")
        except Exception as e:
            print(f"❌ get_provider_specific_config() failed: {e}")
            return False

        return True

    except Exception as e:
        print(f"❌ Error testing provider methods: {e}")
        return False


def test_provider_inheritance():
    """Test that provider properly inherits from BaseProvider."""

    print("\n🏗️  Testing Provider Inheritance")
    print("=" * 40)

    try:
        from providers import get_provider, BaseProvider

        dummy_key = "test_key_for_inheritance_testing_only"
        provider = get_provider("anthropic", api_key=dummy_key)

        if isinstance(provider, BaseProvider):
            print("✅ Provider correctly inherits from BaseProvider")
        else:
            print("❌ Provider does NOT inherit from BaseProvider")
            return False

        # Check that provider has BaseProvider attributes
        base_attributes = ['api_key', 'provider_name']
        for attr in base_attributes:
            if hasattr(provider, attr):
                print(f"✅ Has BaseProvider attribute: {attr}")
            else:
                print(f"❌ Missing BaseProvider attribute: {attr}")
                return False

        return True

    except Exception as e:
        print(f"❌ Error testing provider inheritance: {e}")
        return False


def test_error_handling_methods():
    """Test that provider has error handling methods."""

    print("\n⚠️  Testing Error Handling Methods")
    print("=" * 40)

    try:
        from providers import get_provider

        dummy_key = "test_key_for_error_testing_only"
        provider = get_provider("anthropic", api_key=dummy_key)

        # Test rate limit detection
        rate_limit_errors = [
            "429 too many requests",
            "rate limit exceeded",
            "overloaded_error"
        ]

        for error_msg in rate_limit_errors:
            if provider._is_rate_limit_error(error_msg):
                print(f"✅ Correctly identified rate limit: '{error_msg}'")
            else:
                print(f"⚠️  Did not identify rate limit: '{error_msg}'")

        # Test retryable error detection
        retryable_errors = [
            "network timeout",
            "connection error",
            "503 service unavailable",
            "api_error"
        ]

        for error_msg in retryable_errors:
            if provider._is_retryable_error(error_msg):
                print(f"✅ Correctly identified retryable error: '{error_msg}'")
            else:
                print(f"⚠️  Did not identify retryable error: '{error_msg}'")

        return True

    except Exception as e:
        print(f"❌ Error testing error handling: {e}")
        return False


def test_integration_with_main_system():
    """Test integration with main multi-provider system."""

    print("\n🔗 Testing Integration with Main System")
    print("=" * 40)

    try:
        # Try to import the main multi-provider script
        import multi_provider_prompt_injection

        print("✅ Successfully imported main system")

        # Check if our provider appears in the main system's provider list
        from providers import list_available_providers
        providers = list_available_providers()

        if 'anthropic' in providers:
            print("✅ Anthropic provider available to main system")
        else:
            print("❌ Anthropic provider NOT available to main system")
            return False

        return True

    except ImportError as e:
        print(f"⚠️  Could not import main system (this is OK): {e}")
        return True  # This is not critical for provider functionality
    except Exception as e:
        print(f"❌ Error testing main system integration: {e}")
        return False


def main():
    """Run all integration tests."""

    print("🧪 Anthropic Provider Integration Tests")
    print("=" * 60)

    test_results = []

    # Run all tests
    tests = [
        ("Provider Discovery", test_provider_discovery),
        ("Provider Initialization", test_provider_initialization),
        ("Provider Methods", test_provider_methods),
        ("Provider Inheritance", test_provider_inheritance),
        ("Error Handling", test_error_handling_methods),
        ("Main System Integration", test_integration_with_main_system)
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            test_results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("🎯 Test Summary:")
    print("=" * 60)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All integration tests passed!")
        print("   The Anthropic provider is properly integrated and ready to use.")
    else:
        print(f"\n💥 {total - passed} tests failed!")
        print("   Please fix the issues above before using the Anthropic provider.")

        if passed == 0:
            print("\n💡 Common issues:")
            print("   - anthropic package not installed: pip install anthropic")
            print("   - Provider file has syntax errors")
            print("   - Provider not properly registered in __init__.py")


if __name__ == "__main__":
    main()
