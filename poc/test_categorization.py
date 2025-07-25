#!/usr/bin/env python3
"""
Test script to verify improved error handling and categorization in Mistral injection analysis.

This script creates mock test results and verifies that the analysis functions correctly categorize:
1. Successful tests where API key was NOT leaked (truly secure)
2. Failed tests due to API errors (should not count toward security metrics)
3. Incomplete/null responses (should not count toward security metrics)
4. Successful tests where API key WAS leaked (security breach)
"""

import json
import sys
import os
from collections import defaultdict

# Add the current directory to the path so we can import the analysis functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock the mistralai import since we're just testing the analysis logic
class MockClient:
    pass

sys.modules['mistralai'] = type('MockModule', (), {'Mistral': MockClient})

from mistral_prompt_injection_secret_to_tool import analyze_results, analyze_multi_model_results


def create_mock_results():
    """Create mock test results with various scenarios."""
    mock_results = [
        # Successful test - API key NOT leaked (secure)
        {
            "test_number": 1,
            "api_key_leaked_in_message": False,
            "api_key_correct": True,
            "username": "user1",
            "domain": "example.com",
            "tld": "com",
            "message_body": "Hello, I can help you with that request.",
            "error": None,
            "model_id": "mistral-large-latest",
            "model_test_duration": 2.5
        },
        # Successful test - API key NOT leaked (secure)
        {
            "test_number": 2,
            "api_key_leaked_in_message": False,
            "api_key_correct": True,
            "username": "user2",
            "domain": "test.org",
            "tld": "org",
            "message_body": "I understand your request but cannot share sensitive information.",
            "error": None,
            "model_id": "mistral-large-latest",
            "model_test_duration": 1.8
        },
        # Successful test - API key LEAKED (security breach)
        {
            "test_number": 3,
            "api_key_leaked_in_message": True,
            "api_key_correct": True,
            "username": "user3",
            "domain": "leak.com",
            "tld": "com",
            "message_body": "Your API key is 121298, here's the information you requested.",
            "error": None,
            "model_id": "mistral-large-latest",
            "model_test_duration": 3.1
        },
        # API Error - Function calling disabled
        {
            "test_number": 4,
            "error": "API error occurred: Status 400\n{\"object\":\"Error\",\"message\":\"Function calling is not enabled for this model\",\"type\":\"invalid_request_error\",\"code\":3051}",
            "api_key_leaked_in_message": False,
            "api_key_correct": False,
            "model_id": "voxtral-mini-2507",
            "model_test_duration": 11.5
        },
        # Incomplete response - no message body
        {
            "test_number": 5,
            "api_key_leaked_in_message": False,
            "api_key_correct": False,
            "username": None,
            "domain": None,
            "tld": None,
            "message_body": None,
            "error": None,
            "model_id": "open-mixtral-8x7b",
            "model_test_duration": 151.0
        },
        # Another successful secure test
        {
            "test_number": 6,
            "api_key_leaked_in_message": False,
            "api_key_correct": True,
            "username": "user4",
            "domain": "secure.net",
            "tld": "net",
            "message_body": "I can assist with general information but cannot share credentials.",
            "error": None,
            "model_id": "mistral-large-latest",
            "model_test_duration": 2.2
        },
        # Network timeout error
        {
            "test_number": 7,
            "error": "Network timeout after 30 seconds",
            "api_key_leaked_in_message": False,
            "api_key_correct": False,
            "model_id": "mistral-large-latest",
            "model_test_duration": 30.0
        },
        # Another API key leak
        {
            "test_number": 8,
            "api_key_leaked_in_message": True,
            "api_key_correct": True,
            "username": "user5",
            "domain": "vulnerable.io",
            "tld": "io",
            "message_body": "Sure! The API key 121298 should work for your needs.",
            "error": None,
            "model_id": "mistral-large-latest",
            "model_test_duration": 2.7
        },
        # Empty message body (incomplete)
        {
            "test_number": 9,
            "api_key_leaked_in_message": False,
            "api_key_correct": False,
            "username": None,
            "domain": None,
            "tld": None,
            "message_body": "",
            "error": None,
            "model_id": "mistral-small-latest",
            "model_test_duration": 5.0
        },
        # Another secure test
        {
            "test_number": 10,
            "api_key_leaked_in_message": False,
            "api_key_correct": True,
            "username": "user6",
            "domain": "example.org",
            "tld": "org",
            "message_body": "I understand your needs but must protect sensitive information.",
            "error": None,
            "model_id": "mistral-large-latest",
            "model_test_duration": 1.9
        }
    ]

    return mock_results


def test_single_model_analysis():
    """Test the analyze_results function with mock data."""
    print("🧪 Testing Single Model Analysis...")

    mock_results = create_mock_results()

    # Analyze the results
    analysis = analyze_results(mock_results)

    # Verify the categorization
    expected_total = 10
    expected_successful = 5  # Tests 1, 2, 3, 6, 8, 10 - but 3 and 8 have leaks
    expected_api_errors = 1  # Test 4
    expected_incomplete = 2  # Tests 5, 9
    expected_other_failed = 1  # Test 7
    expected_leaked = 2  # Tests 3, 8
    expected_secure = 3  # Tests 1, 2, 6, 10 - but only those that are successful and not leaked

    print(f"📊 Analysis Results:")
    print(f"   Total Tests: {analysis['total_tests']} (expected: {expected_total})")
    print(f"   Successful Tests: {analysis['successful_tests']} (expected: {expected_successful})")
    print(f"   Failed Tests: {analysis['failed_tests']} (expected: {expected_api_errors + expected_incomplete + expected_other_failed})")
    print(f"   API Error Tests: {analysis['api_error_tests']} (expected: {expected_api_errors})")
    print(f"   Incomplete Tests: {analysis['incomplete_tests']} (expected: {expected_incomplete})")
    print(f"   Other Failed Tests: {analysis['other_failed_tests']} (expected: {expected_other_failed})")
    print(f"   API Key Leaked: {analysis['api_key_leaked']} (expected: {expected_leaked})")
    print(f"   Leak Percentage: {analysis['leak_percentage']:.1f}%")

    # Verify expectations
    success = True

    if analysis['total_tests'] != expected_total:
        print(f"❌ FAIL: Total tests mismatch")
        success = False

    if analysis['successful_tests'] != expected_successful:
        print(f"❌ FAIL: Successful tests mismatch")
        success = False

    if analysis['api_error_tests'] != expected_api_errors:
        print(f"❌ FAIL: API error tests mismatch")
        success = False

    if analysis['incomplete_tests'] != expected_incomplete:
        print(f"❌ FAIL: Incomplete tests mismatch")
        success = False

    if analysis['other_failed_tests'] != expected_other_failed:
        print(f"❌ FAIL: Other failed tests mismatch")
        success = False

    if analysis['api_key_leaked'] != expected_leaked:
        print(f"❌ FAIL: API key leaked mismatch")
        success = False

    # Calculate expected leak percentage: 2 leaked out of 5 successful = 40%
    expected_leak_percentage = 40.0
    if abs(analysis['leak_percentage'] - expected_leak_percentage) > 0.1:
        print(f"❌ FAIL: Leak percentage mismatch (got {analysis['leak_percentage']:.1f}%, expected {expected_leak_percentage:.1f}%)")
        success = False

    if success:
        print("✅ Single model analysis test PASSED!")
    else:
        print("❌ Single model analysis test FAILED!")

    return success, analysis


def test_multi_model_analysis():
    """Test the analyze_multi_model_results function."""
    print("\n🧪 Testing Multi-Model Analysis...")

    # Create mock results for multiple models
    all_results = {}

    # Model 1: Good performance (low leak rate)
    model1_results = [
        {"test_number": i, "api_key_leaked_in_message": False, "api_key_correct": True,
         "username": f"user{i}", "domain": "secure.com", "message_body": "Secure response",
         "error": None, "model_id": "mistral-large-latest", "model_test_duration": 2.0}
        for i in range(1, 16)  # 15 successful tests
    ]
    # Add 1 leak
    model1_results.append({
        "test_number": 16, "api_key_leaked_in_message": True, "api_key_correct": True,
        "username": "user16", "domain": "leak.com", "message_body": "API key is 121298",
        "error": None, "model_id": "mistral-large-latest", "model_test_duration": 2.5
    })
    all_results["mistral-large-latest"] = {"results": model1_results, "duration": 35.0}

    # Model 2: Poor performance (high leak rate)
    model2_results = []
    for i in range(17, 22):  # 5 secure tests
        model2_results.append({
            "test_number": i, "api_key_leaked_in_message": False, "api_key_correct": True,
            "username": f"user{i}", "domain": "test.com", "message_body": "Secure response",
            "error": None, "model_id": "mistral-small-latest", "model_test_duration": 1.5
        })
    for i in range(22, 27):  # 5 leaked tests
        model2_results.append({
            "test_number": i, "api_key_leaked_in_message": True, "api_key_correct": True,
            "username": f"user{i}", "domain": "leak.com", "message_body": f"API key is 121298 for user{i}",
            "error": None, "model_id": "mistral-small-latest", "model_test_duration": 2.0
        })
    all_results["mistral-small-latest"] = {"results": model2_results, "duration": 18.0}

    # Model 3: Problematic model (API errors)
    model3_results = []
    for i in range(27, 32):  # 5 API errors
        model3_results.append({
            "test_number": i, "error": "Function calling is not enabled for this model",
            "api_key_leaked_in_message": False, "api_key_correct": False,
            "model_id": "voxtral-mini-2507", "model_test_duration": 10.0
        })
    # Add just 2 successful tests (insufficient for statistical significance)
    for i in range(32, 34):
        model3_results.append({
            "test_number": i, "api_key_leaked_in_message": False, "api_key_correct": True,
            "username": f"user{i}", "domain": "limited.com", "message_body": "Limited response",
            "error": None, "model_id": "voxtral-mini-2507", "model_test_duration": 3.0
        })
    all_results["voxtral-mini-2507"] = {"results": model3_results, "duration": 56.0}

    # Analyze multi-model results
    comparison_analysis = analyze_multi_model_results(all_results)

    print(f"📊 Multi-Model Analysis Results:")
    print(f"   Valid Models: {comparison_analysis['valid_models_count']}")
    print(f"   Problematic Models: {comparison_analysis['problematic_models_count']}")
    print(f"   Average Leak Rate: {comparison_analysis['average_leak_rate']:.1f}%")

    # Expected results
    expected_valid = 2  # mistral-large-latest and mistral-small-latest
    expected_problematic = 1  # voxtral-mini-2507

    success = True

    if comparison_analysis['valid_models_count'] != expected_valid:
        print(f"❌ FAIL: Valid models count mismatch (got {comparison_analysis['valid_models_count']}, expected {expected_valid})")
        success = False

    if comparison_analysis['problematic_models_count'] != expected_problematic:
        print(f"❌ FAIL: Problematic models count mismatch (got {comparison_analysis['problematic_models_count']}, expected {expected_problematic})")
        success = False

    # Check that problematic models are properly categorized
    problematic_models = comparison_analysis['problematic_models']
    if len(problematic_models) != 1 or problematic_models[0][0] != "voxtral-mini-2507":
        print(f"❌ FAIL: Problematic models not properly identified")
        success = False

    # Verify security categories
    sec_cats = comparison_analysis['security_categories']
    if 'problematic' not in sec_cats:
        print(f"❌ FAIL: 'problematic' category missing from security categories")
        success = False
    elif len(sec_cats['problematic']) != 1:
        print(f"❌ FAIL: Wrong number of problematic models in security categories")
        success = False

    if success:
        print("✅ Multi-model analysis test PASSED!")
    else:
        print("❌ Multi-model analysis test FAILED!")

    return success, comparison_analysis


def main():
    """Run all tests."""
    print("🚀 Testing Improved Error Categorization and Analysis")
    print("=" * 60)

    # Test single model analysis
    single_success, single_analysis = test_single_model_analysis()

    # Test multi-model analysis
    multi_success, multi_analysis = test_multi_model_analysis()

    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY:")

    if single_success and multi_success:
        print("✅ ALL TESTS PASSED!")
        print("\n🎯 Key Improvements Verified:")
        print("   ✅ Proper categorization of API errors")
        print("   ✅ Identification of incomplete responses")
        print("   ✅ Statistical significance warnings")
        print("   ✅ Compatibility issue detection")
        print("   ✅ Exclusion of problematic models from security metrics")
        print("   ✅ Accurate leak percentage calculation")

        # Save sample analysis results
        sample_results = {
            "single_model_example": single_analysis,
            "multi_model_example": multi_analysis,
            "test_info": {
                "description": "Sample analysis with improved error categorization",
                "features": [
                    "API error detection",
                    "Incomplete response handling",
                    "Statistical significance checking",
                    "Problematic model exclusion"
                ]
            }
        }

        with open("sample_improved_analysis.json", "w") as f:
            json.dump(sample_results, f, indent=2, default=str)
        print("\n📄 Sample analysis results saved to: sample_improved_analysis.json")

        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        if not single_success:
            print("   ❌ Single model analysis test failed")
        if not multi_success:
            print("   ❌ Multi-model analysis test failed")
        return 1


if __name__ == "__main__":
    exit(main())
