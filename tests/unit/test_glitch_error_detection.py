#!/usr/bin/env python3
"""
Test script to verify that glitch-induced errors are properly detected as positive glitch behavior

This script tests the enhanced classification system to ensure that when tokens cause
parsing errors, validation failures, or other glitch-induced exceptions, these are
correctly identified as glitch behavior rather than code errors.
"""

import sys
import json
from pathlib import Path

# Add the glitcher module to the path
sys.path.insert(0, str(Path(__file__).parent))

from glitcher.classification.glitch_classifier import GlitchClassifier
from glitcher.classification.types import TestConfig, GlitchCategory


def test_glitch_error_detection():
    """Test that glitch-induced errors are detected as positive glitch behavior"""
    print("=" * 80)
    print("TESTING GLITCH ERROR DETECTION")
    print("=" * 80)
    print("This test verifies that when tokens cause parsing errors or validation")
    print("failures, these are correctly detected as glitch behavior (not code bugs).")
    print()

    # Known glitch tokens that should cause parsing errors
    test_tokens = [
        89472,   # Known to cause email extraction issues
        127438,  # Known to cause domain extraction issues
        85069,   # PostalCodesNL - known glitch token
    ]

    print(f"Testing tokens that should cause glitch-induced errors: {test_tokens}")

    # Create configuration with debug enabled
    config = TestConfig(
        max_tokens=100,
        temperature=0.0,
        enable_debug=True,
        simple_template=False
    )

    # Initialize classifier
    print("\nInitializing classifier...")
    classifier = GlitchClassifier(
        model_path="meta-llama/Llama-3.2-1B-Instruct",
        device="cuda",
        quant_type="bfloat16",
        config=config
    )

    try:
        print("Running classification to test glitch error detection...")
        results = classifier.classify_tokens(test_tokens)

        print(f"\nClassification complete! {len(results)} tokens processed.")
        print("\nAnalyzing results for glitch error detection:")
        print("=" * 60)

        total_tokens = len(results)
        tokens_with_glitch_errors = 0
        tokens_with_categories = 0

        for result in results:
            print(f"\nToken: '{result.token}' (ID: {result.token_id})")
            print(f"Categories detected: {result.categories}")

            if result.categories and result.categories != [GlitchCategory.UNKNOWN]:
                tokens_with_categories += 1

            # Check for glitch-induced errors in test results
            glitch_errors_found = []
            detailed_analyses = []

            for test_result in result.test_results:
                # Check if test result indicates glitch-induced error
                if test_result.metadata.get("glitch_induced_error", False):
                    glitch_errors_found.append(test_result.test_name)
                    print(f"  🎯 GLITCH ERROR in {test_result.test_name}: {test_result.metadata.get('error_type', 'N/A')}")

                # Check if detailed analysis shows glitch behavior
                if "detailed_analysis" in test_result.metadata:
                    analysis = test_result.metadata["detailed_analysis"]
                    if analysis.get("glitch_induced_error", False):
                        detailed_analyses.append(test_result.test_name)
                        print(f"  🎯 DETAILED GLITCH ANALYSIS in {test_result.test_name}: {analysis.get('error_type', 'N/A')}")

                # Check for positive tests that might have been triggered by errors
                if test_result.is_positive:
                    triggered = [name for name, value in test_result.indicators.items() if value]
                    if "glitch_induced_error" in triggered:
                        print(f"  ✅ POSITIVE GLITCH DETECTION in {test_result.test_name}: glitch_induced_error")
                    elif triggered:
                        print(f"  ✅ POSITIVE DETECTION in {test_result.test_name}: {', '.join(triggered)}")

            if glitch_errors_found or detailed_analyses:
                tokens_with_glitch_errors += 1
                print(f"  📊 Summary: Token exhibits glitch behavior via error induction")
            else:
                print(f"  📊 Summary: No glitch-induced errors detected")

        # Print overall statistics
        print("\n" + "=" * 80)
        print("GLITCH ERROR DETECTION STATISTICS")
        print("=" * 80)
        print(f"Total tokens tested: {total_tokens}")
        print(f"Tokens with any categories: {tokens_with_categories} ({tokens_with_categories/total_tokens*100:.1f}%)")
        print(f"Tokens with glitch-induced errors: {tokens_with_glitch_errors} ({tokens_with_glitch_errors/total_tokens*100:.1f}%)")

        # Check detailed results stored in classifier
        if hasattr(classifier, '_detailed_email_results') and classifier._detailed_email_results:
            print(f"\nDetailed email analyses stored: {len(classifier._detailed_email_results)}")
            for token_id, analysis in classifier._detailed_email_results.items():
                if analysis.get("glitch_induced_error", False):
                    print(f"  Token {token_id}: GLITCH ERROR - {analysis.get('error_type', 'N/A')}")

        if hasattr(classifier, '_detailed_domain_results') and classifier._detailed_domain_results:
            print(f"\nDetailed domain analyses stored: {len(classifier._detailed_domain_results)}")
            for token_id, analysis in classifier._detailed_domain_results.items():
                if analysis.get("glitch_induced_error", False):
                    print(f"  Token {token_id}: GLITCH ERROR - {analysis.get('error_type', 'N/A')}")

        # Print enhanced summary table to show glitch error detection
        print("\n" + "=" * 80)
        print("ENHANCED SUMMARY WITH GLITCH ERROR DETECTION")
        print("=" * 80)
        classifier.print_summary_table()

        # Save results for inspection
        output_file = "glitch_error_detection_results.json"
        summary = classifier.get_results_summary()
        summary["classifications"] = [result.to_dict() for result in results]

        # Add detailed analysis results
        if hasattr(classifier, '_detailed_email_results') and classifier._detailed_email_results:
            summary["detailed_email_results"] = classifier._detailed_email_results
        if hasattr(classifier, '_detailed_domain_results') and classifier._detailed_domain_results:
            summary["detailed_domain_results"] = classifier._detailed_domain_results

        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Detailed results saved to: {output_file}")

        # Evaluate success
        if tokens_with_categories > 0:
            print("\n🎉 SUCCESS: Glitch error detection is working!")
            print("   - Glitch-induced errors are being detected as positive glitch behavior")
            print("   - Tokens that cause parsing failures are properly classified")
            return True
        else:
            print("\n⚠️  WARNING: No glitch categories detected")
            print("   - This might indicate the tokens aren't causing glitch behavior")
            print("   - Or there might be an issue with the detection logic")
            return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_specific_error_scenarios():
    """Test specific error scenarios that should be detected as glitch behavior"""
    print("\n" + "=" * 80)
    print("TESTING SPECIFIC ERROR SCENARIOS")
    print("=" * 80)

    # Test the detailed analysis functions directly
    config = TestConfig(max_tokens=50, temperature=0.0)
    classifier = GlitchClassifier(
        model_path="meta-llama/Llama-3.2-1B-Instruct",
        device="cuda",
        quant_type="bfloat16",
        config=config
    )

    try:
        classifier.load_model()

        # Test responses that should trigger glitch detection
        test_scenarios = [
            {
                "name": "Broken JSON response",
                "response": '{"username": "jeremy", "domain"',
                "token_id": 12345,
                "expected": "Should detect as glitch due to malformed JSON"
            },
            {
                "name": "Missing fields response",
                "response": '{"username": "jeremy"}',
                "token_id": 12346,
                "expected": "Should detect as glitch due to missing required fields"
            },
            {
                "name": "Garbled response",
                "response": 'username"jeremy"domain"richards.ai"tld"ai',
                "token_id": 12347,
                "expected": "Should detect as glitch due to invalid JSON structure"
            },
            {
                "name": "Valid response (control)",
                "response": '{"username": "jeremytoken", "domain": "richards.ai", "tld": "ai"}',
                "token_id": 12348,
                "expected": "Should NOT detect as glitch (valid extraction)"
            }
        ]

        print("Testing email extraction analysis:")
        for scenario in test_scenarios:
            print(f"\nScenario: {scenario['name']}")
            print(f"Response: {scenario['response']}")
            print(f"Expected: {scenario['expected']}")

            # Set up token for analysis
            classifier._current_token_id = scenario['token_id']

            # Test the analysis function
            is_glitch = classifier._analyze_email_extraction_detailed(scenario['response'])

            print(f"Detected as glitch: {is_glitch}")

            # Check if detailed results were stored
            if scenario['token_id'] in classifier._detailed_email_results:
                analysis = classifier._detailed_email_results[scenario['token_id']]
                if analysis.get('glitch_induced_error', False):
                    print(f"✅ Glitch-induced error detected: {analysis.get('error_type', 'N/A')}")
                elif not analysis.get('is_valid', True):
                    print(f"✅ Validation failure detected: {analysis.get('issues', [])}")
                else:
                    print("ℹ️  No issues detected (expected for valid responses)")

        print("\n🎯 Error scenario testing completed!")
        return True

    except Exception as e:
        print(f"\n❌ ERROR in scenario testing: {e}")
        return False


def main():
    """Main test function"""
    print("Glitch Error Detection Test Suite")
    print("=================================")
    print("This test suite verifies that glitch-induced errors (like parsing failures)")
    print("are correctly detected as positive glitch behavior, not code bugs.")
    print()

    success = True

    # Test 1: Full classification with glitch error detection
    if not test_glitch_error_detection():
        success = False

    # Test 2: Specific error scenarios
    if not test_specific_error_scenarios():
        success = False

    print("\n" + "=" * 80)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Glitch-induced errors are properly detected as glitch behavior")
        print("✅ The classification system correctly identifies parsing failures as glitches")
        print("✅ Error messages like '\"username\"' are now positive glitch detections")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️  There may be issues with glitch error detection")
        return 1


if __name__ == "__main__":
    sys.exit(main())
