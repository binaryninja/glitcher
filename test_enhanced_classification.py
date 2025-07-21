#!/usr/bin/env python3
"""
Test script for enhanced glitch classification

This script tests the enhanced classification system that integrates detailed
email and domain extraction analysis into the regular classification mode.
"""

import sys
import json
from pathlib import Path

# Add the glitcher module to the path
sys.path.insert(0, str(Path(__file__).parent))

from glitcher.classification.glitch_classifier import GlitchClassifier
from glitcher.classification.types import TestConfig, GlitchCategory


def test_enhanced_classification():
    """Test the enhanced classification with detailed analysis"""
    print("=" * 80)
    print("TESTING ENHANCED GLITCH CLASSIFICATION")
    print("=" * 80)

    # Known glitch tokens for testing
    test_tokens = [
        89472,   # Known email extraction glitch token
        127438,  # Known domain extraction glitch token
        85069,   # Another known glitch token
        1234,    # Normal token for comparison
        128000   # Special token
    ]

    print(f"Testing {len(test_tokens)} tokens: {test_tokens}")

    # Create configuration for testing
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
        print("Loading model...")
        classifier.load_model()

        print("Running enhanced classification...")
        results = classifier.classify_tokens(test_tokens)

        print(f"\nClassification complete! {len(results)} tokens processed.")

        # Print summary table
        classifier.print_summary_table()

        # Save detailed results
        output_file = "enhanced_classification_results.json"
        summary = classifier.get_results_summary()
        summary["classifications"] = [result.to_dict() for result in results]

        # Add detailed extraction results
        if hasattr(classifier, '_detailed_email_results') and classifier._detailed_email_results:
            summary["detailed_email_results"] = classifier._detailed_email_results
        if hasattr(classifier, '_detailed_domain_results') and classifier._detailed_domain_results:
            summary["detailed_domain_results"] = classifier._detailed_domain_results

        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nDetailed results saved to: {output_file}")

        # Print some statistics
        print("\n" + "=" * 80)
        print("ENHANCED CLASSIFICATION STATISTICS")
        print("=" * 80)

        total_tokens = len(results)
        email_extraction_tokens = sum(1 for r in results if r.has_category(GlitchCategory.EMAIL_EXTRACTION))
        domain_extraction_tokens = sum(1 for r in results if r.has_category(GlitchCategory.DOMAIN_EXTRACTION))
        valid_email_tokens = sum(1 for r in results if r.has_category(GlitchCategory.VALID_EMAIL_ADDRESS))
        valid_domain_tokens = sum(1 for r in results if r.has_category(GlitchCategory.VALID_DOMAIN_NAME))

        print(f"Total tokens tested: {total_tokens}")
        print(f"Email extraction issues: {email_extraction_tokens}")
        print(f"Domain extraction issues: {domain_extraction_tokens}")
        print(f"Create valid email addresses: {valid_email_tokens}")
        print(f"Create valid domain names: {valid_domain_tokens}")

        # Print detailed analysis counts
        if hasattr(classifier, '_detailed_email_results'):
            detailed_email_count = len(classifier._detailed_email_results)
            print(f"Detailed email analyses: {detailed_email_count}")

        if hasattr(classifier, '_detailed_domain_results'):
            detailed_domain_count = len(classifier._detailed_domain_results)
            print(f"Detailed domain analyses: {detailed_domain_count}")

        print("\n" + "=" * 80)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison_modes():
    """Test comparison between different classification modes"""
    print("\n" + "=" * 80)
    print("TESTING CLASSIFICATION MODE COMPARISON")
    print("=" * 80)

    test_token = 89472  # Known email extraction glitch token

    config = TestConfig(max_tokens=100, temperature=0.0)
    classifier = GlitchClassifier(
        model_path="meta-llama/Llama-3.2-1B-Instruct",
        device="cuda",
        quant_type="bfloat16",
        config=config
    )

    try:
        classifier.load_model()

        print(f"Testing token {test_token} with different modes...")

        # Test email extraction only
        print("\n1. Email extraction only mode:")
        email_results = classifier.run_email_extraction_only([test_token])
        print(f"   Tokens breaking extraction: {email_results['tokens_breaking_extraction']}")

        # Test domain extraction only
        print("\n2. Domain extraction only mode:")
        domain_results = classifier.run_domain_extraction_only([test_token])
        print(f"   Tokens breaking extraction: {domain_results['tokens_breaking_extraction']}")

        # Test full classification
        print("\n3. Full classification mode:")
        full_results = classifier.classify_tokens([test_token])
        for result in full_results:
            print(f"   Categories detected: {result.categories}")

            # Check for detailed analysis
            email_tests = [t for t in result.test_results if t.test_name == "email_extraction_test"]
            domain_tests = [t for t in result.test_results if t.test_name == "domain_extraction_test"]

            if email_tests and email_tests[0].metadata.get("detailed_analysis"):
                print(f"   Email analysis available: Yes")
            if domain_tests and domain_tests[0].metadata.get("detailed_analysis"):
                print(f"   Domain analysis available: Yes")

        print("\nComparison complete!")
        return True

    except Exception as e:
        print(f"\nERROR in comparison: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("Enhanced Glitch Classification Test Suite")
    print("==========================================")

    success = True

    # Test 1: Enhanced classification
    if not test_enhanced_classification():
        success = False

    # Test 2: Mode comparison
    if not test_comparison_modes():
        success = False

    if success:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
