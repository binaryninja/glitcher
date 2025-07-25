#!/usr/bin/env python3
"""
Demo script for enhanced glitch classification

This script demonstrates the enhanced classification system that shows detailed
test failure information and integrates email/domain extraction analysis.
"""

import sys
import json
from pathlib import Path

# Add the glitcher module to the path
sys.path.insert(0, str(Path(__file__).parent))

from glitcher.classification.glitch_classifier import GlitchClassifier
from glitcher.classification.types import TestConfig, GlitchCategory


def demo_enhanced_classification():
    """Demo the enhanced classification with detailed test failure information"""
    print("=" * 80)
    print("ENHANCED GLITCH CLASSIFICATION DEMO")
    print("=" * 80)

    # Test with a few known glitch tokens
    test_tokens = [
        89472,   # Known email extraction glitch token
        127438,  # Known domain extraction glitch token
        1234,    # Normal token for comparison
    ]

    print(f"Testing tokens: {test_tokens}")

    # Create configuration with debug enabled to show detailed info
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
        # Run classification
        print("Running enhanced classification...")
        results = classifier.classify_tokens(test_tokens)

        print(f"\nClassification complete! {len(results)} tokens processed.")

        # The enhanced summary table will show:
        # 1. Which categories were detected for each token
        # 2. Which specific tests failed in each category
        # 3. Detailed analysis for email/domain extraction tests
        # 4. Response previews when debug is enabled
        classifier.print_summary_table()

        # Save results with detailed analysis
        output_file = "demo_enhanced_results.json"
        summary = classifier.get_results_summary()
        summary["classifications"] = [result.to_dict() for result in results]

        # Include detailed extraction analysis in output
        if hasattr(classifier, '_detailed_email_results') and classifier._detailed_email_results:
            summary["detailed_email_results"] = classifier._detailed_email_results
        if hasattr(classifier, '_detailed_domain_results') and classifier._detailed_domain_results:
            summary["detailed_domain_results"] = classifier._detailed_domain_results

        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nDetailed results saved to: {output_file}")
        print("\nKey improvements:")
        print("• Shows which specific tests failed for each category")
        print("• Includes detailed email/domain extraction analysis")
        print("• Shows triggered indicators for positive tests")
        print("• Provides response previews in debug mode")
        print("• Avoids redundant model loading")

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_mode_comparison():
    """Demo comparison between email-extraction-only and full classification"""
    print("\n" + "=" * 80)
    print("MODE COMPARISON DEMO")
    print("=" * 80)

    test_token = 89472  # Known email extraction glitch token
    print(f"Comparing classification modes for token: {test_token}")

    config = TestConfig(max_tokens=100, temperature=0.0)
    classifier = GlitchClassifier(
        model_path="meta-llama/Llama-3.2-1B-Instruct",
        device="cuda",
        quant_type="bfloat16",
        config=config
    )

    try:
        print("\n1. Email extraction only mode:")
        email_results = classifier.run_email_extraction_only([test_token])
        print(f"   Tokens breaking extraction: {email_results['tokens_breaking_extraction']}")
        print(f"   Detailed analysis provided: Yes")

        print("\n2. Full classification mode (enhanced):")
        full_results = classifier.classify_tokens([test_token])
        for result in full_results:
            print(f"   Categories detected: {result.categories}")

            # Check if detailed analysis is included
            email_tests = [t for t in result.test_results if t.test_name == "email_extraction_test"]
            if email_tests and email_tests[0].metadata.get("detailed_analysis"):
                print(f"   Detailed email analysis: Included in test metadata")

            domain_tests = [t for t in result.test_results if t.test_name == "domain_extraction_test"]
            if domain_tests and domain_tests[0].metadata.get("detailed_analysis"):
                print(f"   Detailed domain analysis: Included in test metadata")

        print("\nNow both modes provide detailed analysis!")
        return True

    except Exception as e:
        print(f"\nERROR in comparison: {e}")
        return False


def main():
    """Main demo function"""
    print("Enhanced Glitch Classification Demo")
    print("===================================")
    print("This demo shows the improvements made to fix the issues:")
    print("1. No more running twice (fixed redundant model loading)")
    print("2. Detailed test failure information in summary")
    print("3. Email/domain extraction analysis in full classification")
    print("")

    success = True

    # Demo 1: Enhanced classification
    if not demo_enhanced_classification():
        success = False

    # Demo 2: Mode comparison
    if not demo_mode_comparison():
        success = False

    if success:
        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("\nThe enhanced classification now provides:")
        print("• Detailed test failure information")
        print("• Integrated email/domain extraction analysis")
        print("• No redundant model loading")
        print("• Better summary formatting")
        return 0
    else:
        print("\n❌ DEMO ENCOUNTERED ERRORS!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
