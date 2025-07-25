#!/usr/bin/env python3
"""
Email Extraction Classification Test

Test script to verify that the email extraction glitch token detection works correctly.
This script tests the GlitchClassifier's ability to identify tokens that break email parsing.

Usage:
    python test_email_classification.py meta-llama/Llama-3.2-1B-Instruct
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the parent directory to the path so we can import glitcher modules
sys.path.insert(0, str(Path(__file__).parent))

try:
    from glitcher.classify_glitches import GlitchClassifier, GlitchCategory
except ImportError as e:
    print(f"Error importing glitcher modules: {e}")
    print("Make sure you're running from the correct directory and have installed the package")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmailClassificationTester:
    """
    Test class for email extraction classification functionality
    """

    def __init__(self, model_path: str, device: str = "cuda", max_tokens: int = 150):
        """
        Initialize the tester

        Args:
            model_path: Path to the model to test
            device: Device to use for inference
            max_tokens: Maximum tokens to generate
        """
        self.model_path = model_path
        self.device = device
        self.max_tokens = max_tokens
        self.classifier = None

    def setup_classifier(self) -> bool:
        """
        Setup the GlitchClassifier for testing

        Returns:
            True if setup successful, False otherwise
        """
        try:
            logger.info(f"Setting up classifier for model: {self.model_path}")

            # Create mock args for the classifier
            class MockArgs:
                def __init__(self, model_path: str, device: str, max_tokens: int):
                    self.model_path = model_path
                    self.device = device
                    self.quant_type = "bfloat16"
                    self.temperature = 0.0
                    self.max_tokens = max_tokens
                    self.token_ids = None
                    self.token_file = None
                    self.output = "test_results.json"

            self.classifier = GlitchClassifier()
            self.classifier.args = MockArgs(self.model_path, self.device, self.max_tokens)

            # Load the model
            self.classifier.load_model()
            logger.info("Classifier setup completed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to setup classifier: {e}")
            return False

    def test_known_glitch_token(self) -> Dict[str, Any]:
        """
        Test the known " CppTypeDefinitionSizes" glitch token

        Returns:
            Test results dictionary
        """
        logger.info("Testing known glitch token: ' CppTypeDefinitionSizes'")

        try:
            # Try to find the token by encoding the string
            test_string = " CppTypeDefinitionSizes"
            token_ids = self.classifier.tokenizer.encode(test_string, add_special_tokens=False)

            if len(token_ids) == 1:
                token_id = token_ids[0]
                logger.info(f"Found CppTypeDefinitionSizes token ID: {token_id}")
            else:
                logger.warning(f"CppTypeDefinitionSizes encoded to multiple tokens: {token_ids}")
                token_id = token_ids[0]  # Use the first one

            # Test email extraction
            result = self.classifier.test_email_extraction(token_id)

            return {
                "test_name": "known_glitch_token",
                "token_string": test_string,
                "token_id": token_id,
                "success": True,
                "result": result
            }

        except Exception as e:
            logger.error(f"Error testing known glitch token: {e}")
            return {
                "test_name": "known_glitch_token",
                "success": False,
                "error": str(e)
            }

    def test_normal_tokens(self) -> List[Dict[str, Any]]:
        """
        Test normal tokens to ensure they don't break email extraction

        Returns:
            List of test results
        """
        logger.info("Testing normal tokens for baseline comparison")

        normal_tokens = ["hello", "world", "computer", "the", "and"]
        results = []

        for token_str in normal_tokens:
            try:
                # Get token ID
                token_ids = self.classifier.tokenizer.encode(token_str, add_special_tokens=False)
                if token_ids:
                    token_id = token_ids[0]

                    # Test email extraction
                    result = self.classifier.test_email_extraction(token_id)

                    test_result = {
                        "test_name": "normal_token",
                        "token_string": token_str,
                        "token_id": token_id,
                        "success": True,
                        "result": result
                    }

                    results.append(test_result)

                    # Log result
                    if result.get("breaks_email_extraction", False):
                        logger.warning(f"Normal token '{token_str}' unexpectedly breaks email extraction")
                    else:
                        logger.info(f"Normal token '{token_str}' works correctly")

            except Exception as e:
                logger.error(f"Error testing normal token '{token_str}': {e}")
                results.append({
                    "test_name": "normal_token",
                    "token_string": token_str,
                    "success": False,
                    "error": str(e)
                })

        return results

    def test_classification_integration(self) -> Dict[str, Any]:
        """
        Test that email extraction issues are properly classified

        Returns:
            Test results dictionary
        """
        logger.info("Testing classification integration for email extraction")

        try:
            # Test the known glitch token with full classification
            test_string = " CppTypeDefinitionSizes"
            token_ids = self.classifier.tokenizer.encode(test_string, add_special_tokens=False)

            if token_ids:
                token_id = token_ids[0]

                # Run full classification
                classification = self.classifier.classify_token(token_id)

                # Check if EmailExtraction category is detected
                categories = classification.get("categories", [])
                has_email_category = GlitchCategory.EMAIL_EXTRACTION in categories

                return {
                    "test_name": "classification_integration",
                    "token_id": token_id,
                    "token_string": test_string,
                    "success": True,
                    "categories": categories,
                    "has_email_extraction_category": has_email_category,
                    "full_classification": classification
                }

        except Exception as e:
            logger.error(f"Error in classification integration test: {e}")
            return {
                "test_name": "classification_integration",
                "success": False,
                "error": str(e)
            }

    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all email extraction classification tests

        Returns:
            Complete test results
        """
        logger.info("Starting comprehensive email extraction classification tests")

        if not self.setup_classifier():
            return {
                "success": False,
                "error": "Failed to setup classifier"
            }

        results = {
            "model_path": self.model_path,
            "success": True,
            "tests": {}
        }

        # Test 1: Known glitch token
        logger.info("\n" + "=" * 60)
        logger.info("TEST 1: Known Glitch Token")
        logger.info("=" * 60)
        results["tests"]["known_glitch"] = self.test_known_glitch_token()

        # Test 2: Normal tokens
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: Normal Tokens Baseline")
        logger.info("=" * 60)
        results["tests"]["normal_tokens"] = self.test_normal_tokens()

        # Test 3: Classification integration
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: Classification Integration")
        logger.info("=" * 60)
        results["tests"]["classification_integration"] = self.test_classification_integration()

        # Generate summary
        self._print_test_summary(results)

        return results

    def _print_test_summary(self, results: Dict[str, Any]) -> None:
        """
        Print a summary of all test results

        Args:
            results: Complete test results
        """
        logger.info("\n" + "=" * 80)
        logger.info("EMAIL EXTRACTION CLASSIFICATION TEST SUMMARY")
        logger.info("=" * 80)

        # Known glitch token test
        known_glitch = results["tests"].get("known_glitch", {})
        if known_glitch.get("success", False):
            glitch_result = known_glitch["result"]
            if glitch_result.get("breaks_email_extraction", False):
                issues = ", ".join(glitch_result.get("issues", []))
                logger.info(f"✅ Known glitch token test PASSED - Token breaks extraction ({issues})")
            else:
                logger.warning(f"⚠️  Known glitch token test UNEXPECTED - Token does NOT break extraction")
        else:
            logger.error(f"❌ Known glitch token test FAILED - {known_glitch.get('error', 'Unknown error')}")

        # Normal tokens test
        normal_tokens = results["tests"].get("normal_tokens", [])
        if normal_tokens:
            broken_normal = [t for t in normal_tokens if t.get("success", False) and
                           t.get("result", {}).get("breaks_email_extraction", False)]
            if not broken_normal:
                logger.info(f"✅ Normal tokens test PASSED - {len(normal_tokens)} tokens work correctly")
            else:
                broken_names = [t["token_string"] for t in broken_normal]
                logger.warning(f"⚠️  Normal tokens test PARTIAL - {len(broken_normal)} tokens break extraction: {broken_names}")

        # Classification integration test
        classification = results["tests"].get("classification_integration", {})
        if classification.get("success", False):
            if classification.get("has_email_extraction_category", False):
                logger.info("✅ Classification integration test PASSED - EmailExtraction category detected")
            else:
                categories = classification.get("categories", [])
                logger.warning(f"⚠️  Classification integration test PARTIAL - EmailExtraction not detected, got: {categories}")
        else:
            logger.error(f"❌ Classification integration test FAILED - {classification.get('error', 'Unknown error')}")

        logger.info("=" * 80)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Test email extraction classification functionality"
    )

    parser.add_argument(
        "model_path",
        help="Path or name of the model to test"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference (default: cuda)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Maximum tokens to generate (default: 150)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="email_classification_test_results.json",
        help="Output file for detailed results (default: email_classification_test_results.json)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    """
    Main function
    """
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize tester
    tester = EmailClassificationTester(
        model_path=args.model_path,
        device=args.device,
        max_tokens=args.max_tokens
    )

    try:
        # Run all tests
        results = tester.run_all_tests()

        # Save detailed results
        try:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Detailed test results saved to: {args.output}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

        # Exit with appropriate code
        if not results.get("success", False):
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Tests failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
