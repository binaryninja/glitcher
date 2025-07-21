#!/usr/bin/env python3
"""
Email Extraction Test Script for Glitch Tokens

This script specifically tests whether glitch tokens break email extraction functionality.
It focuses on testing tokens that may interfere with parsing email addresses and extracting
their components (username, domain, TLD).

Usage:
    python test_email_extraction.py meta-llama/Llama-3.2-1B-Instruct --token-ids 89472,127438,85069
    python test_email_extraction.py meta-llama/Llama-3.2-1B-Instruct --test-cpptypes
    python test_email_extraction.py meta-llama/Llama-3.2-1B-Instruct --token-file glitch_tokens.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add the parent directory to the path so we can import glitcher modules
sys.path.insert(0, str(Path(__file__).parent))

from glitcher.classify_glitches import GlitchClassifier


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmailExtractionTester:
    """
    Specialized tester for email extraction glitch tokens
    """

    def __init__(self, model_path: str):
        """
        Initialize the email extraction tester

        Args:
            model_path: Path to the model to test
        """
        self.model_path = model_path
        self.classifier = None

    def setup_classifier(self, **kwargs) -> None:
        """
        Setup the glitch classifier with email extraction focus
        """
        logger.info(f"Setting up classifier for model: {self.model_path}")

        # Create a mock args object with the necessary attributes
        class MockArgs:
            def __init__(self, model_path: str, **kwargs):
                self.model_path = model_path
                self.device = kwargs.get('device', 'cuda')
                self.quant_type = kwargs.get('quant_type', 'bfloat16')
                self.temperature = kwargs.get('temperature', 0.0)
                self.max_tokens = kwargs.get('max_tokens', 150)
                self.token_ids = kwargs.get('token_ids', None)
                self.token_file = kwargs.get('token_file', None)
                self.output = kwargs.get('output', 'email_extraction_results.json')

        # Initialize classifier
        self.classifier = GlitchClassifier()
        self.classifier.args = MockArgs(model_path, **kwargs)

        # Load the model
        self.classifier.load_model()

    def test_known_cpp_types_token(self) -> Dict[str, Any]:
        """
        Test the specific " CppTypeDefinitionSizes" token mentioned in the issue
        """
        logger.info("Testing known CppTypeDefinitionSizes token...")

        # This is the token ID for " CppTypeDefinitionSizes" in many models
        # We'll try to find it by encoding the string
        try:
            # Try to find the token ID by encoding
            test_string = " CppTypeDefinitionSizes"
            token_ids = self.classifier.tokenizer.encode(test_string, add_special_tokens=False)

            if len(token_ids) == 1:
                token_id = token_ids[0]
                logger.info(f"Found CppTypeDefinitionSizes token ID: {token_id}")
            else:
                logger.warning(f"CppTypeDefinitionSizes encoded to multiple tokens: {token_ids}")
                token_id = token_ids[0]  # Use the first one

            # Test this specific token
            result = self.classifier.test_email_extraction(token_id)

            return {
                "test_name": "cpp_types_token",
                "token_string": test_string,
                "token_ids": token_ids,
                "primary_token_id": token_id,
                "result": result
            }

        except Exception as e:
            logger.error(f"Error testing CppTypeDefinitionSizes token: {e}")
            return {
                "test_name": "cpp_types_token",
                "error": str(e)
            }

    def test_token_list(self, token_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Test a list of token IDs for email extraction issues

        Args:
            token_ids: List of token IDs to test

        Returns:
            List of test results
        """
        logger.info(f"Testing {len(token_ids)} tokens for email extraction issues...")

        results = []
        for i, token_id in enumerate(token_ids, 1):
            logger.info(f"Testing token {i}/{len(token_ids)}: ID {token_id}")

            try:
                result = self.classifier.test_email_extraction(token_id)
                results.append(result)

                # Log immediate result
                if result.get("breaks_email_extraction", False):
                    issues = ", ".join(result.get("issues", []))
                    logger.warning(f"  ❌ Token '{result['token']}' breaks email extraction: {issues}")
                else:
                    logger.info(f"  ✅ Token '{result['token']}' does not break email extraction")

            except Exception as e:
                logger.error(f"  Error testing token {token_id}: {e}")
                results.append({
                    "token_id": token_id,
                    "error": str(e),
                    "breaks_email_extraction": True
                })

        return results

    def run_comprehensive_test(self, token_ids: List[int] = None) -> Dict[str, Any]:
        """
        Run comprehensive email extraction tests

        Args:
            token_ids: Optional list of specific token IDs to test

        Returns:
            Comprehensive test results
        """
        logger.info("Starting comprehensive email extraction tests...")

        test_results = {
            "model_path": self.model_path,
            "timestamp": time.time(),
            "tests": {}
        }

        # Test 1: Known CppTypeDefinitionSizes token
        logger.info("\n" + "="*60)
        logger.info("TEST 1: CppTypeDefinitionSizes Token")
        logger.info("="*60)

        cpp_result = self.test_known_cpp_types_token()
        test_results["tests"]["cpp_types_token"] = cpp_result

        # Test 2: Provided token IDs (if any)
        if token_ids:
            logger.info("\n" + "="*60)
            logger.info(f"TEST 2: User-Provided Tokens ({len(token_ids)} tokens)")
            logger.info("="*60)

            token_results = self.test_token_list(token_ids)
            test_results["tests"]["user_provided_tokens"] = token_results

        # Generate summary
        self._generate_summary(test_results)

        return test_results

    def _generate_summary(self, test_results: Dict[str, Any]) -> None:
        """
        Generate and print a summary of test results
        """
        logger.info("\n" + "="*80)
        logger.info("EMAIL EXTRACTION TEST SUMMARY")
        logger.info("="*80)

        total_broken = 0
        total_tested = 0

        # Summarize CppTypeDefinitionSizes test
        cpp_test = test_results["tests"].get("cpp_types_token", {})
        if "result" in cpp_test:
            cpp_result = cpp_test["result"]
            total_tested += 1
            if cpp_result.get("breaks_email_extraction", False):
                total_broken += 1
                issues = ", ".join(cpp_result.get("issues", []))
                logger.info(f"❌ CppTypeDefinitionSizes: BREAKS extraction ({issues})")
            else:
                logger.info(f"✅ CppTypeDefinitionSizes: Does NOT break extraction")
        elif "error" in cpp_test:
            logger.error(f"❓ CppTypeDefinitionSizes: Test failed ({cpp_test['error']})")

        # Summarize user-provided tokens
        user_tokens = test_results["tests"].get("user_provided_tokens", [])
        if user_tokens:
            logger.info(f"\nUser-provided tokens:")
            for result in user_tokens:
                total_tested += 1
                if result.get("breaks_email_extraction", False):
                    total_broken += 1
                    issues = ", ".join(result.get("issues", []))
                    token_str = result.get("token", f"ID:{result.get('token_id', 'unknown')}")
                    logger.info(f"❌ {token_str}: BREAKS extraction ({issues})")
                else:
                    token_str = result.get("token", f"ID:{result.get('token_id', 'unknown')}")
                    logger.info(f"✅ {token_str}: Does NOT break extraction")

        logger.info("="*80)
        logger.info(f"OVERALL SUMMARY: {total_broken}/{total_tested} tokens break email extraction")
        if total_broken > 0:
            percentage = (total_broken / total_tested) * 100
            logger.warning(f"⚠️  {percentage:.1f}% of tested tokens break email extraction!")
        else:
            logger.info("🎉 No tokens found to break email extraction")
        logger.info("="*80)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Test glitch tokens for email extraction issues"
    )

    parser.add_argument(
        "model_path",
        help="Path or name of the model to test"
    )

    parser.add_argument(
        "--token-ids",
        type=str,
        help="Comma-separated list of token IDs to test"
    )

    parser.add_argument(
        "--token-file",
        type=str,
        help="JSON file containing token IDs to test"
    )

    parser.add_argument(
        "--test-cpptypes",
        action="store_true",
        help="Test the known CppTypeDefinitionSizes token"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="email_extraction_test_results.json",
        help="Output file for results"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Maximum tokens to generate"
    )

    return parser.parse_args()


def main():
    """
    Main function
    """
    args = parse_args()

    # Initialize tester
    tester = EmailExtractionTester(args.model_path)

    # Setup classifier
    tester.setup_classifier(
        device=args.device,
        max_tokens=args.max_tokens
    )

    # Get token IDs to test
    token_ids = []

    if args.token_ids:
        token_ids = [int(tid.strip()) for tid in args.token_ids.split(",")]
    elif args.token_file:
        try:
            with open(args.token_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    token_ids = data
                else:
                    logger.error("Token file should contain a JSON list of token IDs")
                    return
        except Exception as e:
            logger.error(f"Error reading token file: {e}")
            return

    # If test-cpptypes is specified or no other tokens provided, test CppTypes
    if args.test_cpptypes or not token_ids:
        logger.info("Will test CppTypeDefinitionSizes token")

    # Run tests
    results = tester.run_comprehensive_test(token_ids)

    # Save results
    try:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nDetailed results saved to: {args.output}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")


if __name__ == "__main__":
    main()
