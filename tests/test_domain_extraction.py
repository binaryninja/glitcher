"""
Domain Extraction Test Script for Glitch Tokens

This script specifically tests whether glitch tokens break domain extraction functionality
from log files. It focuses on testing tokens that may interfere with parsing domain names
from postfix log entries and extracting them correctly for use with tools like 'dig'.

Usage:
    python test_domain_extraction.py meta-llama/Llama-3.2-1B-Instruct --token-ids 89472,127438,85069
    python test_domain_extraction.py meta-llama/Llama-3.2-1B-Instruct --test-cpptypes
    python test_domain_extraction.py meta-llama/Llama-3.2-1B-Instruct --token-file glitch_tokens.json
"""

import argparse
import json
import logging
import time
from typing import List, Dict, Any

from glitcher.classify_glitches import GlitchClassifier


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DomainExtractionTester:
    """
    Test class for domain extraction functionality with glitch tokens
    """

    def __init__(self, model_path: str, device: str = "cuda", quant_type: str = "bfloat16"):
        """
        Initialize the domain extraction tester

        Args:
            model_path: Path to the model to test
            device: Device to use for inference
            quant_type: Quantization type for the model
        """
        self.model_path = model_path
        self.device = device
        self.quant_type = quant_type

        # Set up arguments for the classifier
        args = argparse.Namespace()
        args.model_path = model_path
        args.device = device
        args.quant_type = quant_type
        args.temperature = 0.0
        args.max_tokens = 200
        args.debug_responses = True
        args.simple_template = False
        args.output = "domain_extraction_test_results.json"

        # Initialize the classifier
        self.classifier = GlitchClassifier(args)
        self.classifier.load_model()

        logger.info(f"Initialized domain extraction tester for model: {model_path}")
        logger.info(f"Using device: {device}, quantization: {quant_type}")

    def test_known_cpp_types_token(self) -> Dict[str, Any]:
        """
        Test the known CppTypeDefinitionSizes glitch token

        Returns:
            Test results dictionary
        """
        logger.info("Testing known CppTypeDefinitionSizes glitch token for domain extraction...")

        # Known glitch token ID for CppTypeDefinitionSizes
        cpp_token_id = 89472

        try:
            # Get token string
            token = self.classifier.tokenizer.decode([cpp_token_id])
            logger.info(f"Testing token: '{token}' (ID: {cpp_token_id})")

            # Test this specific token
            result = self.classifier.test_domain_extraction(cpp_token_id)

            return {
                "test_name": "cpp_types_token",
                "token_id": cpp_token_id,
                "token": token,
                "success": True,
                "result": result
            }

        except Exception as e:
            logger.error(f"Error testing CppTypeDefinitionSizes token: {e}")
            return {
                "test_name": "cpp_types_token",
                "token_id": cpp_token_id,
                "error": str(e),
                "success": False
            }

    def test_token_list(self, token_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Test a list of token IDs for domain extraction issues

        Args:
            token_ids: List of token IDs to test

        Returns:
            List of test results
        """
        logger.info(f"Testing {len(token_ids)} tokens for domain extraction issues...")

        results = []
        for i, token_id in enumerate(token_ids):
            try:
                result = self.classifier.test_domain_extraction(token_id)
                results.append(result)

                # Log immediate result
                if result.get("breaks_domain_extraction", False):
                    issues = ", ".join(result.get("issues", []))
                    logger.warning(f"  ❌ Token '{result['token']}' breaks domain extraction: {issues}")
                else:
                    logger.info(f"  ✅ Token '{result['token']}' does not break domain extraction")

            except Exception as e:
                logger.error(f"  ❌ Error testing token {token_id}: {e}")
                results.append({
                    "token_id": token_id,
                    "error": str(e),
                    "breaks_domain_extraction": True
                })

            # Progress update
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(token_ids)} tokens tested")

        return results

    def test_normal_tokens(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Test some normal tokens as a control group

        Args:
            count: Number of normal tokens to test

        Returns:
            List of test results for normal tokens
        """
        logger.info(f"Testing {count} normal tokens as control group...")

        # Common normal tokens that should not break domain extraction
        normal_tokens = ["computer", "hello", "world", "test", "domain", "network", "server", "email", "system", "data"]

        results = []
        for i, token_str in enumerate(normal_tokens[:count]):
            try:
                # Get token IDs for this token
                token_ids = self.classifier.tokenizer.encode(token_str, add_special_tokens=False)
                if token_ids:
                    token_id = token_ids[0]

                    # Test domain extraction
                    result = self.classifier.test_domain_extraction(token_id)

                    test_result = {
                        "test_name": "normal_token",
                        "token_string": token_str,
                        "token_id": token_id,
                        "success": True,
                        "result": result
                    }
                    results.append(test_result)

                    # Log result
                    if result.get("breaks_domain_extraction", False):
                        logger.warning(f"Normal token '{token_str}' unexpectedly breaks domain extraction")
                    else:
                        logger.info(f"Normal token '{token_str}' works correctly for domain extraction")

            except Exception as e:
                logger.error(f"Error testing normal token '{token_str}': {e}")
                results.append({
                    "test_name": "normal_token",
                    "token_string": token_str,
                    "error": str(e),
                    "success": False
                })

        return results

    def _load_token_file(self, token_file: str) -> List[int]:
        """
        Load token IDs from a JSON file

        Args:
            token_file: Path to the JSON file containing token IDs

        Returns:
            List of token IDs
        """
        try:
            with open(token_file, 'r') as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Look for common keys that might contain token IDs
                for key in ['token_ids', 'tokens', 'glitch_tokens', 'ids']:
                    if key in data:
                        return data[key]
                # If no standard key found, try to extract from values
                for value in data.values():
                    if isinstance(value, list) and all(isinstance(x, int) for x in value):
                        return value

            logger.error(f"Could not extract token IDs from {token_file}")
            return []

        except Exception as e:
            logger.error(f"Error loading token file {token_file}: {e}")
            return []

    def _generate_summary(self, results: Dict[str, Any]) -> None:
        """
        Generate and log a summary of test results

        Args:
            results: Complete test results dictionary
        """
        logger.info("\n" + "="*80)
        logger.info("DOMAIN EXTRACTION TEST SUMMARY")
        logger.info("="*80)

        total_tested = 0
        total_broken = 0
        total_valid_domains = 0
        extra_important = 0

        # Analyze CppTypeDefinitionSizes test
        if "cpp_types_test" in results:
            cpp_test = results["cpp_types_test"]
            if cpp_test.get("success", False):
                cpp_result = cpp_test["result"]
                total_tested += 1
                if cpp_result.get("breaks_domain_extraction", False):
                    total_broken += 1
                    issues = ", ".join(cpp_result.get("issues", []))
                    logger.info(f"❌ CppTypeDefinitionSizes: BREAKS extraction ({issues})")

                    if cpp_result.get("creates_valid_domain", False):
                        total_valid_domains += 1
                        extra_important += 1
                        logger.info(f"⭐ CppTypeDefinitionSizes: Creates VALID domain AND breaks extraction!")
                else:
                    logger.info(f"✅ CppTypeDefinitionSizes: Does NOT break extraction")

                if cpp_result.get("creates_valid_domain", False):
                    total_valid_domains += 1

        # Analyze token list tests
        if "token_list_tests" in results:
            for result in results["token_list_tests"]:
                if "error" not in result:
                    total_tested += 1
                    if result.get("breaks_domain_extraction", False):
                        total_broken += 1
                        issues = ", ".join(result.get("issues", []))
                        token_str = result.get("token", f"ID:{result.get('token_id', 'unknown')}")
                        logger.info(f"❌ {token_str}: BREAKS extraction ({issues})")

                        if result.get("creates_valid_domain", False):
                            total_valid_domains += 1
                            extra_important += 1
                            logger.info(f"⭐ {token_str}: Creates VALID domain AND breaks extraction!")
                    else:
                        token_str = result.get("token", f"ID:{result.get('token_id', 'unknown')}")
                        logger.info(f"✅ {token_str}: Does NOT break extraction")

                    if result.get("creates_valid_domain", False):
                        total_valid_domains += 1

        # Analyze normal token tests
        normal_broken = 0
        if "normal_tokens_tests" in results:
            for test_result in results["normal_tokens_tests"]:
                if test_result.get("success", False):
                    result = test_result["result"]
                    if result.get("breaks_domain_extraction", False):
                        normal_broken += 1
                        token_str = test_result.get("token_string", "unknown")
                        logger.warning(f"⚠️  Normal token '{token_str}' breaks domain extraction (unexpected)")

        logger.info("="*80)
        logger.info("FINAL STATISTICS")
        logger.info(f"Total tokens tested: {total_tested}")
        if total_tested > 0:
            logger.info(f"Tokens breaking domain extraction: {total_broken} ({total_broken/total_tested*100:.1f}%)")
        else:
            logger.info("No tokens tested")
        logger.info(f"Tokens creating valid domains: {total_valid_domains}")
        logger.info(f"⭐ EXTRA IMPORTANT - Valid domains that break extraction: {extra_important}")

        if normal_broken > 0:
            logger.warning(f"⚠️  Normal tokens that break extraction: {normal_broken} (may indicate model/prompt issues)")
        else:
            logger.info("✅ All normal tokens work correctly")

        logger.info("="*80)

    def run_tests(self, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Run the complete test suite

        Args:
            args: Command line arguments

        Returns:
            Complete test results
        """
        results = {
            "model_path": self.model_path,
            "test_timestamp": time.time(),
            "test_type": "domain_extraction",
            "args": vars(args)
        }

        # Test 1: Known CppTypeDefinitionSizes token
        if args.test_cpptypes:
            logger.info("\n--- Testing Known CppTypeDefinitionSizes Token ---")
            results["cpp_types_test"] = self.test_known_cpp_types_token()

        # Test 2: Token list from command line or file
        token_ids = []
        if args.token_ids:
            token_ids = [int(x.strip()) for x in args.token_ids.split(',')]
        elif args.token_file:
            token_ids = self._load_token_file(args.token_file)

        if token_ids:
            logger.info(f"\n--- Testing {len(token_ids)} Specified Tokens ---")
            results["token_list_tests"] = self.test_token_list(token_ids)

        # Test 3: Normal tokens as control
        if not args.skip_normal:
            logger.info("\n--- Testing Normal Tokens (Control Group) ---")
            results["normal_tokens_tests"] = self.test_normal_tokens(args.normal_count)

        # Generate summary
        self._generate_summary(results)

        return results


def main():
    """Main function for the domain extraction test script"""
    parser = argparse.ArgumentParser(
        description="Test domain extraction functionality with glitch tokens"
    )

    parser.add_argument(
        "model_path",
        help="Path or name of the model to test"
    )
    parser.add_argument(
        "--token-ids", type=str,
        help="Comma-separated list of token IDs to test"
    )
    parser.add_argument(
        "--token-file", type=str,
        help="JSON file containing token IDs to test"
    )
    parser.add_argument(
        "--test-cpptypes", action="store_true",
        help="Test the known CppTypeDefinitionSizes glitch token"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use for inference (default: cuda)"
    )
    parser.add_argument(
        "--quant-type", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "int8", "int4"],
        help="Quantization type (default: bfloat16)"
    )
    parser.add_argument(
        "--output", type=str, default="domain_extraction_results.json",
        help="Output file for results (default: domain_extraction_results.json)"
    )
    parser.add_argument(
        "--skip-normal", action="store_true",
        help="Skip testing normal tokens as control group"
    )
    parser.add_argument(
        "--normal-count", type=int, default=5,
        help="Number of normal tokens to test as control (default: 5)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not any([args.token_ids, args.token_file, args.test_cpptypes]):
        parser.error("Must specify at least one of: --token-ids, --token-file, or --test-cpptypes")

    try:
        # Initialize tester
        tester = DomainExtractionTester(
            model_path=args.model_path,
            device=args.device,
            quant_type=args.quant_type
        )

        # Run tests
        results = tester.run_tests(args)

        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to: {args.output}")
        logger.info("Domain extraction testing completed!")

    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise


if __name__ == "__main__":
    main()
