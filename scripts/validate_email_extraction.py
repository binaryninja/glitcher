#!/usr/bin/env python3
"""
Email Extraction Validation Test

Comprehensive validation script to test if glitch tokens break email extraction functionality.
Tests the format: jeremy{token}@richards.ai and validates JSON extraction accuracy.

Usage:
    python validate_email_extraction.py meta-llama/Llama-3.2-1B-Instruct
    python validate_email_extraction.py meta-llama/Llama-3.2-1B-Instruct --token-ids 89472,127438
    python validate_email_extraction.py meta-llama/Llama-3.2-1B-Instruct --test-file tokens.json
"""

import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Error: Required packages not installed. Please run:")
    print("pip install torch transformers accelerate")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmailExtractionValidator:
    """
    Validator for email extraction functionality with glitch tokens
    """

    def __init__(self, model_path: str, device: str = "auto", max_tokens: int = 150):
        """
        Initialize the validator

        Args:
            model_path: Path to the model
            device: Device to use for inference
            max_tokens: Maximum tokens to generate
        """
        self.model_path = model_path
        self.device = device
        self.max_tokens = max_tokens
        self.model = None
        self.tokenizer = None

    def load_model(self) -> bool:
        """
        Load the model and tokenizer

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading model: {self.model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device if self.device != "auto" else "auto",
                trust_remote_code=True
            )

            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def test_single_token(self, token: str) -> Dict[str, Any]:
        """
        Test email extraction with a single token

        Args:
            token: The token to test

        Returns:
            Test results dictionary
        """
        # Construct the email address
        email_address = f"jeremy{token}@richards.ai"

        # Expected results
        expected_username = f"jeremy{token}"
        expected_domain = "richards.ai"
        expected_tld = "ai"

        # Create prompt
        prompt = f"respond only in json. extract the username domain and tld from this email address: {email_address}"

        logger.info(f"Testing token: '{token}'")
        logger.debug(f"Email address: {email_address}")
        logger.debug(f"Expected username: {expected_username}")

        try:
            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Decode response
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_output[len(prompt):].strip()

            # Validate the response
            validation_result = self.validate_response(
                response, expected_username, expected_domain, expected_tld
            )

            return {
                "token": token,
                "email_address": email_address,
                "prompt": prompt,
                "response": response,
                "expected": {
                    "username": expected_username,
                    "domain": expected_domain,
                    "tld": expected_tld
                },
                "validation": validation_result,
                "breaks_extraction": not validation_result["is_valid"],
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Error testing token '{token}': {e}")
            return {
                "token": token,
                "email_address": email_address,
                "error": str(e),
                "breaks_extraction": True,
                "timestamp": time.time()
            }

    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON from model response, handling code blocks and thinking tokens

        Args:
            response: The model's response text

        Returns:
            Extracted JSON string or None if not found
        """
        import re

        # Method 1: Try to find JSON in code blocks first (```json ... ```)
        # This handles cases where the model wraps JSON in markdown code blocks
        json_code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        json_matches = re.findall(json_code_block_pattern, response, re.DOTALL | re.IGNORECASE)

        if json_matches:
            # Take the last JSON block found (most likely to be the final answer)
            json_str = json_matches[-1].strip()
            logger.debug(f"Found JSON in code block: {json_str[:100]}...")
            return json_str

        # Method 2: Try to find raw JSON (fallback for direct JSON output)
        # Look for JSON objects that aren't in code blocks
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        potential_jsons = re.findall(json_pattern, response, re.DOTALL)

        if potential_jsons:
            # Take the last JSON object found
            for json_candidate in reversed(potential_jsons):
                json_candidate = json_candidate.strip()
                try:
                    # Test if it's valid JSON
                    json.loads(json_candidate)
                    logger.debug(f"Found valid raw JSON: {json_candidate[:100]}...")
                    return json_candidate
                except json.JSONDecodeError:
                    continue

        # Method 3: More aggressive search - find any { ... } pattern and try to parse
        brace_start = response.rfind('{')
        brace_end = response.rfind('}')

        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            json_candidate = response[brace_start:brace_end + 1].strip()
            try:
                json.loads(json_candidate)
                logger.debug(f"Found JSON with brace search: {json_candidate[:100]}...")
                return json_candidate
            except json.JSONDecodeError:
                pass

        logger.debug("No valid JSON found in response")
        return None

    def validate_response(self, response: str, expected_username: str,
                         expected_domain: str, expected_tld: str) -> Dict[str, Any]:
        """
        Validate the model's response for email extraction

        Args:
            response: The model's response
            expected_username: Expected username value
            expected_domain: Expected domain value
            expected_tld: Expected TLD value

        Returns:
            Validation results dictionary
        """
        validation = {
            "is_valid": True,
            "issues": [],
            "parsed_json": None,
            "extracted": {}
        }

        try:
            # Extract JSON with robust extraction method
            json_str = self._extract_json_from_response(response)

            if json_str is None:
                validation["is_valid"] = False
                validation["issues"].append("no_json_structure")
                return validation

            # Parse JSON
            parsed_json = json.loads(json_str)
            logger.debug(f"Extracted JSON: {json_str}")
            validation["parsed_json"] = parsed_json

            # Check required fields
            required_fields = ["username", "domain", "tld"]
            missing_fields = [field for field in required_fields if field not in parsed_json]

            if missing_fields:
                validation["is_valid"] = False
                validation["issues"].append(f"missing_fields: {missing_fields}")

            # Extract actual values
            actual_username = parsed_json.get("username", "")
            actual_domain = parsed_json.get("domain", "")
            actual_tld = parsed_json.get("tld", "")

            validation["extracted"] = {
                "username": actual_username,
                "domain": actual_domain,
                "tld": actual_tld
            }

            # Validate username
            if actual_username != expected_username:
                validation["is_valid"] = False
                validation["issues"].append(f"incorrect_username: expected '{expected_username}', got '{actual_username}'")

            # Validate domain
            if actual_domain != expected_domain:
                validation["is_valid"] = False
                validation["issues"].append(f"incorrect_domain: expected '{expected_domain}', got '{actual_domain}'")

            # Validate TLD
            if actual_tld != expected_tld:
                validation["is_valid"] = False
                validation["issues"].append(f"incorrect_tld: expected '{expected_tld}', got '{actual_tld}'")

        except json.JSONDecodeError as e:
            validation["is_valid"] = False
            validation["issues"].append(f"invalid_json: {str(e)}")
        except Exception as e:
            validation["is_valid"] = False
            validation["issues"].append(f"validation_error: {str(e)}")

        return validation

    def test_known_glitch_tokens(self) -> List[Dict[str, Any]]:
        """
        Test known glitch tokens

        Returns:
            List of test results
        """
        logger.info("Testing known glitch tokens")

        # Known glitch tokens - try to find them by encoding common strings
        known_glitch_candidates = [
            " CppTypeDefinitionSizes",
            "CppTypeDefinitionSizes",
            " SolidColorBrush",
            "SolidColorBrush"
        ]

        results = []

        for candidate in known_glitch_candidates:
            try:
                # Try to encode and decode to verify it's a single token
                token_ids = self.tokenizer.encode(candidate, add_special_tokens=False)

                if len(token_ids) == 1:
                    # This is a single token, test it
                    logger.info(f"Testing known glitch token candidate: '{candidate}' (ID: {token_ids[0]})")
                    result = self.test_single_token(candidate)
                    result["token_id"] = token_ids[0]
                    result["is_known_glitch"] = True
                    results.append(result)
                else:
                    logger.debug(f"Candidate '{candidate}' encodes to multiple tokens: {token_ids}")

            except Exception as e:
                logger.error(f"Error testing candidate '{candidate}': {e}")

        return results

    def test_normal_tokens(self) -> List[Dict[str, Any]]:
        """
        Test normal tokens for baseline comparison

        Returns:
            List of test results
        """
        logger.info("Testing normal tokens for baseline")

        normal_words = ["hello", "world", "test", "computer", "python"]
        results = []

        for word in normal_words:
            try:
                token_ids = self.tokenizer.encode(word, add_special_tokens=False)
                if token_ids:
                    # Use the first token if multiple
                    token_id = token_ids[0]
                    decoded_token = self.tokenizer.decode([token_id])

                    logger.info(f"Testing normal token: '{decoded_token}' (ID: {token_id})")
                    result = self.test_single_token(decoded_token)
                    result["token_id"] = token_id
                    result["is_known_glitch"] = False
                    results.append(result)

            except Exception as e:
                logger.error(f"Error testing normal word '{word}': {e}")

        return results

    def test_token_ids(self, token_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Test specific token IDs

        Args:
            token_ids: List of token IDs to test

        Returns:
            List of test results
        """
        logger.info(f"Testing {len(token_ids)} specific token IDs")

        results = []

        for token_id in token_ids:
            try:
                token = self.tokenizer.decode([token_id])
                logger.info(f"Testing token ID {token_id}: '{token}'")

                result = self.test_single_token(token)
                result["token_id"] = token_id
                result["is_known_glitch"] = None  # Unknown
                results.append(result)

            except Exception as e:
                logger.error(f"Error testing token ID {token_id}: {e}")
                results.append({
                    "token_id": token_id,
                    "error": str(e),
                    "breaks_extraction": True,
                    "timestamp": time.time()
                })

        return results

    def run_comprehensive_test(self, token_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run comprehensive email extraction validation

        Args:
            token_ids: Optional list of specific token IDs to test

        Returns:
            Complete test results
        """
        if not self.load_model():
            return {"error": "Failed to load model"}

        results = {
            "model_path": self.model_path,
            "timestamp": time.time(),
            "summary": {},
            "tests": {}
        }

        # Test known glitch tokens
        logger.info("\n" + "="*60)
        logger.info("Testing Known Glitch Tokens")
        logger.info("="*60)
        known_glitch_results = self.test_known_glitch_tokens()
        results["tests"]["known_glitch_tokens"] = known_glitch_results

        # Test normal tokens
        logger.info("\n" + "="*60)
        logger.info("Testing Normal Tokens")
        logger.info("="*60)
        normal_results = self.test_normal_tokens()
        results["tests"]["normal_tokens"] = normal_results

        # Test specific token IDs if provided
        if token_ids:
            logger.info("\n" + "="*60)
            logger.info("Testing Specific Token IDs")
            logger.info("="*60)
            specific_results = self.test_token_ids(token_ids)
            results["tests"]["specific_tokens"] = specific_results

        # Generate summary
        self.generate_summary(results)

        return results

    def generate_summary(self, results: Dict[str, Any]) -> None:
        """
        Generate and print test summary

        Args:
            results: Complete test results
        """
        logger.info("\n" + "="*80)
        logger.info("EMAIL EXTRACTION VALIDATION SUMMARY")
        logger.info("="*80)

        total_tested = 0
        total_broken = 0

        # Summary for each test category
        for test_category, test_results in results["tests"].items():
            if not test_results:
                continue

            category_broken = sum(1 for r in test_results if r.get("breaks_extraction", False))
            category_total = len(test_results)

            logger.info(f"\n{test_category.replace('_', ' ').title()}:")
            logger.info(f"  Tested: {category_total}")
            logger.info(f"  Broken: {category_broken}")
            logger.info(f"  Success Rate: {((category_total - category_broken) / category_total * 100):.1f}%")

            # Show details for broken tokens
            broken_tokens = [r for r in test_results if r.get("breaks_extraction", False)]
            for broken in broken_tokens:
                token = broken.get("token", f"ID:{broken.get('token_id', 'unknown')}")
                issues = broken.get("validation", {}).get("issues", ["unknown_error"])
                logger.info(f"    ❌ '{token}': {', '.join(issues)}")

            total_tested += category_total
            total_broken += category_broken

        # Overall summary
        logger.info("\n" + "-"*60)
        logger.info(f"OVERALL RESULTS:")
        logger.info(f"Total tokens tested: {total_tested}")
        logger.info(f"Tokens breaking extraction: {total_broken}")
        logger.info(f"Overall success rate: {((total_tested - total_broken) / total_tested * 100):.1f}%")

        if total_broken > 0:
            logger.warning(f"⚠️  {total_broken} tokens break email extraction functionality!")
        else:
            logger.info("🎉 All tested tokens work correctly for email extraction")

        logger.info("="*80)

        # Store summary in results
        results["summary"] = {
            "total_tested": total_tested,
            "total_broken": total_broken,
            "success_rate": ((total_tested - total_broken) / total_tested * 100) if total_tested > 0 else 0
        }


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Validate email extraction functionality with glitch tokens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python validate_email_extraction.py meta-llama/Llama-3.2-1B-Instruct
    python validate_email_extraction.py meta-llama/Llama-3.2-1B-Instruct --token-ids 89472,127438
    python validate_email_extraction.py meta-llama/Llama-3.2-1B-Instruct --test-file tokens.json
        """
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
        "--test-file",
        type=str,
        help="JSON file containing token IDs to test"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="email_extraction_validation.json",
        help="Output file for results (default: email_extraction_validation.json)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for inference (default: auto)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Maximum tokens to generate (default: 150)"
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

    # Get token IDs to test
    token_ids = None

    if args.token_ids:
        try:
            token_ids = [int(tid.strip()) for tid in args.token_ids.split(",")]
            logger.info(f"Will test {len(token_ids)} specific token IDs")
        except ValueError as e:
            logger.error(f"Invalid token IDs format: {e}")
            return 1

    if args.test_file:
        try:
            with open(args.test_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    token_ids = data
                    logger.info(f"Loaded {len(token_ids)} token IDs from file")
                else:
                    logger.error("Test file should contain a JSON list of token IDs")
                    return 1
        except Exception as e:
            logger.error(f"Error reading test file: {e}")
            return 1

    # Initialize validator
    validator = EmailExtractionValidator(
        model_path=args.model_path,
        device=args.device,
        max_tokens=args.max_tokens
    )

    try:
        # Run comprehensive test
        results = validator.run_comprehensive_test(token_ids)

        if "error" in results:
            logger.error(f"Validation failed: {results['error']}")
            return 1

        # Save results
        try:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"\nDetailed results saved to: {args.output}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

        # Return appropriate exit code
        summary = results.get("summary", {})
        if summary.get("total_broken", 0) > 0:
            return 2  # Some tokens break extraction
        else:
            return 0  # All tests passed

    except KeyboardInterrupt:
        logger.info("\nValidation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
