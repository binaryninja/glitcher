#!/usr/bin/env python3
"""
Command-line interface for glitch token classification

This module provides the CLI interface for the modular glitch classification system,
allowing users to run various types of classification tests on glitch tokens.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from .glitch_classifier import GlitchClassifier
from .types import TestConfig, GlitchCategory
from ..utils import setup_logger


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Glitch Token Classifier - Categorize glitch tokens by their effects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify specific tokens
  %(prog)s meta-llama/Llama-3.2-1B-Instruct --token-ids 89472,127438,85069

  # Classify tokens from file
  %(prog)s meta-llama/Llama-3.2-1B-Instruct --token-file glitch_tokens.json

  # Run only email extraction tests
  %(prog)s meta-llama/Llama-3.2-1B-Instruct --token-ids 89472,127438 --email-extraction-only

  # Run with debug output
  %(prog)s meta-llama/Llama-3.2-1B-Instruct --token-ids 89472 --debug-responses

  # Use different model settings
  %(prog)s meta-llama/Llama-3.2-1B-Instruct --token-ids 89472 --temperature 0.1 --max-tokens 500
        """
    )

    # Required arguments
    parser.add_argument(
        "model_path",
        help="Path or name of the model to use for classification"
    )

    # Token input options
    token_group = parser.add_mutually_exclusive_group(required=True)
    token_group.add_argument(
        "--token-ids",
        type=str,
        help="Comma-separated list of token IDs to classify"
    )
    token_group.add_argument(
        "--token-file",
        type=str,
        help="JSON file containing token IDs to classify"
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="classified_tokens.json",
        help="Output file for results (default: classified_tokens.json)"
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for model inference (default: cuda)"
    )
    model_group.add_argument(
        "--quant-type",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "int8", "int4"],
        help="Quantization type for model loading (default: bfloat16)"
    )
    model_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model inference (default: 0.0)"
    )
    model_group.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate per test (default: 200)"
    )

    # Test configuration
    test_group = parser.add_argument_group("Test Configuration")
    test_group.add_argument(
        "--simple-template",
        action="store_true",
        help="Use simple chat template without system prompt"
    )
    test_group.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout for each test in seconds (default: 30.0)"
    )

    # Test mode options
    mode_group = parser.add_argument_group("Test Modes")
    mode_group.add_argument(
        "--email-extraction-only",
        action="store_true",
        help="Only run email extraction tests"
    )
    mode_group.add_argument(
        "--domain-extraction-only",
        action="store_true",
        help="Only run domain extraction tests"
    )
    mode_group.add_argument(
        "--behavioral-only",
        action="store_true",
        help="Only run behavioral tests (injection, IDOS, etc.)"
    )
    mode_group.add_argument(
        "--functional-only",
        action="store_true",
        help="Only run functional tests (email/domain extraction)"
    )

    # Validation and baseline options
    validation_group = parser.add_argument_group("Validation Options")
    validation_group.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline tests on standard tokens"
    )
    validation_group.add_argument(
        "--run-prompting-tests",
        action="store_true",
        help="Run prompting behavior analysis tests"
    )

    # Output and logging options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--debug-responses",
        action="store_true",
        help="Enable detailed response logging for debugging"
    )
    output_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    output_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    output_group.add_argument(
        "--no-summary-table",
        action="store_true",
        help="Skip printing the summary table"
    )
    output_group.add_argument(
        "--log-file",
        type=str,
        help="Path to log file (default: glitch_classifier.log)"
    )

    # Information options
    info_group = parser.add_argument_group("Information")
    info_group.add_argument(
        "--list-categories",
        action="store_true",
        help="List all available categories and exit"
    )
    info_group.add_argument(
        "--list-tests",
        action="store_true",
        help="List all available tests and exit"
    )
    info_group.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )

    return parser


def load_token_ids(args) -> List[int]:
    """Load token IDs from command line arguments or file"""
    token_ids = []

    if args.token_ids:
        try:
            token_ids = [int(tid.strip()) for tid in args.token_ids.split(",")]
        except ValueError as e:
            raise ValueError("Token IDs must be comma-separated integers") from e

    elif args.token_file:
        token_file = Path(args.token_file)
        if not token_file.exists():
            raise FileNotFoundError(f"Token file not found: {args.token_file}")

        try:
            with open(token_file, "r") as f:
                data = json.load(f)

            # Handle different file formats
            if isinstance(data, list):
                token_ids = data
            elif "glitch_token_ids" in data:
                token_ids = data["glitch_token_ids"]
            elif "validation_results" in data:
                # Extract validated glitch tokens
                token_ids = []
                for result in data["validation_results"]:
                    if result.get("is_glitch", False):
                        token_ids.append(result["token_id"])
            elif "tokens" in data:
                token_ids = data["tokens"]
            elif "classifications" in data:
                # Extract from previous classification results
                token_ids = [c["token_id"] for c in data["classifications"]]
            else:
                raise ValueError("Could not find token IDs in the file. Expected format: list of integers or object with 'glitch_token_ids', 'validation_results', 'tokens', or 'classifications' field")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in token file: {e}") from e
        except Exception as e:
            raise ValueError(f"Error loading token file: {e}") from e

    if not token_ids:
        raise ValueError("No token IDs provided")

    # Validate token IDs
    invalid_ids = [tid for tid in token_ids if not isinstance(tid, int) or tid < 0]
    if invalid_ids:
        raise ValueError(f"Invalid token IDs (must be non-negative integers): {invalid_ids}")

    return token_ids


def setup_logging(args):
    """Setup logging based on command line arguments"""
    import logging

    # Determine log level
    if args.quiet:
        console_level = logging.WARNING
    elif args.verbose or args.debug_responses:
        console_level = logging.DEBUG
    else:
        console_level = logging.INFO

    # Setup logger
    log_file = args.log_file if args.log_file else "glitch_classifier.log"
    return setup_logger(
        name="GlitchClassifier",
        log_file=log_file,
        console_level=console_level,
        enable_file_logging=True
    )


def print_categories():
    """Print all available categories"""
    print("Available Classification Categories:")
    print("=" * 50)

    categories = {
        GlitchCategory.INJECTION: "Prompt injection and jailbreaking attempts",
        GlitchCategory.IDOS: "Infinite output or denial-of-service behavior",
        GlitchCategory.HALLUCINATION: "Nonsensical or incoherent output generation",
        GlitchCategory.DISRUPTION: "Internal reasoning or logic disruption",
        GlitchCategory.BYPASS: "Filter or safety guardrail bypass",
        GlitchCategory.EMAIL_EXTRACTION: "Breaks email parsing/extraction functionality",
        GlitchCategory.VALID_EMAIL_ADDRESS: "Creates valid email addresses when inserted",
        GlitchCategory.DOMAIN_EXTRACTION: "Breaks domain extraction from log files",
        GlitchCategory.VALID_DOMAIN_NAME: "Creates valid domain names when inserted",
        GlitchCategory.UNKNOWN: "Unable to categorize the token's effects"
    }

    for category, description in categories.items():
        print(f"  {category:20} - {description}")

    print(f"\nBehavioral categories: {', '.join(GlitchCategory.behavioral_categories())}")
    print(f"Functional categories: {', '.join(GlitchCategory.functional_categories())}")
    print(f"Validity categories:   {', '.join(GlitchCategory.validity_categories())}")


def print_tests(classifier: GlitchClassifier):
    """Print all available tests"""
    print("Available Classification Tests:")
    print("=" * 50)

    test_descriptions = classifier.get_test_descriptions()
    category_descriptions = classifier.get_category_descriptions()

    for test_name, description in test_descriptions.items():
        # Find the category for this test
        category = None
        for test in classifier.tests:
            if test.name == test_name:
                category = test.category
                break

        category_desc = category_descriptions.get(category or "Unknown", "Unknown category")
        print(f"  {test_name:25} - {description}")
        print(f"    Category: {category} ({category_desc})")
        print()


def run_classification(args, logger):
    """Run the main classification process"""
    # Load token IDs
    try:
        token_ids = load_token_ids(args)
        logger.info(f"Loaded {len(token_ids)} token IDs to classify")
    except Exception as e:
        logger.error(f"Error loading token IDs: {e}")
        return 1

    # Create test configuration
    config = TestConfig.from_args(args)

    # Create classifier
    logger.info(f"Initializing classifier for model: {args.model_path}")
    classifier = GlitchClassifier(
        model_path=args.model_path,
        device=args.device,
        quant_type=args.quant_type,
        config=config
    )

    try:
        # Handle special test modes
        if args.email_extraction_only:
            logger.info("Running email extraction tests only")
            summary = classifier.run_email_extraction_only(token_ids)
            output_file = args.output.replace('.json', '_email_extraction.json')

        elif args.domain_extraction_only:
            logger.info("Running domain extraction tests only")
            summary = classifier.run_domain_extraction_only(token_ids)
            output_file = args.output.replace('.json', '_domain_extraction.json')

        else:
            # Run full classification
            if not args.skip_baseline:
                logger.info("Running baseline validation tests...")
                # TODO: Implement baseline tests when BaselineTester is created

            if args.run_prompting_tests:
                logger.info("Running prompting behavior tests...")
                # TODO: Implement prompting tests when PromptTester is created

            logger.info(f"Running full classification on {len(token_ids)} tokens...")

            # Filter tests based on mode
            if args.behavioral_only:
                # Only run behavioral tests
                original_tests = classifier.tests
                classifier.tests = [t for t in original_tests if t.category in GlitchCategory.behavioral_categories()]
                logger.info(f"Filtered to {len(classifier.tests)} behavioral tests")

            elif args.functional_only:
                # Only run functional tests
                original_tests = classifier.tests
                classifier.tests = [t for t in original_tests if t.category in GlitchCategory.functional_categories()]
                logger.info(f"Filtered to {len(classifier.tests)} functional tests")

            # Run classification
            results = classifier.classify_tokens(token_ids)

            # Get summary
            summary = classifier.get_results_summary()
            summary["classifications"] = [result.to_dict() for result in results]

            output_file = args.output

            # Print summary table unless disabled
            if not args.no_summary_table:
                classifier.print_summary_table()

            # Add detailed extraction results if available
            if hasattr(classifier, '_detailed_email_results') and classifier._detailed_email_results:
                summary["detailed_email_results"] = classifier._detailed_email_results
            if hasattr(classifier, '_detailed_domain_results') and classifier._detailed_domain_results:
                summary["detailed_domain_results"] = classifier._detailed_domain_results

        # Save results
        try:
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {e}")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Error during classification: {e}")
        if args.debug_responses:
            logger.exception("Full traceback:")
        return 1


def main():
    """Main entry point for the CLI"""
    parser = create_parser()
    args = parser.parse_args()

    # Handle information requests
    if args.list_categories:
        print_categories()
        return 0

    # Setup logging
    logger = setup_logging(args)

    # Handle test listing (requires classifier initialization)
    if args.list_tests:
        try:
            # Create a temporary classifier just to get test info
            config = TestConfig(max_tokens=1)  # Minimal config for info only
            temp_classifier = GlitchClassifier("dummy", config=config)
            temp_classifier.initialize_tests()
            print_tests(temp_classifier)
            return 0
        except Exception as e:
            logger.error(f"Error initializing classifier for test listing: {e}")
            return 1

    # Validate mutually exclusive test mode options
    test_modes = [
        args.email_extraction_only,
        args.domain_extraction_only,
        args.behavioral_only,
        args.functional_only
    ]
    if sum(test_modes) > 1:
        logger.error("Error: Only one test mode option can be specified at a time")
        return 1

    # Run classification
    return run_classification(args, logger)


if __name__ == "__main__":
    sys.exit(main())
