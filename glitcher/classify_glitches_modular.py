#!/usr/bin/env python3
"""
Backward-compatible glitch token classifier using modular components

This module provides the same interface as the original classify_glitches.py but
uses the new modular architecture under the hood. This ensures backward compatibility
while taking advantage of the improved code organization.
"""

import argparse
import json
from typing import List, Dict, Any

from .classification.glitch_classifier import GlitchClassifier
from .classification.types import TestConfig
from .utils import get_logger

# For backward compatibility, expose the original classes
TqdmLoggingHandler = None  # Will be imported from utils if needed
logger = get_logger("GlitchClassifier")


class ClassificationWrapper:
    """Wrapper class that provides the original interface using modular components"""

    def __init__(self, args=None):
        self.args = args
        self.classifier = None

    def setup_parser(self):
        """Setup command line argument parser (same as original)"""
        parser = argparse.ArgumentParser(
            description="Glitch Token Classifier - Categorize glitch tokens by their effects"
        )

        parser.add_argument("model_path", help="Path or name of the model to use")
        parser.add_argument(
            "--token-ids", type=str,
            help="Comma-separated list of token IDs to classify"
        )
        parser.add_argument(
            "--token-file", type=str,
            help="JSON file containing token IDs to classify"
        )
        parser.add_argument(
            "--output", type=str, default="classified_tokens.json",
            help="Output file for results (default: classified_tokens.json)"
        )
        parser.add_argument(
            "--device", type=str, default="cuda",
            help="Device to use (default: cuda)"
        )
        parser.add_argument(
            "--quant-type", type=str, default="bfloat16",
            choices=["bfloat16", "float16", "int8", "int4"],
            help="Quantization type (default: bfloat16)"
        )
        parser.add_argument(
            "--temperature", type=float, default=0.0,
            help="Temperature for model inference (default: 0.0)"
        )
        parser.add_argument(
            "--max-tokens", type=int, default=200,
            help="Maximum tokens to generate per test (default: 200)"
        )
        parser.add_argument(
            "--skip-baseline", action="store_true",
            help="Skip baseline tests on standard tokens"
        )
        parser.add_argument(
            "--prompt-comparison-only", action="store_true",
            help="Only run prompt comparison tests without full classification"
        )
        parser.add_argument(
            "--email-extraction-only", action="store_true",
            help="Only run email extraction tests without full classification"
        )
        parser.add_argument(
            "--domain-extraction-only", action="store_true",
            help="Only run domain extraction tests without full classification"
        )
        parser.add_argument(
            "--debug-responses", action="store_true",
            help="Enable detailed response logging for debugging"
        )
        parser.add_argument(
            "--simple-template", action="store_true",
            help="Use simple chat template without system prompt (for testing corrupted tokens)"
        )

        return parser

    def load_model(self):
        """Load the model and tokenizer"""
        config = TestConfig.from_args(self.args)

        self.classifier = GlitchClassifier(
            model_path=self.args.model_path,
            device=self.args.device,
            quant_type=self.args.quant_type,
            config=config
        )

        self.classifier.load_model()

    def get_token_ids(self) -> List[int]:
        """Get token IDs to classify from command line arguments"""
        token_ids = []

        if self.args.token_ids:
            try:
                token_ids = [int(tid.strip()) for tid in self.args.token_ids.split(",")]
            except ValueError:
                logger.error("Error: Token IDs must be comma-separated integers")
                return []

        elif self.args.token_file:
            try:
                with open(self.args.token_file, "r") as f:
                    data = json.load(f)
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
                    else:
                        logger.error("Error: Could not find token IDs in the file")
                        return []
            except Exception as e:
                logger.error(f"Error loading token file: {e}")
                return []

        if not token_ids:
            logger.error("Error: No token IDs provided")
            return []

        return token_ids

    def run_prompting_tests(self):
        """Test if the model tends to repeat for all tokens, not just glitch tokens"""
        logger.info("Prompting tests not yet implemented in modular version")
        # TODO: Implement using PromptTester module when created
        pass

    def run_baseline_tests(self):
        """Run baseline tests on standard tokens"""
        if self.args.skip_baseline:
            logger.info("Skipping baseline tests as requested")
            return

        logger.info("Baseline tests not yet implemented in modular version")
        # TODO: Implement using BaselineTester module when created
        pass

    def test_email_extraction(self, token_id: int) -> Dict[str, Any]:
        """Test if a token breaks email extraction functionality"""
        # This is handled by the EmailTester in the classifier
        token = self.classifier.tokenizer.decode([token_id])
        return self.classifier.email_tester.test_email_extraction(
            token_id, token, self.classifier.model, self.classifier.tokenizer,
            self.classifier.chat_template, self.classifier.format_prompt
        )

    def test_domain_extraction(self, token_id: int) -> Dict[str, Any]:
        """Test if a token breaks domain extraction functionality"""
        # This would use DomainTester when implemented
        return self.classifier._test_domain_extraction(
            token_id, self.classifier.tokenizer.decode([token_id])
        )

    def classify_token(self, token_id: int) -> Dict[str, Any]:
        """Run all classification tests on a token and determine its categories"""
        result = self.classifier.classify_token(token_id)

        # Convert to original format for backward compatibility
        return {
            "token_id": result.token_id,
            "token": result.token,
            "test_results": [test_result.to_dict() for test_result in result.test_results],
            "categories": result.categories,
            "timestamp": result.timestamp
        }

    def run(self):
        """Run the classifier (main entry point)"""
        # Parse arguments
        parser = self.setup_parser()
        self.args = parser.parse_args()

        # Print banner
        logger.info(f"Starting glitch token classification on {self.args.model_path}")

        # Load model
        self.load_model()

        # Handle special modes
        if hasattr(self.args, 'prompt_comparison_only') and self.args.prompt_comparison_only:
            self.run_prompting_tests()
            return

        if hasattr(self.args, 'email_extraction_only') and self.args.email_extraction_only:
            token_ids = self.get_token_ids()
            if not token_ids:
                return

            summary = self.classifier.run_email_extraction_only(token_ids)

            # Save results
            output_file = self.args.output.replace('.json', '_email_extraction.json')
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Email extraction results saved to {output_file}")
            return

        if hasattr(self.args, 'domain_extraction_only') and self.args.domain_extraction_only:
            token_ids = self.get_token_ids()
            if not token_ids:
                return

            summary = self.classifier.run_domain_extraction_only(token_ids)

            # Save results
            output_file = self.args.output.replace('.json', '_domain_extraction.json')
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Domain extraction results saved to {output_file}")
            return

        # Get token IDs to classify
        token_ids = self.get_token_ids()
        if not token_ids:
            return

        # Run baseline tests
        self.run_baseline_tests()

        logger.info(f"Classifying {len(token_ids)} tokens...")

        # Run classification using modular classifier
        results = self.classifier.classify_tokens(token_ids)

        # Convert results to original format
        classifications = []
        for result in results:
            classification = {
                "token_id": result.token_id,
                "token": result.token,
                "test_results": [test_result.to_dict() for test_result in result.test_results],
                "categories": result.categories,
                "timestamp": result.timestamp
            }
            classifications.append(classification)

        # Prepare summary
        summary = self.classifier.get_results_summary()
        summary["classifications"] = classifications

        # Save results
        self._save_results(summary)

        # Print summary table
        self.classifier.print_summary_table()

    def _save_results(self, summary: Dict[str, Any]):
        """Save classification results to file"""
        try:
            with open(self.args.output, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Classification results saved to {self.args.output}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    """Main entry point for the classifier"""
    classifier = ClassificationWrapper()
    classifier.run()


if __name__ == "__main__":
    main()
