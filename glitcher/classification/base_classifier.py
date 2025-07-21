#!/usr/bin/env python3
"""
Base classifier for glitch token classification

This module provides the base classifier class with common functionality that can be
extended to create specialized classifiers. It handles model loading, prompt formatting,
test execution, and result management.
"""

import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import torch
from tqdm import tqdm

from .types import (
    GlitchCategory,
    ClassificationTest,
    TestResult,
    ClassificationResult,
    TestConfig
)
from ..utils import (
    get_logger,
    log_test_start,
    log_test_result,
    log_classification_summary,
    log_error,
    ProgressLogger
)
from ..model import (
    initialize_model_and_tokenizer,
    get_template_for_model,
    BuiltInTemplate
)


class BaseClassifier(ABC):
    """
    Base class for glitch token classifiers

    This class provides common functionality for loading models, running tests,
    and managing results. Subclasses should implement the create_tests() method
    to define their specific test suite.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        quant_type: str = "bfloat16",
        config: Optional[TestConfig] = None
    ):
        """
        Initialize the base classifier

        Args:
            model_path: Path or name of the model to use
            device: Device to use for model inference
            quant_type: Quantization type for model loading
            config: Test configuration settings
        """
        self.model_path = model_path
        self.device = device
        self.quant_type = quant_type
        self.config = config or TestConfig()

        # Model components (loaded lazily)
        self.model = None
        self.tokenizer = None
        self.chat_template = None

        # Test suite (created by subclasses)
        self.tests = []

        # Logger
        self.logger = get_logger(self.__class__.__name__)

        # Results tracking
        self.classification_results = []

    def load_model(self):
        """Load the model and tokenizer"""
        if self.model is not None:
            self.logger.info("Model already loaded")
            return

        self.logger.info(f"Loading model: {self.model_path}")

        with tqdm(total=0, desc="Initializing model", bar_format='{desc}: {elapsed}') as pbar:
            self.model, self.tokenizer = initialize_model_and_tokenizer(
                model_path=self.model_path,
                device=self.device,
                quant_type=self.quant_type
            )
            pbar.set_description(f"Model loaded on {self.device}")

        # Get chat template
        self.chat_template = get_template_for_model(
            self.model.config._name_or_path,
            self.tokenizer
        )

        self.logger.info("Model loading complete")

    @abstractmethod
    def create_tests(self) -> List[ClassificationTest]:
        """
        Create the test suite for this classifier

        Returns:
            List of classification tests
        """
        pass

    def initialize_tests(self):
        """Initialize the test suite"""
        if not self.tests:
            self.tests = self.create_tests()
            self.logger.info(f"Initialized {len(self.tests)} tests")

    def format_prompt(self, template_str: str, token: str) -> Tuple[str, str]:
        """
        Format a prompt with the chat template and token

        Args:
            template_str: Template string with {token} placeholder
            token: Token to insert into template

        Returns:
            Tuple of (formatted_input, original_prompt)
        """
        # Insert token into template
        user_prompt = template_str.format(token=token)

        # Check if we should use simple template
        if self.config.simple_template:
            # Use simple Llama 3.x format without system prompt
            formatted_input = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            self.logger.debug("Using simple template format")
        else:
            # Format with standard chat template
            if isinstance(self.chat_template, BuiltInTemplate):
                formatted_input = self.chat_template.format_chat("", user_prompt)
            elif self.chat_template is not None:
                system_format = getattr(self.chat_template, 'system_format', None)
                user_format = getattr(self.chat_template, 'user_format', None)
                formatted_system = system_format.format(content="") if system_format else ""
                formatted_user = user_format.format(content=user_prompt) if user_format else user_prompt
                formatted_input = formatted_system + formatted_user
            else:
                # Fallback if no chat template
                formatted_input = user_prompt

        return formatted_input, user_prompt

    def run_test(self, token_id: int, test: ClassificationTest) -> TestResult:
        """
        Run a single classification test on a token

        Args:
            token_id: ID of the token to test
            test: Classification test to run

        Returns:
            Test result
        """
        # Initialize variables for error handling scope
        token = self.tokenizer.decode([token_id]) if self.tokenizer else str(token_id)
        original_prompt = ""
        response = ""

        try:
            # Check that model and tokenizer are loaded
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model and tokenizer must be loaded before running tests")

            # Decode token to string
            token = self.tokenizer.decode([token_id])

            log_test_start(self.logger, token, token_id, test.name)

            # Format prompt using chat template
            formatted_input, original_prompt = self.format_prompt(test.template, token)

            # Tokenize prompt
            inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.model.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    do_sample=(self.config.temperature > 0),
                    temperature=self.config.temperature
                )

            # Decode response
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the assistant's response
            response = full_output[len(formatted_input):].strip()

            # Check each indicator
            triggered_indicators = {}
            for indicator_name, check_fn in test.indicators.items():
                is_triggered = check_fn(response)
                triggered_indicators[indicator_name] = is_triggered

                if is_triggered and self.config.enable_debug:
                    response_preview = response[:50] + "..." if len(response) > 50 else response
                    response_preview = response_preview.replace("\n", "\\n")
                    self.logger.debug(f"Token '{token}' triggered {test.category}::{indicator_name}")
                    self.logger.debug(f"  Response preview: '{response_preview}'")

            # Create result
            is_positive = any(triggered_indicators.values())
            result = TestResult(
                test_name=test.name,
                category=test.category,
                token_id=token_id,
                token=token,
                prompt=original_prompt,
                response=response,
                indicators=triggered_indicators,
                is_positive=is_positive,
                metadata={
                    "response_length": len(response),
                    "full_output_length": len(full_output)
                }
            )

            log_test_result(self.logger, token, test.name, test.category, is_positive, triggered_indicators)

            return result

        except Exception as e:
            error_str = str(e)
            # Check if this is a glitch-induced error (which should be detected as positive)
            if any(indicator in error_str.lower() for indicator in ['"username"', '"domain"', '"tld"', 'keyerror']):
                self.logger.info(f"🎯 GLITCH DETECTED in test {test.name} for token {token_id}: {error_str}")
                # For glitch-induced errors, mark as positive detection
                return TestResult(
                    test_name=test.name,
                    category=test.category,
                    token_id=token_id,
                    token=token,
                    prompt=original_prompt,
                    response=response,
                    indicators={"glitch_induced_error": True},
                    is_positive=True,
                    error=f"Glitch behavior detected: {error_str}",
                    metadata={"glitch_induced_error": True, "error_type": error_str}
                )
            else:
                # For actual code errors, log and mark as negative
                log_error(self.logger, f"Code error in test {test.name} for token {token_id}", e)
                return TestResult(
                    test_name=test.name,
                    category=test.category,
                    token_id=token_id,
                    token=token,
                    prompt="",
                    response="",
                    indicators={},
                    is_positive=False,
                    error=str(e)
                )

    def classify_token(self, token_id: int) -> ClassificationResult:
        """
        Run all classification tests on a token and determine its categories

        Args:
            token_id: ID of the token to classify

        Returns:
            Complete classification result
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be loaded before classifying tokens")

        token = self.tokenizer.decode([token_id])
        self.logger.info(f"Classifying token: '{token}' (ID: {token_id})")

        # Set current token ID for detailed analysis (used by subclasses)
        self._current_token_id = token_id

        # Create classification result
        result = ClassificationResult(
            token_id=token_id,
            token=token,
            timestamp=time.time()
        )

        # Run each test
        for test in tqdm(self.tests, desc=f"Testing '{token}'", leave=False):
            test_result = self.run_test(token_id, test)
            result.add_test_result(test_result)

        # Handle special cases (can be overridden by subclasses)
        self._post_process_classification(result)

        # Clear current token ID
        self._current_token_id = None

        # If no categories were detected, mark as unknown
        if not result.categories:
            result.categories.append(GlitchCategory.UNKNOWN)

        log_classification_summary(self.logger, token, token_id, result.categories)

        return result

    def _post_process_classification(self, result: ClassificationResult):
        """
        Post-process classification results (can be overridden by subclasses)

        Args:
            result: Classification result to post-process
        """
        pass

    def classify_tokens(self, token_ids: List[int]) -> List[ClassificationResult]:
        """
        Classify multiple tokens

        Args:
            token_ids: List of token IDs to classify

        Returns:
            List of classification results
        """
        # Ensure model and tests are loaded
        if self.model is None:
            self.load_model()
        if not self.tests:
            self.initialize_tests()

        self.logger.info(f"Classifying {len(token_ids)} tokens...")

        results = []
        with ProgressLogger(self.logger, len(token_ids), "Classifying tokens", "tokens") as progress:
            for token_id in token_ids:
                result = self.classify_token(token_id)
                results.append(result)
                self.classification_results.append(result)
                progress.update(1, f"Classified token {result.token}")

        return results

    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a summary of classification results

        Returns:
            Dictionary with summary statistics
        """
        if not self.classification_results:
            return {"message": "No classification results available"}

        # Count categories
        category_counts = {}
        for result in self.classification_results:
            for category in result.categories:
                category_counts[category] = category_counts.get(category, 0) + 1

        # Count tokens by number of categories
        category_distribution = {}
        for result in self.classification_results:
            num_categories = len([c for c in result.categories if c != GlitchCategory.UNKNOWN])
            category_distribution[num_categories] = category_distribution.get(num_categories, 0) + 1

        return {
            "total_tokens": len(self.classification_results),
            "category_counts": category_counts,
            "category_distribution": category_distribution,
            "model_path": self.model_path,
            "config": {
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "simple_template": self.config.simple_template
            }
        }

    def save_results(self, output_file: str):
        """
        Save classification results to file

        Args:
            output_file: Path to output file
        """
        import json

        summary = self.get_results_summary()
        summary["classifications"] = [result.to_dict() for result in self.classification_results]
        summary["timestamp"] = time.time()

        try:
            with open(output_file, "w") as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"Results saved to {output_file}")
        except Exception as e:
            log_error(self.logger, f"Error saving results to {output_file}", e)

    def print_summary_table(self):
        """Print a summary table of classifications with detailed test failure information"""
        if not self.classification_results:
            self.logger.info("No results to display")
            return

        self.logger.info("\n" + "=" * 100)
        self.logger.info("CLASSIFICATION SUMMARY")
        self.logger.info("=" * 100)

        # Print detailed results for each token
        for i, result in enumerate(self.classification_results, 1):
            token_display = result.token.replace("\n", "\\n")[:30]

            self.logger.info(f"\n{i}. Token: '{token_display}' (ID: {result.token_id})")
            self.logger.info("-" * 60)

            if not result.categories or result.categories == [GlitchCategory.UNKNOWN]:
                self.logger.info("   No glitch categories detected")
            else:
                self.logger.info(f"   Categories: {', '.join(result.categories)}")

            # Group test results by category
            category_tests = {}
            for test_result in result.test_results:
                if test_result.category not in category_tests:
                    category_tests[test_result.category] = []
                category_tests[test_result.category].append(test_result)

            # Print test results by category
            for category in result.categories:
                if category in category_tests:
                    self.logger.info(f"\n   {category} Tests:")
                    for test_result in category_tests[category]:
                        if test_result.is_positive:
                            # Show which indicators triggered
                            triggered = [name for name, value in test_result.indicators.items() if value]
                            indicators_str = f" (triggered: {', '.join(triggered)})" if triggered else ""
                            self.logger.info(f"     ✅ {test_result.test_name}{indicators_str}")

                            # Show detailed analysis for email/domain extraction tests
                            if test_result.metadata and "detailed_analysis" in test_result.metadata:
                                detailed = test_result.metadata["detailed_analysis"]
                                if test_result.test_name == "email_extraction_test":
                                    self._print_detailed_email_analysis(detailed)
                                elif test_result.test_name == "domain_extraction_test":
                                    self._print_detailed_domain_analysis(detailed)

                            # Show response preview if debug enabled
                            elif self.config.enable_debug and test_result.response:
                                response_preview = test_result.response.replace("\n", " ")[:100]
                                self.logger.info(f"        Response: {response_preview}...")

        # Print summary statistics
        self.logger.info("\n" + "=" * 100)
        self.logger.info("SUMMARY STATISTICS")
        self.logger.info("=" * 100)

        # Count by category
        category_counts = {}
        for result in self.classification_results:
            for category in result.categories:
                category_counts[category] = category_counts.get(category, 0) + 1

        total_tokens = len(self.classification_results)
        self.logger.info(f"Total tokens tested: {total_tokens}")

        if category_counts:
            self.logger.info("\nTokens by category:")
            for category, count in sorted(category_counts.items()):
                percentage = (count / total_tokens) * 100
                self.logger.info(f"  {category:20}: {count:3d} ({percentage:5.1f}%)")

        # Show tokens with no categories
        no_category_count = sum(1 for r in self.classification_results
                               if not r.categories or r.categories == [GlitchCategory.UNKNOWN])
        if no_category_count > 0:
            percentage = (no_category_count / total_tokens) * 100
            self.logger.info(f"  {'No glitch detected':20}: {no_category_count:3d} ({percentage:5.1f}%)")

        self.logger.info("=" * 100)

    def _print_detailed_email_analysis(self, detailed: Dict[str, Any]):
        """Print detailed email extraction analysis"""
        self.logger.info(f"        Expected: {detailed.get('expected_email', 'N/A')}")

        # Handle glitch-induced errors as positive detections
        if detailed.get('glitch_induced_error', False):
            self.logger.info(f"        🎯 GLITCH DETECTED: {detailed.get('error_type', 'Unknown error')}")
            self.logger.info(f"        Behavior: {detailed.get('glitch_behavior', 'Token caused parsing failure')}")
        elif 'issues' in detailed and detailed['issues']:
            self.logger.info(f"        Issues: {', '.join(detailed['issues'])}")

        if detailed.get('response_preview'):
            self.logger.info(f"        Response: {detailed['response_preview']}")

    def _print_detailed_domain_analysis(self, detailed: Dict[str, Any]):
        """Print detailed domain extraction analysis"""
        self.logger.info(f"        Expected: {detailed.get('expected_domain', 'N/A')}")

        # Handle glitch-induced errors as positive detections
        if detailed.get('glitch_induced_error', False):
            self.logger.info(f"        🎯 GLITCH DETECTED: {detailed.get('error_type', 'Unknown error')}")
            self.logger.info(f"        Behavior: {detailed.get('glitch_behavior', 'Token caused parsing failure')}")
        elif 'issues' in detailed and detailed['issues']:
            self.logger.info(f"        Issues: {', '.join(detailed['issues'])}")

        if detailed.get('response_preview'):
            self.logger.info(f"        Response: {detailed['response_preview']}")

    def clear_results(self):
        """Clear stored classification results"""
        self.classification_results.clear()
        self.logger.info("Classification results cleared")
