#!/usr/bin/env python3
"""
Email extraction tests for glitch token classification

This module provides tests for email extraction functionality, checking how
glitch tokens affect email parsing, validation, and JSON extraction tasks.
"""

from typing import List, Dict, Any, Optional
from ..classification.types import ClassificationTest, GlitchCategory, TestConfig
from ..utils import (
    get_logger,
    is_valid_email_token,
    validate_extracted_email_data,
    create_test_email_address,
    JSONExtractor
)


class EmailTester:
    """Handles email extraction and validation tests for glitch tokens"""

    def __init__(self, config: Optional[TestConfig] = None):
        """
        Initialize email tester

        Args:
            config: Test configuration settings
        """
        self.config = config or TestConfig()
        self.logger = get_logger(self.__class__.__name__)
        self.json_extractor = JSONExtractor()

        # Cache for test results
        self._last_test_result = None

    def create_email_extraction_test(self) -> ClassificationTest:
        """Create email extraction classification test"""
        return ClassificationTest(
            name="email_extraction_test",
            category=GlitchCategory.EMAIL_EXTRACTION,
            template="Your response must only be json in the following format and nothing else: {\"username\": \"extracted_username\", \"domain\": \"extracted_domain\", \"tld\": \"extracted_tld\"}. Now extract the username domain and tld from this email address: jeremy{token}@richards.ai",
            indicators={
                # Check for malformed JSON or incorrect extraction
                "broken_extraction": lambda response: (
                    # Response doesn't contain expected JSON structure
                    not all(key in response.lower() for key in ["username", "domain", "tld"]) or
                    # JSON structure is broken or incomplete
                    response.count('"') < 6 or  # Should have at least 6 quotes for proper JSON
                    # Contains very long unbroken strings indicating glitch
                    len([part for part in response.split() if len(part) > 50]) > 0
                ),
                # Check for TLD extraction failure
                "tld_extraction_failure": lambda response: (
                    "richards.ai" not in response.lower() or
                    '"tld": "ai"' not in response.lower()
                ),
                # Check for username extraction issues
                "username_extraction_issue": lambda response: (
                    # Username field should contain "jeremy" but may be malformed
                    '"username":' not in response.lower() or
                    ("jeremy" not in response.lower())
                )
            },
            description="Tests if token breaks email extraction functionality by inserting it into an email address"
        )

    def create_valid_email_test(self) -> ClassificationTest:
        """Create test for tokens that create valid email addresses"""
        return ClassificationTest(
            name="valid_email_address_test",
            category=GlitchCategory.VALID_EMAIL_ADDRESS,
            template="respond only in json. extract the username domain and tld from this email address: jeremy{token}@richards.ai",
            indicators={
                # This test is informational only, actual validation happens in test_email_extraction
                "creates_valid_email_address": lambda response: False
            },
            description="Identifies tokens that create valid email addresses when inserted"
        )

    def test_email_extraction(
        self,
        token_id: int,
        token: str,
        model,
        tokenizer,
        chat_template,
        format_prompt_fn
    ) -> Dict[str, Any]:
        """
        Test if a token breaks email extraction functionality

        Args:
            token_id: Token ID
            token: Token string
            model: The model instance
            tokenizer: The tokenizer instance
            chat_template: Chat template
            format_prompt_fn: Function to format prompts

        Returns:
            Dictionary with test analysis results
        """
        self.logger.info(f"Testing email extraction for token: '{token}' (ID: {token_id})")

        # Check if token creates a valid email address
        creates_valid_email = is_valid_email_token(token)
        email_address = create_test_email_address(token)

        if creates_valid_email:
            self.logger.info(f"⭐ Token '{token}' creates VALID email address: {email_address}")
        else:
            self.logger.info(f"Token '{token}' creates invalid email address (?): {email_address}")

        # Test email extraction with the token inserted into jeremy{token}@richards.ai
        email_test_prompt = f"respond only in json. extract the username domain and tld from this email address: {email_address}"

        try:
            # Format prompt using chat template
            formatted_input, original_prompt = format_prompt_fn(email_test_prompt, "")

            # Tokenize prompt
            inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)

            # Generate response
            import torch
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    do_sample=(self.config.temperature > 0),
                    temperature=self.config.temperature,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Debug: Check generated tokens
            input_length = inputs['input_ids'].shape[1]
            output_length = outputs.shape[1]
            new_tokens_generated = output_length - input_length

            if self.config.enable_debug:
                self.logger.info(f"=== TOKEN GENERATION DEBUG FOR '{token}' ===")
                self.logger.info(f"Input tokens: {input_length}")
                self.logger.info(f"Output tokens: {output_length}")
                self.logger.info(f"New tokens generated: {new_tokens_generated}")
                self.logger.info(f"Max tokens requested: {self.config.max_tokens}")

                # Show the new tokens that were generated
                if new_tokens_generated > 0:
                    new_token_ids = outputs[0][input_length:].tolist()
                    self.logger.info(f"New token IDs: {new_token_ids}")

                    # Check for early EOS
                    eos_token_id = tokenizer.eos_token_id
                    if eos_token_id in new_token_ids:
                        eos_position = new_token_ids.index(eos_token_id)
                        self.logger.info(f"🛑 EOS token ({eos_token_id}) found at position {eos_position} in new tokens")
                        self.logger.info(f"Generation stopped early after {eos_position + 1} tokens")
                    else:
                        self.logger.info(f"No EOS token found in generated tokens")

                    # Decode just the new tokens
                    new_tokens_text = tokenizer.decode(new_token_ids, skip_special_tokens=False)
                    self.logger.info(f"New tokens text: {repr(new_tokens_text)}")
                self.logger.info("=" * 50)

            # Decode response
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

            # Extract assistant response properly from chat format
            response = self._extract_assistant_response(full_output, formatted_input, token)

            # Debug logging for response analysis
            if self.config.enable_debug:
                self.logger.info(f"=== DEBUG RESPONSE FOR TOKEN '{token}' ===")
                self.logger.info(f"Formatted input: {repr(formatted_input)}")
                self.logger.info(f"Full model output: {repr(full_output)}")
                self.logger.info(f"Extracted response: {repr(response)}")
                self.logger.info(f"Full output length: {len(full_output)} chars")
                self.logger.info(f"Formatted input length: {len(formatted_input)} chars")
                self.logger.info(f"Extracted response length: {len(response)} chars")
                self.logger.info("=" * 50)
            else:
                self.logger.debug(f"Token '{token}' - Full output length: {len(full_output)} chars")
                self.logger.debug(f"Token '{token}' - Formatted input length: {len(formatted_input)} chars")
                self.logger.debug(f"Token '{token}' - Extracted response length: {len(response)} chars")

            # Analyze the response for email extraction issues
            analysis = self._analyze_email_response(
                token_id, token, original_prompt, response, full_output,
                formatted_input, creates_valid_email, email_address
            )

            # Store result for classification integration
            self._last_test_result = analysis

            # Log response preview for debugging
            self._log_response_preview(token, response, full_output, formatted_input, analysis)

            return analysis

        except Exception as e:
            self.logger.error(f"Error testing email extraction for token {token_id}: {e}")
            return {
                "token_id": token_id,
                "token": token,
                "error": str(e),
                "breaks_email_extraction": True,
                "issues": ["test_error"]
            }

    def _extract_assistant_response(self, full_output: str, formatted_input: str, token: str) -> str:
        """Extract the assistant's response from the full model output"""
        # Look for the assistant's response after <|start_header_id|>assistant<|end_header_id|>
        assistant_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        assistant_start = full_output.rfind(assistant_marker)

        if assistant_start != -1:
            # Extract everything after the assistant marker
            response_start = assistant_start + len(assistant_marker)
            response = full_output[response_start:]

            # Remove trailing <|eot_id|> token if present
            if response.endswith("<|eot_id|>"):
                response = response[:-len("<|eot_id|>")].strip()
            else:
                response = response.strip()

            self.logger.debug(f"Token '{token}' - Extracted assistant response using marker")
        else:
            # Fallback: Handle corrupted/truncated output where full_output is shorter than input
            if len(full_output) < len(formatted_input):
                self.logger.warning(f"Token '{token}' - Corrupted output detected! Full output ({len(full_output)} chars) shorter than input ({len(formatted_input)} chars)")
                response = full_output  # Use the entire corrupted output as response
            else:
                response = full_output[len(formatted_input):].strip()

            self.logger.warning(f"Token '{token}' - Could not find assistant marker, using fallback extraction")

        # If extraction produced an empty string but we know the model returned
        # content, fall back to the full_output so we can inspect it later.
        if not response and full_output:
            response = full_output.strip()
            self.logger.debug(f"Token '{token}' - Response empty after extraction, using full_output as fallback")

        return response

    def _analyze_email_response(
        self,
        token_id: int,
        token: str,
        prompt: str,
        response: str,
        full_output: str,
        formatted_input: str,
        creates_valid_email: bool,
        email_address: str
    ) -> Dict[str, Any]:
        """Analyze the response for email extraction issues"""
        analysis = {
            "token_id": token_id,
            "token": token,
            "prompt": prompt,
            "response": response,
            "response_length": len(response),
            "full_output": full_output,
            "full_output_length": len(full_output),
            "formatted_input_length": len(formatted_input),
            "is_corrupted_output": len(full_output) < len(formatted_input),
            "issues": [],
            "creates_valid_email": creates_valid_email,
            "email_address": email_address
        }

        # Expected values for validation
        expected_username = f"jeremy{token}"
        expected_domain = "richards.ai"
        expected_tld = "ai"

        # Check for corrupted output
        if analysis["is_corrupted_output"]:
            analysis["issues"].append("corrupted_model_output")
            self.logger.warning(f"Token '{token}' caused corrupted model output")

        # Validate the extracted email data
        validation_result = validate_extracted_email_data(
            response, expected_username, expected_domain, expected_tld
        )

        # Add validation issues to analysis
        if not validation_result["is_valid"]:
            analysis["issues"].extend(validation_result["issues"])

        # Overall classification
        analysis["breaks_email_extraction"] = len(analysis["issues"]) > 0

        # Check if this is EXTRA IMPORTANT (valid email AND breaks extraction)
        is_extra_important = (creates_valid_email and analysis["breaks_email_extraction"])

        if is_extra_important:
            analysis["issues"].append("creates_valid_email_address")
            self.logger.info(f"⭐ EXTRA IMPORTANT: Token '{token}' creates valid email address AND breaks extraction: {email_address}")

        if analysis["breaks_email_extraction"]:
            issues_str = ', '.join([i for i in analysis["issues"] if i != "creates_valid_email_address"])
            if is_extra_important:
                self.logger.info(f"Token '{token}' BREAKS email extraction - Issues: {issues_str}")
            else:
                self.logger.info(f"Token '{token}' BREAKS email extraction - Issues: {issues_str}")
        else:
            self.logger.info(f"Token '{token}' does NOT break email extraction")

        return analysis

    def _log_response_preview(self, token: str, response: str, full_output: str, formatted_input: str, analysis: Dict):
        """Log response preview for debugging"""
        if len(response) == 0:
            self.logger.warning(f"Token '{token}' generated empty response!")
            self.logger.warning(f"Full output was: {repr(full_output)}")
            self.logger.warning(f"Formatted input was: {repr(formatted_input[:200])}")
        elif analysis["is_corrupted_output"]:
            self.logger.warning(f"Token '{token}' generated corrupted output!")
            self.logger.warning(f"Full output: {repr(full_output)}")
            self.logger.warning(f"Expected input length: {len(formatted_input)}, got output length: {len(full_output)}")
        elif self.config.enable_debug:
            self.logger.info(f"Response preview: {repr(response[:100])}")
        else:
            response_preview = response[:100] + "..." if len(response) > 100 else response
            self.logger.debug(f"Email extraction response: {response_preview}")

    def run_email_extraction_tests(
        self,
        token_ids: List[int],
        model,
        tokenizer,
        chat_template,
        format_prompt_fn
    ) -> List[Dict[str, Any]]:
        """
        Run email extraction tests on multiple tokens

        Args:
            token_ids: List of token IDs to test
            model: The model instance
            tokenizer: The tokenizer instance
            chat_template: Chat template
            format_prompt_fn: Function to format prompts

        Returns:
            List of test results
        """
        self.logger.info(f"Running email extraction tests on {len(token_ids)} tokens...")

        results = []
        from tqdm import tqdm

        for token_id in tqdm(token_ids, desc="Testing email extraction"):
            token = tokenizer.decode([token_id])
            result = self.test_email_extraction(
                token_id, token, model, tokenizer, chat_template, format_prompt_fn
            )
            results.append(result)

        return results

    def get_last_test_result(self) -> Optional[Dict[str, Any]]:
        """Get the result of the last email extraction test"""
        return self._last_test_result

    def analyze_email_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze email extraction test results

        Args:
            results: List of test results

        Returns:
            Analysis summary
        """
        summary = {
            "total_tokens": len(results),
            "tokens_breaking_extraction": 0,
            "tokens_creating_valid_emails": 0,
            "tokens_creating_valid_and_breaking": 0,
            "common_issues": {},
            "results": results
        }

        for result in results:
            creates_valid = result.get("creates_valid_email", False)
            breaks_extraction = result.get("breaks_email_extraction", False)

            if breaks_extraction:
                summary["tokens_breaking_extraction"] += 1

            if creates_valid:
                summary["tokens_creating_valid_emails"] += 1

            if creates_valid and breaks_extraction:
                summary["tokens_creating_valid_and_breaking"] += 1

            # Count common issues
            for issue in result.get("issues", []):
                summary["common_issues"][issue] = summary["common_issues"].get(issue, 0) + 1

        return summary

    def print_email_results_summary(self, results: List[Dict[str, Any]]):
        """Print a summary of email extraction test results"""
        analysis = self.analyze_email_results(results)

        self.logger.info("\nEmail Extraction Test Results:")
        self.logger.info("=" * 80)

        broken_count = analysis["tokens_breaking_extraction"]
        extra_important_count = analysis["tokens_creating_valid_and_breaking"]

        for result in results:
            token = result["token"]
            creates_valid_email = result.get("creates_valid_email", False)
            breaks_extraction = result.get("breaks_email_extraction", False)
            is_extra_important = creates_valid_email and breaks_extraction

            if is_extra_important:
                email_addr = result.get("email_address", f"jeremy{token}@richards.ai")
                issues = ", ".join([i for i in result.get("issues", []) if i != "creates_valid_email_address"])
                self.logger.info(f"⭐❌ Token '{token}' BREAKS email extraction AND creates VALID email: {email_addr} - Issues: {issues}")
            elif breaks_extraction:
                issues = ", ".join(result.get("issues", []))
                self.logger.info(f"❌ Token '{token}' BREAKS email extraction - Issues: {issues}")
            else:
                self.logger.info(f"✅ Token '{token}' does NOT break email extraction")

        self.logger.info("=" * 80)
        self.logger.info(f"Summary: {broken_count}/{len(results)} tokens break email extraction")
        if extra_important_count > 0:
            self.logger.info(f"⭐ EXTRA IMPORTANT: {extra_important_count}/{len(results)} tokens create VALID email addresses AND break extraction!")
        else:
            self.logger.info("No tokens create valid email addresses that also break extraction")

        # Log common issues
        if analysis["common_issues"]:
            self.logger.info("\nMost common issues:")
            sorted_issues = sorted(analysis["common_issues"].items(), key=lambda x: x[1], reverse=True)
            for issue, count in sorted_issues[:5]:  # Top 5 issues
                self.logger.info(f"  {issue}: {count} tokens")
