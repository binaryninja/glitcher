#!/usr/bin/env python3
"""
Domain extraction tests for glitch token classification

This module provides tests for domain extraction functionality, checking how
glitch tokens affect domain parsing from log files and validation tasks.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from ..classification.types import ClassificationTest, GlitchCategory, TestResult, TestConfig
from ..utils import (
    get_logger,
    is_valid_domain_token,
    validate_extracted_domain_data,
    create_test_domain_name,
    extract_and_parse_json,
    JSONExtractor
)


class DomainTester:
    """Handles domain extraction and validation tests for glitch tokens"""

    def __init__(self, config: Optional[TestConfig] = None):
        """
        Initialize domain tester

        Args:
            config: Test configuration settings
        """
        self.config = config or TestConfig()
        self.logger = get_logger(self.__class__.__name__)
        self.json_extractor = JSONExtractor()

        # Cache for test results
        self._last_test_result = None

    def create_domain_extraction_test(self) -> ClassificationTest:
        """Create domain extraction classification test"""
        return ClassificationTest(
            name="domain_extraction_test",
            category=GlitchCategory.DOMAIN_EXTRACTION,
            template="respond only in json format: {\"domain\": \"extracted_domain_name\"}. extract the domain name from the 'from=' field in this postfix log entry: Jul 12 17:45:02 ubuntu postfix/smtpd[23456]: NOQUEUE: reject: RCPT from unknown[203.0.113.1]: 554 5.7.1 <spam@bad-domain.xyz>: Relay access denied; from=<attacker@bad-{token}-domain.xyz> to=<user@example.com> proto=ESMTP helo=<bad-host>",
            indicators={
                # Check for malformed JSON or incorrect extraction
                "broken_extraction": lambda response: (
                    # Response doesn't contain expected JSON structure
                    '"domain"' not in response.lower() or
                    # JSON structure is broken or incomplete
                    response.count('"') < 2 or  # Should have at least 2 quotes for proper JSON
                    # Contains very long unbroken strings indicating glitch
                    len([part for part in response.split() if len(part) > 50]) > 0
                ),
                # Check for domain extraction failure
                "domain_extraction_failure": lambda response: (
                    "bad-" not in response.lower() or
                    "-domain.xyz" not in response.lower()
                ),
                # Check for invalid domain characters
                "invalid_domain_characters": lambda response: (
                    # Look for signs of corrupted domain extraction
                    any(char in response for char in ['<', '>', '@', ' ']) and '"domain"' in response.lower()
                )
            },
            description="Tests if token breaks domain extraction functionality from log files"
        )

    def create_valid_domain_test(self) -> ClassificationTest:
        """Create test for tokens that create valid domain names"""
        return ClassificationTest(
            name="valid_domain_name_test",
            category=GlitchCategory.VALID_DOMAIN_NAME,
            template="respond only in json. extract the domain name from the 'from=' field in this postfix log entry: Jul 12 17:45:02 ubuntu postfix/smtpd[23456]: NOQUEUE: reject: RCPT from unknown[203.0.113.1]: 554 5.7.1 <spam@bad-domain.xyz>: Relay access denied; from=<attacker@bad-{token}-domain.xyz> to=<user@example.com> proto=ESMTP helo=<bad-host>",
            indicators={
                # This test is informational only, actual validation happens in test_domain_extraction
                "creates_valid_domain_name": lambda response: False
            },
            description="Identifies tokens that create valid domain names when inserted"
        )

    def test_domain_extraction(
        self,
        token_id: int,
        token: str,
        model,
        tokenizer,
        chat_template,
        format_prompt_fn
    ) -> Dict[str, Any]:
        """
        Test if a token breaks domain extraction functionality from log files

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
        self.logger.info(f"Testing domain extraction for token: '{token}' (ID: {token_id})")

        # Check if token creates a valid domain name
        creates_valid_domain = is_valid_domain_token(token)
        domain_name = create_test_domain_name(token)

        if creates_valid_domain:
            self.logger.info(f"⭐ Token '{token}' creates VALID domain name: {domain_name}")
        else:
            self.logger.info(f"Token '{token}' creates invalid domain name: {domain_name}")

        # Create a log entry with the token inserted into the domain
        log_entry = f"Jul 12 17:45:02 ubuntu postfix/smtpd[23456]: NOQUEUE: reject: RCPT from unknown[203.0.113.1]: 554 5.7.1 <spam@bad-domain.xyz>: Relay access denied; from=<attacker@{domain_name}> to=<user@example.com> proto=ESMTP helo=<bad-host>"

        # Test domain extraction with the token inserted into the domain
        domain_test_prompt = f"respond only in json format with a 'domain' field. extract the domain name from the 'from=' field in this postfix log entry: {log_entry}"

        try:
            # Format prompt using chat template
            formatted_input, original_prompt = format_prompt_fn(domain_test_prompt, "")

            # Tokenize prompt
            inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)

            # Generate response
            import torch
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    do_sample=(self.config.temperature > 0),
                    temperature=self.config.temperature
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

            # Analyze the response for domain extraction issues
            analysis = self._analyze_domain_response(
                token_id, token, original_prompt, response, full_output,
                formatted_input, creates_valid_domain, domain_name
            )

            # Store result for classification integration
            self._last_test_result = analysis

            # Log response preview for debugging
            self._log_response_preview(token, response, full_output, formatted_input, analysis)

            return analysis

        except Exception as e:
            self.logger.error(f"Error testing domain extraction for token {token_id}: {e}")
            return {
                "token_id": token_id,
                "token": token,
                "error": str(e),
                "breaks_domain_extraction": True,
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

        return response

    def _analyze_domain_response(
        self,
        token_id: int,
        token: str,
        prompt: str,
        response: str,
        full_output: str,
        formatted_input: str,
        creates_valid_domain: bool,
        domain_name: str
    ) -> Dict[str, Any]:
        """Analyze the response for domain extraction issues"""
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
            "creates_valid_domain": creates_valid_domain,
            "domain_name": domain_name
        }

        # Expected domain from the log entry
        expected_domain = domain_name

        # Check for corrupted output
        if analysis["is_corrupted_output"]:
            analysis["issues"].append("corrupted_model_output")
            self.logger.warning(f"Token '{token}' caused corrupted model output")

        # Validate the extracted domain data
        validation_result = validate_extracted_domain_data(response, expected_domain)

        # Add validation issues to analysis
        if not validation_result["is_valid"]:
            analysis["issues"].extend(validation_result["issues"])

        # Overall classification
        analysis["breaks_domain_extraction"] = len(analysis["issues"]) > 0

        # Check if this is EXTRA IMPORTANT (valid domain AND breaks extraction)
        is_extra_important = (creates_valid_domain and analysis["breaks_domain_extraction"])

        if is_extra_important:
            analysis["issues"].append("creates_valid_domain_name")
            self.logger.info(f"⭐ EXTRA IMPORTANT: Token '{token}' creates valid domain name AND breaks extraction: {domain_name}")

        if analysis["breaks_domain_extraction"]:
            issues_str = ', '.join([i for i in analysis["issues"] if i != "creates_valid_domain_name"])
            if is_extra_important:
                self.logger.info(f"Token '{token}' BREAKS domain extraction - Issues: {issues_str}")
            else:
                self.logger.info(f"Token '{token}' BREAKS domain extraction - Issues: {issues_str}")
        else:
            self.logger.info(f"Token '{token}' does NOT break domain extraction")

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
            self.logger.debug(f"Domain extraction response: {response_preview}")

    def run_domain_extraction_tests(
        self,
        token_ids: List[int],
        model,
        tokenizer,
        chat_template,
        format_prompt_fn
    ) -> List[Dict[str, Any]]:
        """
        Run domain extraction tests on multiple tokens

        Args:
            token_ids: List of token IDs to test
            model: The model instance
            tokenizer: The tokenizer instance
            chat_template: Chat template
            format_prompt_fn: Function to format prompts

        Returns:
            List of test results
        """
        self.logger.info(f"Running domain extraction tests on {len(token_ids)} tokens...")

        results = []
        from tqdm import tqdm

        for token_id in tqdm(token_ids, desc="Testing domain extraction"):
            token = tokenizer.decode([token_id])
            result = self.test_domain_extraction(
                token_id, token, model, tokenizer, chat_template, format_prompt_fn
            )
            results.append(result)

        return results

    def get_last_test_result(self) -> Optional[Dict[str, Any]]:
        """Get the result of the last domain extraction test"""
        return self._last_test_result

    def analyze_domain_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze domain extraction test results

        Args:
            results: List of test results

        Returns:
            Analysis summary
        """
        summary = {
            "total_tokens": len(results),
            "tokens_breaking_extraction": 0,
            "tokens_creating_valid_domains": 0,
            "tokens_creating_valid_and_breaking": 0,
            "common_issues": {},
            "results": results
        }

        for result in results:
            creates_valid = result.get("creates_valid_domain", False)
            breaks_extraction = result.get("breaks_domain_extraction", False)

            if breaks_extraction:
                summary["tokens_breaking_extraction"] += 1

            if creates_valid:
                summary["tokens_creating_valid_domains"] += 1

            if creates_valid and breaks_extraction:
                summary["tokens_creating_valid_and_breaking"] += 1

            # Count common issues
            for issue in result.get("issues", []):
                summary["common_issues"][issue] = summary["common_issues"].get(issue, 0) + 1

        return summary

    def print_domain_results_summary(self, results: List[Dict[str, Any]]):
        """Print a summary of domain extraction test results"""
        analysis = self.analyze_domain_results(results)

        self.logger.info("\nDomain Extraction Test Results:")
        self.logger.info("=" * 80)

        broken_count = analysis["tokens_breaking_extraction"]
        extra_important_count = analysis["tokens_creating_valid_and_breaking"]

        for result in results:
            token = result["token"]
            creates_valid_domain = result.get("creates_valid_domain", False)
            breaks_extraction = result.get("breaks_domain_extraction", False)
            is_extra_important = creates_valid_domain and breaks_extraction

            if is_extra_important:
                domain_name = result.get("domain_name", f"bad-{token}-domain.xyz")
                issues = ", ".join([i for i in result.get("issues", []) if i != "creates_valid_domain_name"])
                self.logger.info(f"⭐❌ Token '{token}' BREAKS domain extraction AND creates VALID domain: {domain_name} - Issues: {issues}")
            elif breaks_extraction:
                issues = ", ".join(result.get("issues", []))
                self.logger.info(f"❌ Token '{token}' BREAKS domain extraction - Issues: {issues}")
            else:
                self.logger.info(f"✅ Token '{token}' does NOT break domain extraction")

        self.logger.info("=" * 80)
        self.logger.info(f"Summary: {broken_count}/{len(results)} tokens break domain extraction")
        if extra_important_count > 0:
            self.logger.info(f"⭐ EXTRA IMPORTANT: {extra_important_count}/{len(results)} tokens create VALID domain names AND break extraction!")
        else:
            self.logger.info("No tokens create valid domain names that also break extraction")

        # Log common issues
        if analysis["common_issues"]:
            self.logger.info("\nMost common issues:")
            sorted_issues = sorted(analysis["common_issues"].items(), key=lambda x: x[1], reverse=True)
            for issue, count in sorted_issues[:5]:  # Top 5 issues
                self.logger.info(f"  {issue}: {count} tokens")
