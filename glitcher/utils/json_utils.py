#!/usr/bin/env python3
"""
JSON utilities for glitch token classification

This module provides utilities for extracting and parsing JSON from model responses,
handling various edge cases and malformed JSON that can occur with glitch tokens.
"""

import json
import re
from typing import Optional, Dict, Any, Union, List
from .logging_utils import get_logger

logger = get_logger(__name__)


def extract_json_from_response(response: str, fallback_strategy: str = "aggressive") -> Optional[str]:
    """
    Extract JSON from model response, handling code blocks and malformed JSON

    Args:
        response: The model's response text
        fallback_strategy: Strategy for extraction - "conservative" or "aggressive"

    Returns:
        Extracted JSON string or None if not found
    """
    if not response or not response.strip():
        logger.debug("Empty response, no JSON to extract")
        return None

    # Method 1: Try to find JSON in code blocks first (```json ... ```)
    json_str = _extract_from_code_blocks(response)
    if json_str:
        return json_str

    # Method 2: Try to find raw JSON objects
    json_str = _extract_raw_json(response)
    if json_str:
        return json_str

    # Method 3: Aggressive search with brace matching
    if fallback_strategy == "aggressive":
        json_str = _extract_with_braces(response)
        if json_str:
            return json_str

        # Method 4: Try to complete incomplete JSON
        json_str = _complete_incomplete_json(response)
        if json_str:
            return json_str

    logger.debug("No valid JSON found in response")
    return None


def _extract_from_code_blocks(response: str) -> Optional[str]:
    """Extract JSON from markdown code blocks"""
    # Pattern for ```json ... ``` or ``` ... ``` containing JSON
    json_code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    json_matches = re.findall(json_code_block_pattern, response, re.DOTALL | re.IGNORECASE)

    if json_matches:
        # Take the last JSON block found (most likely to be the final answer)
        json_str = json_matches[-1].strip()
        if _is_valid_json(json_str):
            logger.debug(f"Found JSON in code block: {json_str[:100]}...")
            return json_str

    return None


def _extract_raw_json(response: str) -> Optional[str]:
    """Extract raw JSON objects from response text"""
    # Look for JSON objects that aren't in code blocks
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    potential_jsons = re.findall(json_pattern, response, re.DOTALL)

    if potential_jsons:
        # Take the last JSON object found and validate it
        for json_candidate in reversed(potential_jsons):
            json_candidate = json_candidate.strip()
            if _is_valid_json(json_candidate):
                logger.debug(f"Found valid raw JSON: {json_candidate[:100]}...")
                return json_candidate

    return None


def _extract_with_braces(response: str) -> Optional[str]:
    """Extract JSON using brace matching"""
    brace_start = response.rfind('{')
    brace_end = response.rfind('}')

    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        json_candidate = response[brace_start:brace_end + 1].strip()
        if _is_valid_json(json_candidate):
            logger.debug(f"Found JSON with brace search: {json_candidate[:100]}...")
            return json_candidate

    return None


def _complete_incomplete_json(response: str) -> Optional[str]:
    """Try to complete incomplete JSON by adding missing braces"""
    # Method 1: Try to complete incomplete JSON (missing closing braces)
    incomplete_json_pattern = r'\{[^{}]*(?:"[^"]*":\s*"[^"]*"[^{}]*)*'
    incomplete_matches = re.findall(incomplete_json_pattern, response, re.DOTALL)

    if incomplete_matches:
        for incomplete_candidate in reversed(incomplete_matches):
            # Try to complete the JSON by adding missing closing braces
            for num_braces in range(1, 4):  # Try adding 1-3 closing braces
                completed_candidate = incomplete_candidate + ('}' * num_braces)
                if _is_valid_json(completed_candidate):
                    logger.debug(f"Completed incomplete JSON with {num_braces} closing brace(s)")
                    return completed_candidate

    # Method 2: Try to complete JSON missing opening braces
    missing_opening_pattern = r'"[^"]*":\s*"[^"]*"(?:\s*,\s*"[^"]*":\s*"[^"]*")*'
    missing_opening_matches = re.findall(missing_opening_pattern, response, re.DOTALL)

    if missing_opening_matches:
        for missing_opening_candidate in reversed(missing_opening_matches):
            # Try to complete by adding opening and closing braces
            completed_candidate = '{' + missing_opening_candidate + '}'
            if _is_valid_json(completed_candidate):
                logger.debug("Completed JSON missing opening brace")
                return completed_candidate

    return None


def _is_valid_json(json_str: str) -> bool:
    """Check if a string is valid JSON"""
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        return False


def parse_json_safely(json_str: str) -> Optional[Dict[str, Any]]:
    """
    Safely parse JSON string with error handling

    Args:
        json_str: JSON string to parse

    Returns:
        Parsed JSON as dict or None if parsing fails
    """
    if not json_str:
        return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parsing failed: {e}")
        return None


def extract_and_parse_json(response: str, fallback_strategy: str = "aggressive") -> Optional[Dict[str, Any]]:
    """
    Extract and parse JSON from response in one step

    Args:
        response: Model response text
        fallback_strategy: Extraction strategy

    Returns:
        Parsed JSON dict or None if extraction/parsing fails
    """
    json_str = extract_json_from_response(response, fallback_strategy)
    if json_str:
        return parse_json_safely(json_str)
    return None


def validate_json_fields(
    parsed_json: Dict[str, Any],
    required_fields: list[str],
    optional_fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate that JSON contains required fields and return validation results

    Args:
        parsed_json: Parsed JSON dictionary
        required_fields: List of required field names
        optional_fields: List of optional field names

    Returns:
        Dictionary with validation results
    """
    optional_fields = optional_fields or []

    validation = {
        "is_valid": True,
        "missing_required": [],
        "present_required": [],
        "present_optional": [],
        "extra_fields": []
    }

    # Check required fields
    for field in required_fields:
        if field in parsed_json:
            validation["present_required"].append(field)
        else:
            validation["missing_required"].append(field)
            validation["is_valid"] = False

    # Check optional fields
    for field in optional_fields:
        if field in parsed_json:
            validation["present_optional"].append(field)

    # Check for extra fields
    all_expected = set(required_fields + optional_fields)
    for field in parsed_json.keys():
        if field not in all_expected:
            validation["extra_fields"].append(field)

    return validation


def extract_field_safely(
    response: str,
    field_name: str,
    expected_type: type = str,
    fallback_strategy: str = "aggressive"
) -> Optional[Union[str, int, float, bool]]:
    """
    Extract a specific field from JSON in response

    Args:
        response: Model response text
        field_name: Name of field to extract
        expected_type: Expected type of the field value
        fallback_strategy: JSON extraction strategy

    Returns:
        Field value if found and correct type, None otherwise
    """
    parsed_json = extract_and_parse_json(response, fallback_strategy)
    if not parsed_json:
        return None

    if field_name not in parsed_json:
        return None

    value = parsed_json[field_name]

    # Type checking
    if not isinstance(value, expected_type):
        logger.debug(f"Field '{field_name}' has wrong type: expected {expected_type}, got {type(value)}")
        return None

    return value


def format_json_for_logging(data: Union[Dict, str], max_length: int = 200) -> str:
    """
    Format JSON data for logging with length limit

    Args:
        data: JSON data (dict or string)
        max_length: Maximum length for output

    Returns:
        Formatted string for logging
    """
    if isinstance(data, dict):
        json_str = json.dumps(data, separators=(',', ':'))
    else:
        json_str = str(data)

    if len(json_str) <= max_length:
        return json_str

    return json_str[:max_length] + "..."


class JSONExtractor:
    """Class-based JSON extractor with configurable behavior"""

    def __init__(
        self,
        fallback_strategy: str = "aggressive",
        enable_logging: bool = True,
        max_attempts: int = 3
    ):
        self.fallback_strategy = fallback_strategy
        self.enable_logging = enable_logging
        self.max_attempts = max_attempts
        self.logger = get_logger(self.__class__.__name__) if enable_logging else None

    def extract(self, response: str) -> Optional[str]:
        """Extract JSON string from response"""
        return extract_json_from_response(response, self.fallback_strategy)

    def parse(self, json_str: str) -> Optional[Dict[str, Any]]:
        """Parse JSON string to dictionary"""
        return parse_json_safely(json_str)

    def extract_and_parse(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON in one step"""
        return extract_and_parse_json(response, self.fallback_strategy)

    def extract_field(
        self,
        response: str,
        field_name: str,
        expected_type: type = str
    ) -> Optional[Union[str, int, float, bool]]:
        """Extract specific field from JSON"""
        return extract_field_safely(response, field_name, expected_type, self.fallback_strategy)
