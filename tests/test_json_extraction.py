#!/usr/bin/env python3
"""
JSON Extraction Test Script

Test script to verify that JSON extraction from model responses works correctly
with thinking tokens and code blocks.

Usage:
    python test_json_extraction.py
"""

import json
import re
import sys
from typing import Optional


def extract_json_from_response(response: str) -> Optional[str]:
    """
    Extract JSON from model response, handling code blocks and thinking tokens

    Args:
        response: The model's response text

    Returns:
        Extracted JSON string or None if not found
    """
    # Method 1: Try to find JSON in code blocks first (```json ... ```)
    # This handles cases where the model wraps JSON in markdown code blocks
    json_code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    json_matches = re.findall(json_code_block_pattern, response, re.DOTALL | re.IGNORECASE)

    if json_matches:
        # Take the last JSON block found (most likely to be the final answer)
        json_str = json_matches[-1].strip()
        print(f"Found JSON in code block: {json_str[:100]}...")
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
                print(f"Found valid raw JSON: {json_candidate[:100]}...")
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
            print(f"Found JSON with brace search: {json_candidate[:100]}...")
            return json_candidate
        except json.JSONDecodeError:
            pass

    print("No valid JSON found in response")
    return None


def validate_email_extraction(json_str: str, expected_username: str,
                            expected_domain: str, expected_tld: str) -> dict:
    """
    Validate extracted JSON for email extraction test

    Args:
        json_str: Extracted JSON string
        expected_username: Expected username value
        expected_domain: Expected domain value
        expected_tld: Expected TLD value

    Returns:
        Validation results
    """
    try:
        parsed = json.loads(json_str)

        results = {
            "valid_json": True,
            "has_all_fields": all(field in parsed for field in ["username", "domain", "tld"]),
            "correct_username": parsed.get("username") == expected_username,
            "correct_domain": parsed.get("domain") == expected_domain,
            "correct_tld": parsed.get("tld") == expected_tld,
            "extracted": parsed
        }

        results["all_correct"] = (results["has_all_fields"] and
                                results["correct_username"] and
                                results["correct_domain"] and
                                results["correct_tld"])

        return results

    except json.JSONDecodeError as e:
        return {
            "valid_json": False,
            "error": str(e),
            "all_correct": False
        }


def test_json_extraction():
    """
    Test JSON extraction with various response formats
    """
    print("=" * 80)
    print("JSON EXTRACTION TEST SUITE")
    print("=" * 80)

    # Test cases mimicking real model responses
    test_cases = [
        {
            "name": "Code block with json specifier",
            "response": """## Step 1: Parse the email address
The email address is jeremyhello@richards.ai. We need to extract the username, domain, and tld from this email address.

## Step 2: Extract the username
The username is jeremyhello.

## Step 3: Extract the domain
The domain is richards.ai.

## Step 4: Extract the tld
The tld is ai.

## Step 5: Combine the extracted information into a JSON object
We can represent the extracted information as a JSON object:
```json
{
  "username": "jeremyhello",
  "domain": "richards.ai",
  "tld": "ai"
}
```""",
            "expected": {
                "username": "jeremyhello",
                "domain": "richards.ai",
                "tld": "ai"
            }
        },

        {
            "name": "Code block without json specifier",
            "response": """Looking at the email jeremytest@richards.ai, I need to extract the components.

The answer is:
```
{
  "username": "jeremytest",
  "domain": "richards.ai",
  "tld": "ai"
}
```""",
            "expected": {
                "username": "jeremytest",
                "domain": "richards.ai",
                "tld": "ai"
            }
        },

        {
            "name": "Raw JSON with thinking",
            "response": """I need to parse jeremyworld@richards.ai

Let me break this down:
- The part before @ is the username: jeremyworld
- The part after @ is the domain: richards.ai
- The TLD is the part after the last dot: ai

{"username": "jeremyworld", "domain": "richards.ai", "tld": "ai"}""",
            "expected": {
                "username": "jeremyworld",
                "domain": "richards.ai",
                "tld": "ai"
            }
        },

        {
            "name": "Multiple JSON blocks (should take last)",
            "response": """First, let me show an example format:
```json
{
  "username": "example",
  "domain": "example.com",
  "tld": "com"
}
```

Now for the actual email jeremycomputer@richards.ai:
```json
{
  "username": "jeremycomputer",
  "domain": "richards.ai",
  "tld": "ai"
}
```""",
            "expected": {
                "username": "jeremycomputer",
                "domain": "richards.ai",
                "tld": "ai"
            }
        },

        {
            "name": "Glitch token causing issues",
            "response": """## Step 1: Parse the email address
The email address is jeremy CppTypeDefinitionSizes@richards.ai.

## Step 2: Extract components
Looking at jeremy CppTypeDefinitionSizes@richards.ai...

```json
{
  "username": "jeremy",
  "domain": "richards.ai",
  "tld": "ai"
}
```""",
            "expected": {
                "username": "jeremy CppTypeDefinitionSizes",
                "domain": "richards.ai",
                "tld": "ai"
            },
            "should_fail": True  # This should fail because username is wrong
        },

        {
            "name": "No JSON found",
            "response": """I cannot process this request as it contains invalid characters that prevent proper email parsing.""",
            "expected": {
                "username": "jeremytest",
                "domain": "richards.ai",
                "tld": "ai"
            },
            "should_fail": True
        }
    ]

    total_tests = len(test_cases)
    passed_tests = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}/{total_tests}: {test_case['name']}")
        print("-" * 60)

        # Extract JSON
        extracted_json = extract_json_from_response(test_case["response"])

        if extracted_json is None:
            if test_case.get("should_fail", False):
                print("✅ PASS - Correctly failed to extract JSON")
                passed_tests += 1
            else:
                print("❌ FAIL - Could not extract JSON")
            continue

        # Validate extraction
        expected = test_case["expected"]
        validation = validate_email_extraction(
            extracted_json,
            expected["username"],
            expected["domain"],
            expected["tld"]
        )

        print(f"Extracted JSON: {extracted_json}")
        print(f"Valid JSON: {validation.get('valid_json', False)}")
        print(f"Has all fields: {validation.get('has_all_fields', False)}")
        print(f"Correct username: {validation.get('correct_username', False)}")
        print(f"Correct domain: {validation.get('correct_domain', False)}")
        print(f"Correct TLD: {validation.get('correct_tld', False)}")

        if test_case.get("should_fail", False):
            if not validation.get("all_correct", False):
                print("✅ PASS - Correctly detected issue")
                passed_tests += 1
            else:
                print("❌ FAIL - Should have detected issue but didn't")
        else:
            if validation.get("all_correct", False):
                print("✅ PASS - All validations passed")
                passed_tests += 1
            else:
                print("❌ FAIL - Validation failed")
                if "extracted" in validation:
                    print(f"   Got: {validation['extracted']}")
                    print(f"   Expected: {expected}")

    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed")
    print("=" * 80)

    return passed_tests == total_tests


def main():
    """
    Main function
    """
    try:
        success = test_json_extraction()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Test suite failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
