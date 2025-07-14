#!/usr/bin/env python3
"""
Simple Demo: Email Extraction Testing for Glitch Tokens

This demo shows how to test if a glitch token breaks email extraction functionality.

Usage:
    python demo_email_extraction.py meta-llama/Llama-3.2-1B-Instruct
"""

import sys
import json
import logging
from typing import Dict, Any
from pathlib import Path

# Add the parent directory to the path so we can import glitcher modules
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Error: Required packages not installed. Please run:")
    print("pip install torch transformers accelerate")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_email_extraction_simple(model_path: str, token_text: str) -> Dict[str, Any]:
    """
    Simple test for email extraction with a given token

    Args:
        model_path: Path to the model
        token_text: The token text to test

    Returns:
        Test results dictionary
    """
    logger.info(f"Loading model: {model_path}")

    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {"error": str(e)}

    logger.info(f"Testing token: '{token_text}'")

    # Create the email extraction prompt
    prompt = f"respond only in json. extract the username domain and tld from this email address: jeremy{token_text}@richards.ai"

    logger.info("Running email extraction test...")
    logger.info(f"Prompt: {prompt}")

    try:
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode response
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_output[len(prompt):].strip()

        logger.info(f"Response: {response}")

        # Analyze the response
        analysis = analyze_email_response(response, token_text)

        return {
            "token": token_text,
            "prompt": prompt,
            "response": response,
            "analysis": analysis,
            "breaks_email_extraction": analysis["has_issues"]
        }

    except Exception as e:
        logger.error(f"Error during generation: {e}")
        return {
            "token": token_text,
            "error": str(e),
            "breaks_email_extraction": True
        }


def extract_json_from_response(response: str) -> str:
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
            return json_candidate
        except json.JSONDecodeError:
            pass

    return None


def analyze_email_response(response: str, original_token: str) -> Dict[str, Any]:
    """
    Analyze the email extraction response for issues

    Args:
        response: The model's response
        original_token: The original token being tested

    Returns:
        Analysis results
    """
    issues = []

    # Expected values
    expected_username = f"jeremy{original_token}"
    expected_domain = "richards.ai"
    expected_tld = "ai"

    # Try to parse JSON response with robust extraction
    try:
        json_str = extract_json_from_response(response)

        if json_str is None:
            issues.append("no_json_found")
        else:
            parsed_json = json.loads(json_str)

            # Validate required fields exist
            required_fields = ["username", "domain", "tld"]
            missing_fields = [field for field in required_fields if field not in parsed_json]
            if missing_fields:
                issues.append(f"missing_fields: {', '.join(missing_fields)}")

            # Validate username
            actual_username = parsed_json.get("username", "")
            if actual_username != expected_username:
                issues.append(f"incorrect_username: expected '{expected_username}', got '{actual_username}'")

            # Validate domain
            actual_domain = parsed_json.get("domain", "")
            if actual_domain != expected_domain:
                issues.append(f"incorrect_domain: expected '{expected_domain}', got '{actual_domain}'")

            # Validate TLD
            actual_tld = parsed_json.get("tld", "")
            if actual_tld != expected_tld:
                issues.append(f"incorrect_tld: expected '{expected_tld}', got '{actual_tld}'")

    except json.JSONDecodeError as e:
        issues.append(f"invalid_json: {e}")
    except Exception as e:
        issues.append(f"json_parsing_error: {e}")

    # Check for very short or very long responses (indicators of failure)
    if len(response) < 10:
        issues.append("response_too_short")
    elif len(response) > 1000:
        issues.append("response_too_long")

    return {
        "has_issues": len(issues) > 0,
        "issues": issues,
        "response_length": len(response),
        "expected_username": expected_username,
        "expected_domain": expected_domain,
        "expected_tld": expected_tld
    }


def demo_cpp_types_token(model_path: str) -> None:
    """
    Demo testing the known CppTypeDefinitionSizes glitch token

    Args:
        model_path: Path to the model to test
    """
    print("=" * 80)
    print("EMAIL EXTRACTION GLITCH TOKEN DEMO")
    print("=" * 80)

    # Test the known glitch token
    glitch_token = " CppTypeDefinitionSizes"  # Note the leading space

    print(f"\nTesting known glitch token: '{glitch_token}'")

    result = test_email_extraction_simple(model_path, glitch_token)

    if "error" in result:
        print(f"❌ Test failed: {result['error']}")
        return

    print(f"\nPrompt sent to model:")
    print(f"'{result['prompt']}'")

    print(f"\nModel response:")
    print(f"'{result['response']}'")

    analysis = result['analysis']
    print(f"\nAnalysis:")
    print(f"Response length: {analysis['response_length']} characters")
    print(f"Expected username: {analysis['expected_username']}")
    print(f"Expected domain: {analysis['expected_domain']}")
    print(f"Expected TLD: {analysis['expected_tld']}")
    print(f"Issues found: {len(analysis['issues'])}")

    if analysis['issues']:
        print("Issue details:")
        for issue in analysis['issues']:
            print(f"  - {issue}")

    print(f"\nConclusion:")
    if result['breaks_email_extraction']:
        print("❌ This token BREAKS email extraction functionality!")
    else:
        print("✅ This token does NOT break email extraction.")

    # For comparison, test a normal token
    print("\n" + "-" * 60)
    print("Testing normal token for comparison:")

    normal_token = "computer"
    normal_result = test_email_extraction_simple(model_path, normal_token)

    if "error" not in normal_result:
        print(f"Normal token '{normal_token}' response: '{normal_result['response']}'")
        if normal_result['breaks_email_extraction']:
            print("⚠️  Warning: Normal token also breaks extraction (check model/prompt)")
        else:
            print("✅ Normal token works correctly")

    print("=" * 80)


def main():
    """
    Main function
    """
    if len(sys.argv) != 2:
        print("Usage: python demo_email_extraction.py <model_path>")
        print("Example: python demo_email_extraction.py meta-llama/Llama-3.2-1B-Instruct")
        sys.exit(1)

    model_path = sys.argv[1]

    try:
        demo_cpp_types_token(model_path)
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    main()
