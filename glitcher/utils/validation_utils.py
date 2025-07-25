#!/usr/bin/env python3
"""
Validation utilities for glitch token classification

This module provides utilities for validating email addresses, domain names,
and other data formats that may be affected by glitch tokens.
"""

import re
import string
from typing import Optional, Dict, Any, List, Tuple
from .logging_utils import get_logger

logger = get_logger(__name__)


def is_valid_email_token(token: str) -> bool:
    """
    Check if a token creates a valid email address when inserted into jeremy{token}@richards.ai

    Args:
        token: The token to validate

    Returns:
        True if the resulting email would be valid
    """
    # Check if token contains only valid email characters (letters, numbers, dots, hyphens, underscores)
    # No spaces, special characters, or other invalid email characters
    valid_chars = set(string.ascii_letters + string.digits + '.-_')

    # Check if all characters in token are valid for email (don't strip - spaces anywhere are invalid)
    if not all(c in valid_chars for c in token):
        return False

    # Check for other invalid patterns
    if (
        token == '' or                    # Empty token
        token.startswith('.') or          # Starts with dot
        token.endswith('.') or            # Ends with dot
        '..' in token or                  # Consecutive dots
        token.startswith('-') or          # Starts with hyphen
        token.endswith('-')               # Ends with hyphen
    ):
        return False

    return True


def is_valid_domain_token(token: str) -> bool:
    """
    Check if a token can create a valid domain name when inserted

    Args:
        token: The token to validate

    Returns:
        True if the token can create a valid domain name
    """
    # Remove any leading/trailing whitespace
    token = token.strip()

    # Check if token is empty or contains invalid characters
    if not token or not re.match(r'^[a-zA-Z0-9-]+$', token):
        return False

    # Check if token starts or ends with hyphen
    if token.startswith('-') or token.endswith('-'):
        return False

    # Check for consecutive hyphens (not allowed in domains)
    if '--' in token:
        return False

    return True


def validate_email_address(email: str) -> Dict[str, Any]:
    """
    Validate an email address and return detailed validation results

    Args:
        email: Email address to validate

    Returns:
        Dictionary with validation results
    """
    validation = {
        "is_valid": False,
        "email": email,
        "issues": [],
        "parts": {
            "local": None,
            "domain": None
        }
    }

    if not email:
        validation["issues"].append("empty_email")
        return validation

    # Split into local and domain parts
    if "@" not in email:
        validation["issues"].append("missing_at_symbol")
        return validation

    at_count = email.count("@")
    if at_count != 1:
        validation["issues"].append("multiple_at_symbols")
        return validation

    local_part, domain_part = email.split("@")
    validation["parts"]["local"] = local_part
    validation["parts"]["domain"] = domain_part

    # Validate local part
    if not local_part:
        validation["issues"].append("empty_local_part")
    elif len(local_part) > 64:
        validation["issues"].append("local_part_too_long")
    else:
        # Check for valid characters in local part
        valid_local_chars = set(string.ascii_letters + string.digits + '.-_+')
        if not all(c in valid_local_chars for c in local_part):
            validation["issues"].append("invalid_local_characters")

        # Check for dot rules
        if local_part.startswith('.') or local_part.endswith('.'):
            validation["issues"].append("local_part_starts_or_ends_with_dot")
        if '..' in local_part:
            validation["issues"].append("consecutive_dots_in_local_part")

    # Validate domain part
    domain_validation = validate_domain_name(domain_part)
    if not domain_validation["is_valid"]:
        validation["issues"].extend([f"domain_{issue}" for issue in domain_validation["issues"]])

    # Overall validation
    validation["is_valid"] = len(validation["issues"]) == 0

    return validation


def validate_domain_name(domain: str) -> Dict[str, Any]:
    """
    Validate a domain name and return detailed validation results

    Args:
        domain: Domain name to validate

    Returns:
        Dictionary with validation results
    """
    validation = {
        "is_valid": False,
        "domain": domain,
        "issues": [],
        "parts": {
            "labels": [],
            "tld": None
        }
    }

    if not domain:
        validation["issues"].append("empty_domain")
        return validation

    # Check overall length
    if len(domain) > 253:
        validation["issues"].append("domain_too_long")

    # Split into labels
    labels = domain.split('.')
    validation["parts"]["labels"] = labels

    if len(labels) < 2:
        validation["issues"].append("missing_tld")
        return validation

    validation["parts"]["tld"] = labels[-1]

    # Validate each label
    for i, label in enumerate(labels):
        if not label:
            validation["issues"].append(f"empty_label_{i}")
            continue

        if len(label) > 63:
            validation["issues"].append(f"label_{i}_too_long")

        # Check for valid characters (letters, numbers, hyphens)
        if not re.match(r'^[a-zA-Z0-9-]+$', label):
            validation["issues"].append(f"label_{i}_invalid_characters")

        # Check hyphen rules
        if label.startswith('-') or label.endswith('-'):
            validation["issues"].append(f"label_{i}_starts_or_ends_with_hyphen")

        # Check for consecutive hyphens
        if '--' in label:
            validation["issues"].append(f"label_{i}_consecutive_hyphens")

    # Validate TLD specifically
    tld = labels[-1]
    if len(tld) < 2:
        validation["issues"].append("tld_too_short")
    elif not re.match(r'^[a-zA-Z]+$', tld):
        validation["issues"].append("tld_invalid_characters")

    # Overall validation
    validation["is_valid"] = len(validation["issues"]) == 0

    return validation


def extract_email_parts(email: str) -> Dict[str, Optional[str]]:
    """
    Extract username, domain, and TLD from email address

    Args:
        email: Email address to parse

    Returns:
        Dictionary with extracted parts
    """
    result = {
        "username": None,
        "domain": None,
        "tld": None,
        "is_valid": False
    }

    validation = validate_email_address(email)
    if not validation["is_valid"]:
        return result

    # Extract username (local part)
    result["username"] = validation["parts"]["local"]

    # Extract domain and TLD
    domain_part = validation["parts"]["domain"]
    if domain_part and "." in domain_part:
        domain_labels = domain_part.split(".")
        result["domain"] = domain_part
        result["tld"] = domain_labels[-1]
        result["is_valid"] = True

    return result


def extract_domain_from_log_entry(log_entry: str) -> Optional[str]:
    """
    Extract domain from a log entry, specifically looking for from= field

    Args:
        log_entry: Log entry text

    Returns:
        Extracted domain or None if not found
    """
    # Look for from=<email@domain> pattern
    from_pattern = r'from=<[^@]+@([^>]+)>'
    match = re.search(from_pattern, log_entry)

    if match:
        return match.group(1)

    return None


def validate_extracted_email_data(
    response: str,
    expected_username: str,
    expected_domain: str,
    expected_tld: str
) -> Dict[str, Any]:
    """
    Validate extracted email data against expected values

    Args:
        response: Model response containing JSON with email data
        expected_username: Expected username value
        expected_domain: Expected domain value
        expected_tld: Expected TLD value

    Returns:
        Dictionary with validation results
    """
    from .json_utils import extract_and_parse_json

    validation = {
        "is_valid": True,
        "issues": [],
        "extracted_data": None,
        "expected_data": {
            "username": expected_username,
            "domain": expected_domain,
            "tld": expected_tld
        }
    }

    # Extract JSON from response
    parsed_json = extract_and_parse_json(response)
    if not parsed_json:
        validation["issues"].append("no_json_found")
        validation["is_valid"] = False
        return validation

    validation["extracted_data"] = parsed_json

    # Check required fields
    required_fields = ["username", "domain", "tld"]
    missing_fields = [field for field in required_fields if field not in parsed_json]
    if missing_fields:
        validation["issues"].append(f"missing_fields_{','.join(missing_fields)}")
        validation["is_valid"] = False

    # Validate each field
    actual_username = parsed_json.get("username", "")
    if actual_username != expected_username:
        validation["issues"].append("incorrect_username")
        validation["is_valid"] = False

    actual_domain = parsed_json.get("domain", "")
    if actual_domain != expected_domain:
        validation["issues"].append("incorrect_domain")
        validation["is_valid"] = False

    actual_tld = parsed_json.get("tld", "")
    if actual_tld != expected_tld:
        validation["issues"].append("incorrect_tld")
        validation["is_valid"] = False

    return validation


def validate_extracted_domain_data(
    response: str,
    expected_domain: str
) -> Dict[str, Any]:
    """
    Validate extracted domain data against expected values

    Args:
        response: Model response containing JSON with domain data
        expected_domain: Expected domain value

    Returns:
        Dictionary with validation results
    """
    from .json_utils import extract_and_parse_json

    validation = {
        "is_valid": True,
        "issues": [],
        "extracted_data": None,
        "expected_data": {
            "domain": expected_domain
        }
    }

    # Extract JSON from response
    parsed_json = extract_and_parse_json(response)
    if not parsed_json:
        validation["issues"].append("no_json_found")
        validation["is_valid"] = False
        return validation

    validation["extracted_data"] = parsed_json

    # Check required field
    if "domain" not in parsed_json:
        validation["issues"].append("missing_domain_field")
        validation["is_valid"] = False
        return validation

    # Validate domain value
    actual_domain = parsed_json.get("domain", "")
    if actual_domain != expected_domain:
        validation["issues"].append("incorrect_domain")
        validation["is_valid"] = False

    # Check for invalid domain characters
    if not re.match(r'^[a-zA-Z0-9.-]+$', actual_domain):
        validation["issues"].append("invalid_domain_characters")
        validation["is_valid"] = False

    return validation


class EmailValidator:
    """Class-based email validator with configurable behavior"""

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.logger = get_logger(self.__class__.__name__)

    def is_valid_token(self, token: str) -> bool:
        """Check if token creates valid email address"""
        return is_valid_email_token(token)

    def validate_address(self, email: str) -> Dict[str, Any]:
        """Validate complete email address"""
        return validate_email_address(email)

    def extract_parts(self, email: str) -> Dict[str, Optional[str]]:
        """Extract email parts"""
        return extract_email_parts(email)

    def validate_extraction(
        self,
        response: str,
        expected_username: str,
        expected_domain: str,
        expected_tld: str
    ) -> Dict[str, Any]:
        """Validate extracted email data"""
        return validate_extracted_email_data(
            response, expected_username, expected_domain, expected_tld
        )


class DomainValidator:
    """Class-based domain validator with configurable behavior"""

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.logger = get_logger(self.__class__.__name__)

    def is_valid_token(self, token: str) -> bool:
        """Check if token creates valid domain name"""
        return is_valid_domain_token(token)

    def validate_domain(self, domain: str) -> Dict[str, Any]:
        """Validate complete domain name"""
        return validate_domain_name(domain)

    def extract_from_log(self, log_entry: str) -> Optional[str]:
        """Extract domain from log entry"""
        return extract_domain_from_log_entry(log_entry)

    def validate_extraction(
        self,
        response: str,
        expected_domain: str
    ) -> Dict[str, Any]:
        """Validate extracted domain data"""
        return validate_extracted_domain_data(response, expected_domain)


def create_test_email_address(token: str, base_email: str = "jeremy{}@richards.ai") -> str:
    """
    Create a test email address with the given token

    Args:
        token: Token to insert into email
        base_email: Base email template with {} placeholder

    Returns:
        Complete email address with token inserted
    """
    return base_email.format(token)


def create_test_domain_name(token: str, base_domain: str = "bad-{}-domain.xyz") -> str:
    """
    Create a test domain name with the given token

    Args:
        token: Token to insert into domain
        base_domain: Base domain template with {} placeholder

    Returns:
        Complete domain name with token inserted
    """
    return base_domain.format(token)


def analyze_token_impact(token: str) -> Dict[str, Any]:
    """
    Analyze how a token affects email and domain validation

    Args:
        token: Token to analyze

    Returns:
        Dictionary with analysis results
    """
    analysis = {
        "token": token,
        "email_impact": {
            "creates_valid_email": is_valid_email_token(token),
            "test_email": create_test_email_address(token),
            "issues": []
        },
        "domain_impact": {
            "creates_valid_domain": is_valid_domain_token(token),
            "test_domain": create_test_domain_name(token),
            "issues": []
        }
    }

    # Analyze email impact
    test_email = analysis["email_impact"]["test_email"]
    email_validation = validate_email_address(test_email)
    if not email_validation["is_valid"]:
        analysis["email_impact"]["issues"] = email_validation["issues"]

    # Analyze domain impact
    test_domain = analysis["domain_impact"]["test_domain"]
    domain_validation = validate_domain_name(test_domain)
    if not domain_validation["is_valid"]:
        analysis["domain_impact"]["issues"] = domain_validation["issues"]

    return analysis
