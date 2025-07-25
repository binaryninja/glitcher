#!/usr/bin/env python3
"""
Glitch token classification tests module

This module provides modular test components for classifying glitch tokens.
Each test module focuses on a specific type of behavior or functionality
that can be affected by glitch tokens.

Modules:
    prompt_tests: Tests for prompting behavior and repetition patterns
    email_tests: Tests for email extraction and validation functionality
    domain_tests: Tests for domain extraction from log files
    baseline_tests: Tests using standard tokens to validate test reliability

Classes:
    PromptTester: Handles prompting behavior tests
    EmailTester: Handles email extraction tests
    DomainTester: Handles domain extraction tests
    BaselineTester: Handles baseline validation tests

Functions:
    create_default_test_suite(): Create standard set of all tests
    run_test_suite(): Run a complete test suite on tokens
"""

from typing import List, Dict, Any

# Test module imports will be added as modules are created
# from .prompt_tests import PromptTester
# from .email_tests import EmailTester
# from .domain_tests import DomainTester
# from .baseline_tests import BaselineTester

__all__ = [
    # Will be populated as test modules are created
]

# Version info
__version__ = '1.0.0'

def get_available_test_types() -> List[str]:
    """Get list of available test types"""
    return [
        'prompt_tests',
        'email_tests',
        'domain_tests',
        'baseline_tests'
    ]

def get_test_descriptions() -> Dict[str, str]:
    """Get descriptions of available test types"""
    return {
        'prompt_tests': 'Tests for prompting behavior and response patterns',
        'email_tests': 'Tests for email extraction and validation functionality',
        'domain_tests': 'Tests for domain extraction from log entries',
        'baseline_tests': 'Tests using standard tokens for validation'
    }
