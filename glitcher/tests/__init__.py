#!/usr/bin/env python3
"""
Glitch token classification tests module

This module provides modular test components for classifying glitch tokens.
Each test module focuses on a specific type of behavior or functionality
that can be affected by glitch tokens.

Modules:
    email_tests: Tests for email extraction and validation functionality
    domain_tests: Tests for domain extraction from log files
    control_char_tests: Tests for control character semantic confusion

Classes:
    EmailTester: Handles email extraction tests
    DomainTester: Handles domain extraction tests
    ControlCharTester: Handles control character confusion tests

Functions:
    get_available_test_types(): Get list of available test types
    get_test_descriptions(): Get descriptions of available test types
"""

from typing import List, Dict

from .email_tests import EmailTester
from .domain_tests import DomainTester
from .control_char_tests import ControlCharTester

__all__ = [
    'EmailTester',
    'DomainTester',
    'ControlCharTester',
]

# Version info
__version__ = '1.0.0'


def get_available_test_types() -> List[str]:
    """Get list of available test types"""
    return [
        'email_tests',
        'domain_tests',
        'control_char_tests',
    ]


def get_test_descriptions() -> Dict[str, str]:
    """Get descriptions of available test types"""
    return {
        'email_tests': 'Tests for email extraction and validation functionality',
        'domain_tests': 'Tests for domain extraction from log entries',
        'control_char_tests': 'Tests for control character semantic confusion',
    }
