"""
Glitcher - A CLI tool for mining and testing glitch tokens in large language models

This package provides both the original monolithic classifier and a new modular
architecture for building custom classification systems.

Modules:
    classification: Core classification framework and main GlitchClassifier
    tests: Modular test components (email, domain, prompt, baseline tests)
    utils: Utilities for JSON parsing, validation, and logging
    model: Model loading and template management
    enhanced_validation: Enhanced validation functionality
    scan_and_validate: Scanning and validation tools

Classes:
    GlitchClassifier: Main classifier using modular architecture
    BaseClassifier: Base class for building custom classifiers
    EmailTester: Email extraction testing module
    DomainTester: Domain extraction testing module
    GlitchCategory: Standard categories for classification
    TestConfig: Configuration for running tests

Functions:
    setup_logger: Set up logging with tqdm compatibility
    is_valid_email_token: Check if token creates valid email
    is_valid_domain_token: Check if token creates valid domain
    extract_json_from_response: Extract JSON from model responses
"""

from typing import Optional

__version__ = "1.0.0"
__author__ = "Glitcher Development Team"

# Import main components for easy access
from .classification.glitch_classifier import GlitchClassifier
from .classification.types import (
    GlitchCategory,
    TestConfig,
    ClassificationTest,
    TestResult,
    ClassificationResult
)

from .classification.base_classifier import BaseClassifier

from .tests.email_tests import EmailTester
from .tests.domain_tests import DomainTester

from .utils import (
    setup_logger,
    get_logger,
    is_valid_email_token,
    is_valid_domain_token,
    extract_json_from_response,
    extract_and_parse_json,
    validate_email_address,
    validate_domain_name,
    analyze_token_impact,
    JSONExtractor,
    EmailValidator,
    DomainValidator
)

# Backward compatibility - expose original interface
from .classify_glitches_modular import ClassificationWrapper as LegacyClassifier

__all__ = [
    # Main classification components
    'GlitchClassifier',
    'BaseClassifier',
    'GlitchCategory',
    'TestConfig',
    'ClassificationTest',
    'TestResult',
    'ClassificationResult',

    # Test modules
    'EmailTester',
    'DomainTester',

    # Utilities
    'setup_logger',
    'get_logger',
    'is_valid_email_token',
    'is_valid_domain_token',
    'extract_json_from_response',
    'extract_and_parse_json',
    'validate_email_address',
    'validate_domain_name',
    'analyze_token_impact',
    'JSONExtractor',
    'EmailValidator',
    'DomainValidator',

    # Backward compatibility
    'LegacyClassifier'
]

def get_version():
    """Get the package version"""
    return __version__

def get_available_classifiers():
    """Get information about available classifier types"""
    return {
        'GlitchClassifier': 'Main classifier with comprehensive test suite',
        'BaseClassifier': 'Base class for building custom classifiers',
        'LegacyClassifier': 'Backward-compatible wrapper for original interface'
    }

def get_available_testers():
    """Get information about available test modules"""
    return {
        'EmailTester': 'Email extraction and validation testing',
        'DomainTester': 'Domain extraction from log files testing'
    }

def create_default_classifier(model_path: str, **kwargs):
    """
    Create a default GlitchClassifier instance

    Args:
        model_path: Path or name of the model to use
        **kwargs: Additional arguments passed to GlitchClassifier

    Returns:
        Configured GlitchClassifier instance
    """
    return GlitchClassifier(model_path=model_path, **kwargs)

def create_email_tester(config: Optional[TestConfig] = None):
    """
    Create an EmailTester instance

    Args:
        config: Optional test configuration

    Returns:
        EmailTester instance
    """
    return EmailTester(config)

def create_domain_tester(config: Optional[TestConfig] = None):
    """
    Create a DomainTester instance

    Args:
        config: Optional test configuration

    Returns:
        DomainTester instance
    """
    return DomainTester(config)
