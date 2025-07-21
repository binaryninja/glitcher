#!/usr/bin/env python3
"""
Utilities module for glitch token classification

This module provides common utilities used throughout the glitch classification
system, including JSON parsing, validation, and logging functionality.

Modules:
    json_utils: Utilities for extracting and parsing JSON from model responses
    validation_utils: Utilities for validating email addresses and domain names
    logging_utils: Logging setup and utilities with tqdm compatibility

Classes:
    JSONExtractor: Class-based JSON extraction with configurable behavior
    EmailValidator: Class-based email validation utilities
    DomainValidator: Class-based domain validation utilities
    TqdmLoggingHandler: Custom logging handler for tqdm compatibility
    ProgressLogger: Context manager for progress logging

Functions:
    extract_json_from_response(): Extract JSON from model response text
    is_valid_email_token(): Check if token creates valid email address
    is_valid_domain_token(): Check if token creates valid domain name
    setup_logger(): Set up logger with console and file output
    get_logger(): Get existing logger or create with defaults
"""

from .json_utils import (
    extract_json_from_response,
    extract_and_parse_json,
    parse_json_safely,
    extract_field_safely,
    validate_json_fields,
    format_json_for_logging,
    JSONExtractor
)

from .validation_utils import (
    is_valid_email_token,
    is_valid_domain_token,
    validate_email_address,
    validate_domain_name,
    extract_email_parts,
    extract_domain_from_log_entry,
    validate_extracted_email_data,
    validate_extracted_domain_data,
    create_test_email_address,
    create_test_domain_name,
    analyze_token_impact,
    EmailValidator,
    DomainValidator
)

from .logging_utils import (
    setup_logger,
    get_logger,
    get_default_logger,
    set_log_level,
    enable_debug_logging,
    disable_debug_logging,
    log_test_start,
    log_test_result,
    log_classification_summary,
    log_error,
    log_warning,
    log_info,
    log_banner,
    log_section,
    TqdmLoggingHandler,
    ProgressLogger
)

__all__ = [
    # JSON utilities
    'extract_json_from_response',
    'extract_and_parse_json',
    'parse_json_safely',
    'extract_field_safely',
    'validate_json_fields',
    'format_json_for_logging',
    'JSONExtractor',

    # Validation utilities
    'is_valid_email_token',
    'is_valid_domain_token',
    'validate_email_address',
    'validate_domain_name',
    'extract_email_parts',
    'extract_domain_from_log_entry',
    'validate_extracted_email_data',
    'validate_extracted_domain_data',
    'create_test_email_address',
    'create_test_domain_name',
    'analyze_token_impact',
    'EmailValidator',
    'DomainValidator',

    # Logging utilities
    'setup_logger',
    'get_logger',
    'get_default_logger',
    'set_log_level',
    'enable_debug_logging',
    'disable_debug_logging',
    'log_test_start',
    'log_test_result',
    'log_classification_summary',
    'log_error',
    'log_warning',
    'log_info',
    'log_banner',
    'log_section',
    'TqdmLoggingHandler',
    'ProgressLogger'
]

# Version info
__version__ = '1.0.0'
__author__ = 'Glitcher Development Team'

def get_version():
    """Get the utils module version"""
    return __version__

def get_available_utilities():
    """Get list of available utility categories"""
    return {
        'json': 'JSON extraction and parsing utilities',
        'validation': 'Email and domain validation utilities',
        'logging': 'Logging setup and progress tracking utilities'
    }
