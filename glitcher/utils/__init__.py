#!/usr/bin/env python3
"""
Public convenience re-exports for the `glitcher.utils` namespace.

Keeping this file stable means downstream code can do:

    from glitcher.utils import get_logger, extract_email_parts, ...

without caring about the underlying module layout.

Only lightweight symbol re-exports live here – the real implementations are
located in the sibling modules:

* logging_utils.py
* json_utils.py
* validation_utils.py
"""

# ---------------------------------------------------------------------
#  JSON helpers
# ---------------------------------------------------------------------
from .json_utils import (
    extract_json_from_response,
    extract_and_parse_json,
    parse_json_safely,
    extract_field_safely,
    validate_json_fields,
    format_json_for_logging,
    JSONExtractor,
)

# ---------------------------------------------------------------------
#  Validation helpers
# ---------------------------------------------------------------------
from .validation_utils import (
    # RFC-compliant token validators
    is_valid_email_token,
    is_valid_domain_token,
    validate_email_address,
    validate_domain_name,
    # Back-compat wrappers
    extract_email_parts,
    extract_domain_from_log_entry,
    validate_extracted_email_data,
    validate_extracted_domain_data,
    create_test_email_address,
    create_test_domain_name,
    EmailValidator,
   analyze_token_impact,
   DomainValidator,
)

# ---------------------------------------------------------------------
#  Response extraction helpers
# ---------------------------------------------------------------------
from .response_utils import (
    extract_assistant_response,
)

# ---------------------------------------------------------------------
#  Logging / progress helpers
# ---------------------------------------------------------------------
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
    ProgressLogger,
)

# ---------------------------------------------------------------------
#  Public export list
# ---------------------------------------------------------------------
__all__ = [
    # JSON
    "extract_json_from_response",
    "extract_and_parse_json",
    "parse_json_safely",
    "extract_field_safely",
    "validate_json_fields",
    "format_json_for_logging",
    "JSONExtractor",
    # Validation
    "is_valid_email_token",
    "is_valid_domain_token",
    "validate_email_address",
    "validate_domain_name",
    "extract_email_parts",
    "extract_domain_from_log_entry",
    "validate_extracted_email_data",
    "validate_extracted_domain_data",
    "create_test_email_address",
   "create_test_domain_name",
   "analyze_token_impact",
   "EmailValidator",
   "DomainValidator",
    # Response extraction
    "extract_assistant_response",
    # Logging
    "setup_logger",
    "get_logger",
    "get_default_logger",
    "set_log_level",
    "enable_debug_logging",
    "disable_debug_logging",
    "log_test_start",
    "log_test_result",
    "log_classification_summary",
    "log_error",
    "log_warning",
    "log_info",
    "log_banner",
    "log_section",
    "TqdmLoggingHandler",
    "ProgressLogger",
]

# Meta
__version__ = "1.0.0"
__author__ = "Glitcher Development Team"
