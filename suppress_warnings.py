#!/usr/bin/env python3
"""
Suppress Warnings Module

This module provides utilities to suppress common warnings from the transformers library
that appear during glitch token validation and generation.
"""

import warnings
import logging
import os
from contextlib import contextmanager
from typing import Optional


def suppress_transformers_warnings():
    """
    Suppress common transformers library warnings that don't affect functionality.

    This function suppresses:
    - Generation parameter warnings (top_p, temperature, etc.)
    - Attention mask warnings
    - Tokenizer warnings
    - Model loading warnings
    """
    # Set transformers verbosity to error level only
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    # Suppress specific warning categories
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    warnings.filterwarnings("ignore", message=".*generation flags.*")
    warnings.filterwarnings("ignore", message=".*attention mask.*")
    warnings.filterwarnings("ignore", message=".*pad token.*")
    warnings.filterwarnings("ignore", message=".*eos token.*")

    # Set transformers logger to error level
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)


def enable_transformers_warnings():
    """
    Re-enable transformers warnings for debugging purposes.
    """
    # Reset transformers verbosity
    os.environ["TRANSFORMERS_VERBOSITY"] = "info"

    # Reset warning filters
    warnings.resetwarnings()

    # Reset transformers logger
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)


@contextmanager
def suppress_warnings_context():
    """
    Context manager to temporarily suppress warnings.

    Usage:
        with suppress_warnings_context():
            # Code that might generate warnings
            model.generate(...)
    """
    # Store original state
    original_verbosity = os.environ.get("TRANSFORMERS_VERBOSITY", "info")
    original_filters = warnings.filters.copy()

    try:
        suppress_transformers_warnings()
        yield
    finally:
        # Restore original state
        os.environ["TRANSFORMERS_VERBOSITY"] = original_verbosity
        warnings.filters = original_filters


def setup_quiet_mode():
    """
    Set up quiet mode for glitch token validation.

    This reduces console output to essential information only.
    """
    suppress_transformers_warnings()

    # Also suppress other common warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Set up minimal logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s"
    )


def setup_verbose_mode():
    """
    Set up verbose mode for debugging.

    This enables all warnings and detailed logging.
    """
    enable_transformers_warnings()

    # Enable all warnings
    warnings.filterwarnings("default")

    # Set up detailed logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


class QuietGeneration:
    """
    Context manager class for quiet model generation.

    Usage:
        with QuietGeneration():
            outputs = model.generate(...)
    """

    def __init__(self, suppress_all: bool = True):
        self.suppress_all = suppress_all
        self.original_verbosity = None
        self.original_filters = None

    def __enter__(self):
        if self.suppress_all:
            # Store original state
            self.original_verbosity = os.environ.get("TRANSFORMERS_VERBOSITY", "info")
            self.original_filters = warnings.filters.copy()

            # Suppress warnings
            suppress_transformers_warnings()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress_all and self.original_verbosity is not None:
            # Restore original state
            os.environ["TRANSFORMERS_VERBOSITY"] = self.original_verbosity
            warnings.filters = self.original_filters


def get_warning_status() -> dict:
    """
    Get the current warning suppression status.

    Returns:
        dict: Dictionary containing current warning settings
    """
    return {
        "transformers_verbosity": os.environ.get("TRANSFORMERS_VERBOSITY", "info"),
        "warning_filters_count": len(warnings.filters),
        "transformers_logger_level": logging.getLogger("transformers").level,
        "root_logger_level": logging.getLogger().level
    }


def print_warning_status():
    """
    Print current warning suppression status to console.
    """
    status = get_warning_status()
    print("Warning Suppression Status:")
    print(f"  Transformers verbosity: {status['transformers_verbosity']}")
    print(f"  Warning filters active: {status['warning_filters_count']}")
    print(f"  Transformers logger level: {status['transformers_logger_level']}")
    print(f"  Root logger level: {status['root_logger_level']}")


# Auto-suppress warnings when module is imported
# This can be overridden by calling enable_transformers_warnings()
if os.environ.get("GLITCHER_SUPPRESS_WARNINGS", "true").lower() == "true":
    suppress_transformers_warnings()
