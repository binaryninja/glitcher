#!/usr/bin/env python3
"""
Glitch token classification module

This module provides a modular framework for classifying glitch tokens by their
behavioral effects. It includes base classifiers, test definitions, and utilities
for building custom classification systems.

Classes:
    GlitchCategory: Defines standard categories for token classification
    ClassificationTest: Template for defining classification tests
    TestResult: Result of running a single test
    ClassificationResult: Complete classification result for a token
    TestConfig: Configuration for running tests
    BaseClassifier: Base class for building classifiers
    GlitchClassifier: Main classifier implementation

Functions:
    create_default_tests(): Create the standard set of classification tests
    setup_classification_logging(): Set up logging for classification
"""

from .types import (
    GlitchCategory,
    ClassificationTest,
    TestResult,
    ClassificationResult,
    TestConfig
)

__all__ = [
    'GlitchCategory',
    'ClassificationTest',
    'TestResult',
    'ClassificationResult',
    'TestConfig'
]

# Version info
__version__ = '1.0.0'
__author__ = 'Glitcher Development Team'

def get_version():
    """Get the module version"""
    return __version__
