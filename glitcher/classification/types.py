#!/usr/bin/env python3
"""
Core types and categories for glitch token classification

This module defines the fundamental data structures used throughout the glitch
classification system, including category definitions and test templates.
"""

from typing import Dict, Callable, Any, Optional, List


class GlitchCategory:
    """Defines categories for classifying glitch tokens"""
    INJECTION = "Injection"  # Prompt injection / jailbreaking
    IDOS = "IDOS"  # Infinite/Denial-of-Service
    HALLUCINATION = "Hallucination"  # Nonsensical output
    DISRUPTION = "Disruption"  # Internal reasoning disruption
    BYPASS = "Bypass"  # Filter or guardrail bypass
    EMAIL_EXTRACTION = "EmailExtraction"  # Email/TLD extraction issues
    VALID_EMAIL_ADDRESS = "ValidEmailAddress"  # Token creates valid email address
    DOMAIN_EXTRACTION = "DomainExtraction"  # Domain extraction from logs issues
    VALID_DOMAIN_NAME = "ValidDomainName"  # Token creates valid domain name
    CONTROL_CHAR_CONFUSION = "ControlCharConfusion"  # Control character semantic interpretation
    ENCODED_CHAR_CONFUSION = "EncodedCharConfusion"  # Encoded character text decoding
    UNKNOWN = "Unknown"  # Unable to categorize

    @classmethod
    def all_categories(cls) -> List[str]:
        """Get all available categories"""
        return [
            cls.INJECTION,
            cls.IDOS,
            cls.HALLUCINATION,
            cls.DISRUPTION,
            cls.BYPASS,
            cls.EMAIL_EXTRACTION,
            cls.VALID_EMAIL_ADDRESS,
            cls.DOMAIN_EXTRACTION,
            cls.VALID_DOMAIN_NAME,
            cls.CONTROL_CHAR_CONFUSION,
            cls.ENCODED_CHAR_CONFUSION,
            cls.UNKNOWN
        ]

    @classmethod
    def behavioral_categories(cls) -> List[str]:
        """Get categories that represent behavioral glitches"""
        return [
            cls.INJECTION,
            cls.IDOS,
            cls.HALLUCINATION,
            cls.DISRUPTION,
            cls.BYPASS
        ]

    @classmethod
    def functional_categories(cls) -> List[str]:
        """Get categories that represent functional issues"""
        return [
            cls.EMAIL_EXTRACTION,
            cls.DOMAIN_EXTRACTION,
            cls.CONTROL_CHAR_CONFUSION,
            cls.ENCODED_CHAR_CONFUSION
        ]

    @classmethod
    def validity_categories(cls) -> List[str]:
        """Get categories that represent token validity"""
        return [
            cls.VALID_EMAIL_ADDRESS,
            cls.VALID_DOMAIN_NAME
        ]


class ClassificationTest:
    """Defines a test template for glitch classification"""

    def __init__(
        self,
        name: str,
        category: str,
        template: str,
        indicators: Dict[str, Callable[[str], bool]],
        description: str = "",
        priority: int = 0
    ):
        """
        Initialize a classification test

        Args:
            name: Unique name for the test
            category: GlitchCategory this test detects
            template: Template string with {token} placeholder
            indicators: Dict of indicator_name -> check_function
            description: Human-readable description of what this test does
            priority: Higher priority tests run first (default: 0)
        """
        self.name = name
        self.category = category
        self.template = template
        self.indicators = indicators
        self.description = description
        self.priority = priority

    def __repr__(self) -> str:
        return f"ClassificationTest(name='{self.name}', category='{self.category}')"


class TestResult:
    """Represents the result of running a single classification test"""

    def __init__(
        self,
        test_name: str,
        category: str,
        token_id: int,
        token: str,
        prompt: str,
        response: str,
        indicators: Dict[str, bool],
        is_positive: Optional[bool] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.test_name = test_name
        self.category = category
        self.token_id = token_id
        self.token = token
        self.prompt = prompt
        self.response = response
        self.indicators = indicators
        self.is_positive = is_positive if is_positive is not None else any(indicators.values())
        self.error = error
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            "test_name": self.test_name,
            "category": self.category,
            "token_id": self.token_id,
            "token": self.token,
            "prompt": self.prompt,
            "response": self.response,
            "response_length": len(self.response),
            "indicators": self.indicators,
            "is_positive": self.is_positive,
            "error": self.error,
            "metadata": self.metadata
        }

    def __repr__(self) -> str:
        status = "POSITIVE" if self.is_positive else "NEGATIVE"
        return f"TestResult({self.test_name}: {status} for token '{self.token}')"


class ClassificationResult:
    """Represents the complete classification result for a token"""

    def __init__(
        self,
        token_id: int,
        token: str,
        test_results: Optional[List[TestResult]] = None,
        categories: Optional[List[str]] = None,
        timestamp: Optional[float] = None
    ):
        self.token_id = token_id
        self.token = token
        self.test_results = test_results or []
        self.categories = categories or []
        self.timestamp = timestamp

    def add_test_result(self, result: TestResult):
        """Add a test result and update categories if positive"""
        self.test_results.append(result)
        if result.is_positive and result.category not in self.categories:
            self.categories.append(result.category)

    def has_category(self, category: str) -> bool:
        """Check if token has been classified with given category"""
        return category in self.categories

    def get_positive_tests(self) -> List[TestResult]:
        """Get all test results that were positive"""
        return [result for result in self.test_results if result.is_positive]

    def get_tests_for_category(self, category: str) -> List[TestResult]:
        """Get all test results for a specific category"""
        return [result for result in self.test_results if result.category == category]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            "token_id": self.token_id,
            "token": self.token,
            "test_results": [result.to_dict() for result in self.test_results],
            "categories": self.categories,
            "timestamp": self.timestamp
        }

    def __repr__(self) -> str:
        categories_str = ", ".join(self.categories) if self.categories else "None"
        return f"ClassificationResult(token='{self.token}', categories=[{categories_str}])"


class TestConfig:
    """Configuration for running classification tests"""

    def __init__(
        self,
        max_tokens: int = 200,
        temperature: float = 0.0,
        timeout: float = 30.0,
        enable_debug: bool = False,
        simple_template: bool = False
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.enable_debug = enable_debug
        self.simple_template = simple_template

    @classmethod
    def from_args(cls, args) -> 'TestConfig':
        """Create config from command line arguments"""
        return cls(
            max_tokens=getattr(args, 'max_tokens', 200),
            temperature=getattr(args, 'temperature', 0.0),
            timeout=getattr(args, 'timeout', 30.0),
            enable_debug=getattr(args, 'debug_responses', False),
            simple_template=getattr(args, 'simple_template', False)
        )
