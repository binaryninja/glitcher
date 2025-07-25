from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a model from a provider."""
    id: str
    name: str
    provider: str
    capabilities: List[str]
    supports_function_calling: bool


@dataclass
class RequestResult:
    """Result from a single API request."""
    test_number: int
    model_id: str
    provider: str
    model_test_duration: float
    api_key_leaked_in_message: bool
    api_key_correct: bool
    username: Optional[str]
    domain: Optional[str]
    tld: Optional[str]
    message_body: Optional[str]
    api_key_used: Optional[str]
    full_response: Optional[str]
    raw_response: Optional[Any]
    parsing_error: Optional[str]
    error: Optional[str] = None


class BaseProvider(ABC):
    """Abstract base class for API providers."""

    def __init__(self, api_key: str, **kwargs):
        """Initialize the provider with API key and any provider-specific options."""
        self.api_key = api_key
        self.provider_name = self.__class__.__name__.replace('Provider', '').lower()

    @abstractmethod
    def list_models(self, quiet=False) -> Tuple[List[ModelInfo], List[ModelInfo]]:
        """
        List available models from the provider.

        Args:
            quiet: If True, suppress verbose output during model listing

        Returns:
            Tuple of (function_calling_models, other_models)
        """
        pass

    def validate_model(self, model_id: str) -> Optional[ModelInfo]:
        """
        Validate a specific model without listing all models.

        This method provides an efficient way to check if a specific model
        is available without needing to list all models. Providers can override
        this for better performance.

        Args:
            model_id: The specific model to validate

        Returns:
            ModelInfo if model is valid and available, None otherwise
        """
        # Default implementation: fall back to listing all models
        function_calling_models, other_models = self.list_models(quiet=True)
        all_models = function_calling_models + other_models

        for model in all_models:
            if model.id == model_id:
                return model

        return None

    @abstractmethod
    def make_request(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        instructions: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Make a request to the provider's API.

        Args:
            model_id: The model to use
            messages: List of messages in the conversation
            tools: List of tool definitions
            instructions: System instructions (if supported)
            **kwargs: Additional provider-specific parameters

        Returns:
            Raw response from the provider
        """
        pass

    @abstractmethod
    def analyze_response(self, response: Any) -> Dict[str, Any]:
        """
        Analyze the response from the provider and extract relevant information.

        Args:
            response: Raw response from the provider

        Returns:
            Dictionary with analyzed response data
        """
        pass

    def make_request_with_retry(
        self,
        test_number: int,
        model_id: str,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        instructions: Optional[str] = None,
        max_retries: int = 3,
        **kwargs
    ) -> RequestResult:
        """
        Make a request with retry logic for rate limiting and temporary failures.

        Args:
            test_number: Test identifier
            model_id: The model to use
            messages: List of messages in the conversation
            tools: List of tool definitions
            instructions: System instructions (if supported)
            max_retries: Maximum number of retry attempts
            **kwargs: Additional provider-specific parameters

        Returns:
            RequestResult with the analyzed response or error information
        """
        import time
        import random

        last_response = None
        start_time = time.time()

        for attempt in range(max_retries):
            try:
                # Add small random delay to avoid thundering herd
                time.sleep(random.uniform(0.1, 0.3))

                response = self.make_request(
                    model_id=model_id,
                    messages=messages,
                    tools=tools,
                    instructions=instructions,
                    **kwargs
                )

                last_response = response
                result_data = self.analyze_response(response)

                return RequestResult(
                    test_number=test_number,
                    model_id=model_id,
                    provider=self.provider_name,
                    model_test_duration=time.time() - start_time,
                    **result_data
                )

            except Exception as e:
                error_str = str(e).lower()

                # Handle rate limiting (429 errors)
                if self._is_rate_limit_error(error_str):
                    backoff_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limited on test {test_number}, attempt {attempt + 1}. Backing off for {backoff_time:.1f}s")
                    time.sleep(backoff_time)
                    continue

                # Handle other retryable errors
                elif attempt < max_retries - 1 and self._is_retryable_error(error_str):
                    backoff_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Retryable error on test {test_number}, attempt {attempt + 1}: {str(e)}")
                    time.sleep(backoff_time)
                    continue

                # Non-retryable error or max retries reached
                print(f"Error in test {test_number} after {attempt + 1} attempts: {str(e)}")

                # Create error result with captured response data if available
                error_result = RequestResult(
                    test_number=test_number,
                    model_id=model_id,
                    provider=self.provider_name,
                    model_test_duration=time.time() - start_time,
                    error=str(e),
                    api_key_leaked_in_message=False,
                    api_key_correct=False,
                    username=None,
                    domain=None,
                    tld=None,
                    message_body=None,
                    api_key_used=None,
                    full_response=None,
                    raw_response=None,
                    parsing_error=None
                )

                # If we got a response before the error, try to capture it
                if last_response:
                    try:
                        partial_result = self.analyze_response(last_response)
                        error_result.full_response = partial_result.get('full_response')
                        error_result.raw_response = partial_result.get('raw_response')
                        error_result.parsing_error = partial_result.get('parsing_error')
                        error_result.username = partial_result.get('username')
                        error_result.domain = partial_result.get('domain')
                        error_result.tld = partial_result.get('tld')
                        error_result.message_body = partial_result.get('message_body')
                        error_result.api_key_used = partial_result.get('api_key_used')
                        error_result.api_key_correct = partial_result.get('api_key_correct')
                        error_result.api_key_leaked_in_message = partial_result.get('api_key_leaked_in_message')
                    except Exception as parse_error:
                        error_result.parsing_error = f"Failed to parse response after error: {str(parse_error)}"

                return error_result

        # Should not reach here, but just in case
        return RequestResult(
            test_number=test_number,
            model_id=model_id,
            provider=self.provider_name,
            model_test_duration=time.time() - start_time,
            error=f"Max retries ({max_retries}) exceeded",
            api_key_leaked_in_message=False,
            api_key_correct=False,
            username=None,
            domain=None,
            tld=None,
            message_body=None,
            api_key_used=None,
            full_response=None,
            raw_response=None,
            parsing_error=None
        )

    def _is_rate_limit_error(self, error_str: str) -> bool:
        """Check if an error string indicates rate limiting."""
        rate_limit_indicators = ['429', 'rate limit', 'too many requests']
        return any(indicator in error_str for indicator in rate_limit_indicators)

    def _is_retryable_error(self, error_str: str) -> bool:
        """Check if an error string indicates a retryable error."""
        retryable_indicators = ['timeout', 'connection', 'network', 'temporary']
        return any(indicator in error_str for indicator in retryable_indicators)

    @abstractmethod
    def get_provider_specific_config(self) -> Dict[str, Any]:
        """Return provider-specific configuration options."""
        pass
