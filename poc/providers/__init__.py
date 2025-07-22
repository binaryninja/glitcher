"""
Multi-provider API client package for testing prompt injection vulnerabilities.

This package provides a unified interface for testing different AI API providers
including Mistral, Lambda AI, OpenAI, Anthropic, and local transformers models.
"""


try:
    from dotenv import load_dotenv
    # Try to load .env file from current directory and parent directories
    load_dotenv()
except ImportError:
    # dotenv not installed, skip loading .env files
    pass

from .base import BaseProvider, ModelInfo, RequestResult

# Provider registry will be populated dynamically
AVAILABLE_PROVIDERS = {}


def _get_available_providers():
    """Dynamically discover available providers based on installed dependencies."""
    if AVAILABLE_PROVIDERS:
        return AVAILABLE_PROVIDERS

    # Try to import and register Mistral provider
    try:
        from .mistral import MistralProvider
        AVAILABLE_PROVIDERS['mistral'] = MistralProvider
    except ImportError:
        pass

    # Try to import and register Lambda provider
    try:
        from .lambda_ai import LambdaProvider
        AVAILABLE_PROVIDERS['lambda'] = LambdaProvider
    except ImportError:
        pass

    # Try to import and register OpenAI provider
    try:
        from .openai_provider import OpenAIProvider
        AVAILABLE_PROVIDERS['openai'] = OpenAIProvider
    except ImportError:
        pass

    # Try to import and register Anthropic provider
    try:
        from .anthropic_provider import AnthropicProvider
        AVAILABLE_PROVIDERS['anthropic'] = AnthropicProvider
    except ImportError:
        pass

    # Try to import and register Transformers provider
    try:
        from .transformers_provider import TransformersProvider
        AVAILABLE_PROVIDERS['transformers'] = TransformersProvider
    except ImportError:
        pass

    return AVAILABLE_PROVIDERS


def get_provider(provider_name: str, api_key: str = None, **kwargs) -> BaseProvider:
    """
    Get a provider instance by name.

    Args:
        provider_name: Name of the provider ('mistral', 'lambda', 'openai', 'anthropic', 'transformers')
        api_key: API key for the provider (optional, can use env vars, not needed for transformers)
        **kwargs: Additional provider-specific arguments (for transformers: model_path, device, quant_type)

    Returns:
        Configured provider instance

    Raises:
        ValueError: If provider name is not supported
        ImportError: If provider dependencies are not installed
    """
    provider_name = provider_name.lower()
    available_providers = _get_available_providers()

    if provider_name not in available_providers:
        available = ', '.join(available_providers.keys()) if available_providers else 'none'
        if not available_providers:
            raise ValueError(
                "No providers available. Install dependencies: "
                f"'pip install mistralai' for Mistral, 'pip install openai' for Lambda/OpenAI, 'pip install anthropic' for Anthropic, "
                f"'pip install transformers accelerate' for local transformers models"
            )
        raise ValueError(f"Unsupported provider '{provider_name}'. Available providers: {available}")

    provider_class = available_providers[provider_name]

    # Handle case where api_key is None (e.g., for local transformers models)
    if api_key is None and provider_name == 'transformers':
        return provider_class(**kwargs)
    else:
        return provider_class(api_key=api_key, **kwargs)


def list_available_providers():
    """List all available provider names."""
    return list(_get_available_providers().keys())


__all__ = [
    'BaseProvider',
    'ModelInfo',
    'RequestResult',
    'get_provider',
    'list_available_providers',
    'AVAILABLE_PROVIDERS',
]
