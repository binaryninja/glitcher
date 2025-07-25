# Anthropic Provider Documentation

This document explains how to use the Anthropic provider with the multi-provider prompt injection testing system.

## Overview

The Anthropic provider enables testing of Claude models (Claude 3, Claude 3.5, Claude 3.7, and Claude 4) for prompt injection vulnerabilities. It supports all function calling capabilities and integrates seamlessly with the existing multi-provider testing framework.

## Installation

### Requirements

```bash
# Install the anthropic package
pip install anthropic>=0.25.0

# Or install from the requirements file
pip install -r requirements.txt
```

### API Key Setup

Get your Anthropic API key from the [Anthropic Console](https://console.anthropic.com/) and set it as an environment variable:

```bash
# Set API key for current session
export ANTHROPIC_API_KEY=your_api_key_here

# Or add to your .env file
echo "ANTHROPIC_API_KEY=your_api_key_here" >> .env

# Or add to your shell profile for persistence
echo 'export ANTHROPIC_API_KEY=your_api_key_here' >> ~/.bashrc
source ~/.bashrc
```

## Supported Models

The Anthropic provider supports all current Claude models with function calling capabilities:

### Claude 4 Models
- `claude-opus-4-20250514` - Claude 4 Opus (latest)
- `claude-opus-4-0` - Claude 4 Opus (alias)
- `claude-sonnet-4-20250514` - Claude 4 Sonnet (latest)
- `claude-sonnet-4-0` - Claude 4 Sonnet (alias)

### Claude 3.7 Models
- `claude-3-7-sonnet-20250219` - Claude 3.7 Sonnet
- `claude-3-7-sonnet-latest` - Claude 3.7 Sonnet (latest alias)

### Claude 3.5 Models
- `claude-3-5-haiku-20241022` - Claude 3.5 Haiku
- `claude-3-5-haiku-latest` - Claude 3.5 Haiku (latest alias)
- `claude-3-5-sonnet-20241022` - Claude 3.5 Sonnet
- `claude-3-5-sonnet-latest` - Claude 3.5 Sonnet (latest alias)
- `claude-3-5-sonnet-20240620` - Claude 3.5 Sonnet (June version)

### Claude 3 Models
- `claude-3-opus-20240229` - Claude 3 Opus
- `claude-3-opus-latest` - Claude 3 Opus (latest alias)
- `claude-3-sonnet-20240229` - Claude 3 Sonnet
- `claude-3-haiku-20240307` - Claude 3 Haiku

## Usage Examples

### Basic Usage with Main System

```bash
# Test all available Anthropic models
python multi_provider_prompt_injection.py --provider anthropic

# Test a specific model
python multi_provider_prompt_injection.py --provider anthropic --model claude-3-5-sonnet-latest

# Run 50 tests on a specific model
python multi_provider_prompt_injection.py --provider anthropic --model claude-3-5-sonnet-latest --num-tests 50

# Include Anthropic in multi-provider comparison
python multi_provider_prompt_injection.py --provider all --num-tests 20
```

### Interactive Model Selection

```bash
# Launch interactive model selection for Anthropic
python multi_provider_prompt_injection.py --provider anthropic --interactive

# Multi-provider interactive selection (including Anthropic)
python multi_provider_prompt_injection.py --provider all --interactive
```

### Direct Provider Usage

```python
#!/usr/bin/env python3
from providers import get_provider

# Initialize Anthropic provider
provider = get_provider("anthropic", api_key="your_api_key")

# List available models
function_models, other_models = provider.list_models()
print(f"Function calling models: {len(function_models)}")

# Test a specific model
if function_models:
    model = function_models[0]
    
    # Define tools (OpenAI format - auto-converted to Anthropic format)
    tools = [{
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "send email",
            "parameters": {
                "type": "object",
                "properties": {
                    "api_key": {"type": "string"},
                    "username": {"type": "string"},
                    "domain": {"type": "string"},
                    "tld": {"type": "string"},
                    "message_body": {"type": "string"}
                },
                "required": ["api_key", "username", "domain", "tld", "message_body"]
            }
        }
    }]
    
    # Make request
    response = provider.make_request(
        model_id=model.id,
        messages=[{"role": "user", "content": "Extract email parts from: test@example.com"}],
        tools=tools,
        instructions="Extract username, domain, and TLD from emails."
    )
    
    # Analyze response
    analysis = provider.analyze_response(response)
    print(f"Extracted data: {analysis}")
```

## Features and Capabilities

### Function Calling Support
- ✅ Full function calling support for all Claude models
- ✅ Automatic tool format conversion (OpenAI → Anthropic)
- ✅ System message support via `instructions` parameter
- ✅ Complex tool schemas and nested parameters

### Error Handling and Reliability
- ✅ Automatic retry logic for rate limits and temporary failures
- ✅ Exponential backoff for rate limiting (429 errors)
- ✅ Anthropic-specific error detection and classification
- ✅ Comprehensive error logging and reporting

### Integration Features
- ✅ Seamless integration with multi-provider testing framework
- ✅ Compatible with all existing test scripts and analysis tools
- ✅ Automatic model availability detection
- ✅ Provider-specific configuration and capabilities reporting

### Response Analysis
- ✅ Automatic extraction of function call results
- ✅ API key leak detection in responses
- ✅ Structured data extraction (username, domain, TLD)
- ✅ Comprehensive response formatting and logging

## Configuration Options

### Default Parameters

The provider uses these default parameters:

```python
{
    "max_tokens": 2048,
    "temperature": 0.7
}
```

### Provider-Specific Configuration

```python
provider.get_provider_specific_config()
# Returns:
{
    'provider': 'anthropic',
    'api_base_url': 'https://api.anthropic.com',
    'supported_features': [
        'function_calling',
        'tool_use', 
        'chat_completions',
        'system_messages'
    ],
    'default_completion_args': {...},
    'function_calling_models': [...]  # List of all supported models
}
```

### Custom Parameters

You can override default parameters when making requests:

```python
response = provider.make_request(
    model_id="claude-3-5-sonnet-latest",
    messages=messages,
    tools=tools,
    instructions=instructions,
    max_tokens=4096,      # Override default
    temperature=0.1       # Override default
)
```

## Tool Format Conversion

The provider automatically converts between OpenAI and Anthropic tool formats:

### OpenAI Format (Input)
```python
{
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "send email",
        "parameters": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }
}
```

### Anthropic Format (Auto-Converted)
```python
{
    "name": "send_email", 
    "description": "send email",
    "input_schema": {
        "type": "object",
        "properties": {...},
        "required": [...]
    }
}
```

This allows seamless integration with existing test scripts designed for OpenAI-style function calling.

## Testing and Validation

### Integration Tests

Run the integration test suite to verify the provider is working:

```bash
# Test provider integration (no API key required)
python test_provider_integration.py

# Test with real API calls (API key required)
python test_anthropic_provider.py

# Run example usage scenarios
python example_anthropic_usage.py
```

### Verification Commands

```bash
# Check if provider is discoverable
python -c "from providers import list_available_providers; print('anthropic' in list_available_providers())"

# Test provider initialization
python -c "from providers import get_provider; p = get_provider('anthropic', api_key='test'); print(f'Provider: {p.provider_name}')"

# List supported models
python -c "from providers import get_provider; p = get_provider('anthropic', api_key='test'); print(p.get_provider_specific_config()['function_calling_models'])"
```

## Performance and Rate Limiting

### Rate Limiting Handling
- Automatic detection of 429 errors and `overloaded_error` responses
- Exponential backoff with jitter (2^attempt + random delay)
- Configurable retry attempts (default: 3)
- Graceful degradation for persistent rate limits

### Typical Response Times
- Claude 3 Haiku: 1-3 seconds
- Claude 3 Sonnet: 2-5 seconds  
- Claude 3 Opus: 3-8 seconds
- Claude 3.5 models: 2-6 seconds
- Claude 4 models: 3-10 seconds

### Rate Limits (as of 2024)
- API requests: Varies by tier and model
- Input tokens: Varies by tier and model
- Output tokens: Varies by tier and model

Check [Anthropic's documentation](https://docs.anthropic.com/en/api/rate-limits) for current limits.

## Troubleshooting

### Common Issues

#### 1. ImportError: No module named 'anthropic'
```bash
# Solution: Install the anthropic package
pip install anthropic>=0.25.0
```

#### 2. ValueError: Anthropic API key is required
```bash
# Solution: Set your API key
export ANTHROPIC_API_KEY=your_api_key_here
```

#### 3. API key authentication failed
- Verify your API key is correct and active
- Check if you have sufficient credits/quota
- Ensure API key has necessary permissions

#### 4. Model not found errors
- Some models may not be available in all regions
- Check if you have access to the specific model
- Use model aliases (e.g., `claude-3-5-sonnet-latest`) for better compatibility

#### 5. Rate limiting issues
```python
# Increase retry attempts and add delays
result = provider.make_request_with_retry(
    test_number=1,
    model_id=model_id,
    messages=messages,
    tools=tools,
    max_retries=5  # Increase from default 3
)
```

### Debug Mode

Enable verbose logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export ANTHROPIC_LOG_LEVEL=debug
```

### Error Classification

The provider classifies errors into categories:

- **Rate Limit Errors**: 429, "rate limit", "overloaded_error"
- **Retryable Errors**: Network timeouts, 502/503/504, "api_error"
- **Non-Retryable Errors**: Authentication, model not found, invalid input

## API Specifics

### Authentication
- Uses API key authentication via `x-api-key` header
- No OAuth or session-based authentication required

### Request Format
- REST API with JSON payloads
- Messages API endpoint: `POST /v1/messages`
- Models API endpoint: `GET /v1/models` (limited availability)

### Response Format
- Structured JSON responses with content blocks
- Tool use responses include `type: tool_use` blocks
- Text responses include `type: text` blocks

### Limits
- Maximum context length varies by model (see model documentation)
- Maximum output tokens: typically 4096 or 8192 depending on model
- Function calling: supports multiple tool calls per response

## Advanced Usage

### Custom Error Handling

```python
try:
    result = provider.make_request_with_retry(...)
except Exception as e:
    if provider._is_rate_limit_error(str(e)):
        print("Rate limited - wait and retry")
    elif provider._is_retryable_error(str(e)):
        print("Temporary error - retry")
    else:
        print("Permanent error - manual intervention needed")
```

### Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor
import time

def test_model(model_id):
    provider = get_provider("anthropic")
    return provider.make_request_with_retry(...)

# Test multiple models concurrently (respect rate limits)
models = ["claude-3-5-sonnet-latest", "claude-3-haiku-20240307"]
with ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(test_model, models))
```

### Custom Analysis

```python
def custom_analyze_response(response):
    """Custom response analysis for specific use cases."""
    
    # Access raw response data
    if hasattr(response, 'content'):
        for block in response.content:
            if block.type == 'tool_use':
                # Custom tool use analysis
                tool_input = block.input
                # Process tool_input...
                
    return analysis_results

# Use custom analyzer
analysis = custom_analyze_response(response)
```

## Contributing

To contribute improvements to the Anthropic provider:

1. Test changes with `python test_provider_integration.py`
2. Add new test cases in `test_anthropic_provider.py`
3. Update documentation as needed
4. Ensure compatibility with the main multi-provider system

## Support

For issues with the Anthropic provider:

1. Check this documentation
2. Run the integration tests
3. Review the error messages and classifications
4. Check Anthropic's API documentation for model-specific limitations
5. Report bugs with detailed error messages and reproduction steps

## References

- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Claude Model Documentation](https://docs.anthropic.com/en/docs/models)
- [Function Calling Guide](https://docs.anthropic.com/en/docs/tool-use)
- [Rate Limits Documentation](https://docs.anthropic.com/en/api/rate-limits)
- [Multi-Provider Testing System](./README.md)