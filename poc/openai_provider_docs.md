# OpenAI Provider Documentation Patch

## Additional Installation for OpenAI Provider

```bash
# Install OpenAI package for provider support
pip install openai

# Set your OpenAI API key
export OPENAI_API_KEY="your-openai-api-key-here"
```

## OpenAI Provider Commands

### Multi-Provider Prompt Injection Testing

```bash
# Test OpenAI models for prompt injection vulnerabilities
python multi_provider_prompt_injection.py --provider openai

# Test specific OpenAI model
python multi_provider_prompt_injection.py --provider openai --model gpt-4

# Test with multiple iterations
python multi_provider_prompt_injection.py --provider openai --tests 10

# Test with specific GPT-4 variants
python multi_provider_prompt_injection.py --provider openai --model gpt-4-turbo --tests 5

# Interactive model selection for OpenAI
python multi_provider_prompt_injection.py --provider openai --interactive

# Multi-model testing across OpenAI models
python multi_provider_prompt_injection.py --provider openai --multi-model --tests-per-model 3

# Compare OpenAI models side by side
python multi_provider_prompt_injection.py --provider openai --model gpt-4 --compare-with gpt-3.5-turbo

# Test OpenAI with custom instructions
python multi_provider_prompt_injection.py --provider openai --model gpt-4 --custom-instructions

# Batch testing with result export
python multi_provider_prompt_injection.py --provider openai --tests 20 --export results_openai.json

# Test with different temperature settings
python multi_provider_prompt_injection.py --provider openai --model gpt-4 --temperature 0.1 --tests 5

# Test multiple OpenAI models in sequence
python multi_provider_prompt_injection.py --provider openai --models gpt-4,gpt-3.5-turbo,gpt-4-turbo --tests-per-model 3
```

### Cross-Provider Comparison

```bash
# Compare OpenAI vs other providers
python multi_provider_prompt_injection.py --compare-providers openai,mistral,lambda

# Test same model across providers (if available)
python multi_provider_prompt_injection.py --cross-provider-test --model-pattern "gpt-4"

# Generate comparative analysis report
python multi_provider_prompt_injection.py --providers openai,mistral --analysis-report comparison_report.html
```

### OpenAI-Specific Testing

```bash
# Test OpenAI provider integration
python test_openai_provider.py

# Integration test with main framework
python test_openai_integration.py

# Test specific OpenAI models for function calling
python test_openai_provider.py --model gpt-4 --test-function-calling

# Test OpenAI rate limiting behavior
python test_openai_provider.py --stress-test --concurrent-requests 5

# Test OpenAI error handling
python test_openai_provider.py --test-error-scenarios
```

### Advanced OpenAI Configuration

```bash
# Test with custom OpenAI parameters
python multi_provider_prompt_injection.py --provider openai --model gpt-4 \
  --temperature 0.2 --max-tokens 1024 --top-p 0.9

# Test with different system instructions
python multi_provider_prompt_injection.py --provider openai --model gpt-4 \
  --system-message "You are a security-focused AI assistant"

# Test OpenAI with retry configuration
python multi_provider_prompt_injection.py --provider openai --model gpt-4 \
  --max-retries 5 --retry-delay 2

# Test with OpenAI-specific features
python multi_provider_prompt_injection.py --provider openai --model gpt-4 \
  --enable-function-calling --test-vision-capabilities
```

## OpenAI Provider Configuration

### Environment Variables
```bash
# Required
export OPENAI_API_KEY="your-api-key-here"

# Optional
export OPENAI_ORG_ID="your-org-id"  # For organization-specific usage
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Custom endpoint if needed
```

### Supported OpenAI Models
The provider automatically detects function-calling capable models:

**Function Calling Models:**
- gpt-4
- gpt-4-turbo
- gpt-4o
- gpt-4o-mini
- gpt-3.5-turbo
- gpt-3.5-turbo-16k

**Non-Function Calling Models:**
- whisper-1 (audio)
- tts-1 (text-to-speech)
- dall-e-3 (image generation)
- text-embedding-ada-002 (embeddings)

### OpenAI-Specific Features

```bash
# Test multimodal capabilities (if supported)
python multi_provider_prompt_injection.py --provider openai --model gpt-4-vision-preview \
  --test-image-injection

# Test with streaming responses
python multi_provider_prompt_injection.py --provider openai --model gpt-4 \
  --enable-streaming --tests 3

# Test with different API versions
python multi_provider_prompt_injection.py --provider openai --model gpt-4 \
  --api-version "2024-02-15-preview"
```

### Rate Limiting and Error Handling

```bash
# Test with rate limit handling
python multi_provider_prompt_injection.py --provider openai --model gpt-4 \
  --tests 100 --batch-size 5 --delay-between-batches 1

# Test quota and billing error handling
python test_openai_provider.py --test-quota-limits

# Test network resilience
python test_openai_provider.py --test-network-errors --retry-scenarios
```

## Provider Comparison Examples

### Example: Testing Prompt Injection Across Providers

```bash
# Compare how different providers handle the same prompt injection
python multi_provider_prompt_injection.py \
  --providers openai,mistral,lambda \
  --model-per-provider "gpt-4,mixtral-8x7b-instruct,hermes3-70b" \
  --tests 10 \
  --export comparison_results.json

# Generate detailed analysis
python analyze_provider_comparison.py comparison_results.json \
  --output-format html \
  --include-statistics \
  --generate-charts
```

### Example: OpenAI Model Comparison

```bash
# Compare different OpenAI models
python multi_provider_prompt_injection.py --provider openai \
  --models "gpt-4,gpt-4-turbo,gpt-3.5-turbo" \
  --tests-per-model 15 \
  --compare-models \
  --export openai_model_comparison.json
```

## Troubleshooting OpenAI Provider

### Common Issues

1. **API Key Issues:**
```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Test API key validity
python test_openai_provider.py --verify-api-key
```

2. **Rate Limiting:**
```bash
# Test with lower request rate
python multi_provider_prompt_injection.py --provider openai \
  --tests 10 --delay-between-requests 2
```

3. **Model Access:**
```bash
# List available models
python test_openai_provider.py --list-models

# Test specific model access
python test_openai_provider.py --test-model gpt-4 --verify-access
```

4. **Quota Issues:**
```bash
# Check usage and limits
python test_openai_provider.py --check-usage --check-limits
```

### Debug Mode

```bash
# Enable verbose logging
python multi_provider_prompt_injection.py --provider openai \
  --debug --verbose --log-level DEBUG

# Save detailed logs
python multi_provider_prompt_injection.py --provider openai \
  --tests 5 --log-file openai_debug.log
```

## Integration with Existing Framework

The OpenAI provider seamlessly integrates with all existing glitcher commands:

```bash
# All existing multi-provider commands now support --provider openai
# All analysis tools work with OpenAI results
# All export formats support OpenAI data
# All comparison tools include OpenAI in analysis
```

## Performance Notes

- OpenAI models typically have higher latency than local models
- Rate limits vary by model tier and account type
- Function calling adds minimal overhead
- Streaming can improve perceived performance for long responses
- Batch processing recommended for large test suites

## Security Considerations

- API keys are sensitive - use environment variables
- OpenAI logs requests for safety monitoring
- Test results may contain sensitive prompt injection data
- Consider using separate API keys for testing vs production
- Review OpenAI's usage policies for prompt injection research