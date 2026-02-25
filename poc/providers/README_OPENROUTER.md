# OpenRouter Provider

The OpenRouter Provider enables you to access 200+ AI models from multiple providers through a single unified API endpoint. OpenRouter automatically handles fallbacks, routing, and cost optimization across providers.

## Features

- **200+ Models**: Access models from OpenAI, Anthropic, Google, Meta, Mistral, and many more
- **Unified API**: Single endpoint for all models using OpenAI-compatible format
- **Automatic Fallbacks**: Built-in reliability with automatic provider failover
- **Cost Optimization**: Automatically routes to the most cost-effective provider
- **Function Calling**: Support for function/tool calling on compatible models
- **Vision Models**: Access to multimodal models with image understanding
- **Web Search**: Models with real-time web search capabilities
- **Simple Integration**: Works with OpenAI SDK out of the box

## Installation

The OpenRouter provider uses the OpenAI SDK:

```bash
pip install openai
```

## Quick Start

### Basic Usage

```python
from providers import get_provider

# Initialize with OpenRouter
provider = get_provider(
    'openrouter',
    api_key='your-openrouter-api-key',  # or set OPENROUTER_API_KEY env var
    site_url='https://your-site.com',   # Optional: for attribution
    site_name='Your App Name'           # Optional: for attribution
)

# Make a request using any available model
messages = [
    {"role": "user", "content": "Hello! Tell me about OpenRouter."}
]

response = provider.make_request(
    model_id='openai/gpt-4o',  # Use any model from openrouter.ai/models
    messages=messages,
    max_tokens=100
)

print(response.choices[0].message.content)
```

### Using Different Providers' Models

```python
# OpenAI models
response = provider.make_request(
    model_id='openai/gpt-4o',
    messages=messages
)

# Anthropic models
response = provider.make_request(
    model_id='anthropic/claude-3.5-sonnet',
    messages=messages
)

# Google models
response = provider.make_request(
    model_id='google/gemini-pro-1.5',
    messages=messages
)

# Meta Llama models
response = provider.make_request(
    model_id='meta-llama/llama-3.2-3b-instruct',
    messages=messages
)

# Mistral models
response = provider.make_request(
    model_id='mistralai/mixtral-8x22b-instruct',
    messages=messages
)
```

## Configuration Options

### Provider Initialization

```python
provider = get_provider(
    'openrouter',
    api_key='your-api-key',           # Required: OpenRouter API key
    site_url='https://your-site.com', # Optional: For attribution/rankings
    site_name='Your App Name'         # Optional: For attribution/rankings
)
```

### Request Parameters

```python
response = provider.make_request(
    model_id='openai/gpt-4o',
    messages=messages,
    max_tokens=1000,         # Maximum tokens to generate
    temperature=0.7,         # Sampling temperature (0.0-2.0)
    top_p=0.9,              # Nucleus sampling
    stream=False,           # Streaming responses (if needed)
    tools=[...],            # Function/tool definitions
    instructions="..."      # System prompt
)
```

## Available Models

OpenRouter provides access to 200+ models. Here are some popular options:

### OpenAI Models
- `openai/gpt-4o` - GPT-4 Optimized
- `openai/gpt-4o-mini` - GPT-4 Optimized Mini
- `openai/gpt-4-turbo` - GPT-4 Turbo
- `openai/gpt-3.5-turbo` - GPT-3.5 Turbo

### Anthropic Models
- `anthropic/claude-3.5-sonnet` - Claude 3.5 Sonnet
- `anthropic/claude-3.5-haiku` - Claude 3.5 Haiku
- `anthropic/claude-3-opus` - Claude 3 Opus
- `anthropic/claude-3-sonnet` - Claude 3 Sonnet

### Google Models
- `google/gemini-pro-1.5` - Gemini Pro 1.5
- `google/gemini-flash-1.5` - Gemini Flash 1.5
- `google/gemini-pro` - Gemini Pro

### Meta Llama Models
- `meta-llama/llama-3.3-70b-instruct` - Llama 3.3 70B
- `meta-llama/llama-3.2-3b-instruct` - Llama 3.2 3B
- `meta-llama/llama-3.1-405b-instruct` - Llama 3.1 405B
- `meta-llama/llama-3.1-70b-instruct` - Llama 3.1 70B

### Other Notable Models
- `mistralai/mixtral-8x22b-instruct` - Mixtral 8x22B
- `deepseek/deepseek-r1` - DeepSeek R1
- `qwen/qwen-2.5-72b-instruct` - Qwen 2.5 72B
- `x-ai/grok-2` - Grok 2
- `perplexity/llama-3.1-sonar-large-128k-online` - Perplexity with web search

For the complete and up-to-date list, visit: https://openrouter.ai/models

## Function Calling

Most modern models on OpenRouter support function calling:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = provider.make_request(
    model_id='openai/gpt-4o',
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools
)

# Check for function calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
```

## Testing the Provider

### Basic Test

```bash
# Test with default model (GPT-4o Mini)
python test_openrouter_provider.py

# Test with specific model
python test_openrouter_provider.py openai/gpt-4o
python test_openrouter_provider.py anthropic/claude-3.5-sonnet
python test_openrouter_provider.py meta-llama/llama-3.2-3b-instruct

# List available models
python test_openrouter_provider.py --list-models

# Test function calling
python test_openrouter_provider.py openai/gpt-4o --test-function-calling
```

### Integration with Multi-Provider Framework

```python
from providers import get_provider

# Compare responses across different models
models = [
    'openai/gpt-4o',
    'anthropic/claude-3.5-sonnet',
    'google/gemini-pro-1.5',
    'meta-llama/llama-3.1-70b-instruct'
]

provider = get_provider('openrouter')

for model_id in models:
    try:
        response = provider.make_request(
            model_id=model_id,
            messages=[{"role": "user", "content": "Explain quantum computing briefly"}],
            max_tokens=100
        )
        print(f"\n{model_id}:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"\n{model_id}: Error - {e}")
```

## API Key and Credits

1. **Get API Key**: Sign up at https://openrouter.ai and get your API key from https://openrouter.ai/keys

2. **Set Environment Variable**:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

3. **Add Credits**: Some models require credits. Add them at https://openrouter.ai/settings/credits

4. **Free Models**: Many models offer free tiers with rate limits. Check model pricing at https://openrouter.ai/models

## Cost Optimization

OpenRouter automatically optimizes costs by:
- Routing to the cheapest available provider for a model
- Handling rate limits and failures transparently
- Providing detailed usage tracking

## Attribution

Setting `site_url` and `site_name` allows your app to appear on OpenRouter leaderboards:

```python
provider = get_provider(
    'openrouter',
    site_url='https://github.com/your-project',
    site_name='Your Project Name'
)
```

## Error Handling

```python
try:
    response = provider.make_request(
        model_id='openai/gpt-4o',
        messages=messages
    )
except ValueError as e:
    if "model not found" in str(e).lower():
        print("Model not available on OpenRouter")
    elif "insufficient credits" in str(e).lower():
        print("Please add credits at https://openrouter.ai/settings/credits")
    else:
        print(f"Error: {e}")
```

## Advantages of OpenRouter

1. **Single API Key**: Access all providers with one key
2. **Automatic Fallbacks**: Built-in reliability
3. **Cost Tracking**: Detailed usage and cost analytics
4. **No Vendor Lock-in**: Switch between providers easily
5. **Rate Limit Handling**: Automatic routing around limits
6. **Unified Billing**: Single invoice for all providers

## Limitations

- **Credits Required**: Some premium models require credits
- **Rate Limits**: Free tier has rate limits
- **Model Availability**: Some models may be temporarily unavailable
- **Regional Restrictions**: Some models may have geographic restrictions

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Ensure OPENROUTER_API_KEY is set
   - Or pass api_key parameter explicitly

2. **Model Not Found**
   - Check model ID at https://openrouter.ai/models
   - Ensure correct format: `provider/model-name`

3. **Insufficient Credits**
   - Add credits at https://openrouter.ai/settings/credits
   - Check free tier availability for the model

4. **Rate Limiting**
   - Wait and retry
   - Consider upgrading to paid tier
   - Use different models as fallback

## See Also

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [Available Models](https://openrouter.ai/models)
- [API Reference](https://openrouter.ai/docs/api-reference)
- [Pricing](https://openrouter.ai/pricing)
- [Multi-Provider Framework Documentation](../README.md)