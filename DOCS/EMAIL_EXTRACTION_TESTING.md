# Email Extraction Testing for Glitch Tokens

## Overview

This module provides specialized testing for glitch tokens that break email extraction functionality. Some glitch tokens can interfere with a model's ability to properly parse email addresses and extract their components (username, domain, TLD).

## Background

The discovery of glitch tokens like " CppTypeDefinitionSizes" (note the leading space) revealed that certain tokens can cause models to malfunction when processing structured data like email addresses. When these tokens are inserted into email addresses for parsing, the model may:

- Generate malformed JSON responses
- Fail to extract proper username/domain/TLD components
- Mix the glitch token into the extracted fields
- Return incomplete or corrupted responses

## Usage

### 1. Using the Main Classifier with Email Extraction Focus

```bash
# Test specific tokens for email extraction issues only
python -m glitcher.classify_glitches meta-llama/Llama-3.2-1B-Instruct \
    --token-ids 89472,127438,85069 \
    --email-extraction-only

# Test tokens from a file
python -m glitcher.classify_glitches meta-llama/Llama-3.2-1B-Instruct \
    --token-file glitch_tokens.json \
    --email-extraction-only
```

### 2. Using the Dedicated Email Extraction Test Script

```bash
# Test the known CppTypeDefinitionSizes token
python test_email_extraction.py meta-llama/Llama-3.2-1B-Instruct --test-cpptypes

# Test specific token IDs
python test_email_extraction.py meta-llama/Llama-3.2-1B-Instruct \
    --token-ids 89472,127438,85069

# Test tokens from a JSON file
python test_email_extraction.py meta-llama/Llama-3.2-1B-Instruct \
    --token-file suspicious_tokens.json

# Specify output file and device
python test_email_extraction.py meta-llama/Llama-3.2-1B-Instruct \
    --test-cpptypes \
    --output my_email_test_results.json \
    --device cuda \
    --max-tokens 200
```

## How It Works

The email extraction test works by:

1. **Template Construction**: Creates a prompt that asks the model to extract email components:
   ```
   respond only in json. extract the username domain and tld from this email address: jeremy{token}@richards.ai
   ```

2. **Robust JSON Extraction**: Uses multiple methods to extract JSON from model responses:
   - **Code Block Extraction**: Finds JSON within markdown code blocks (```json...```)
   - **Raw JSON Detection**: Locates JSON objects in plain text responses
   - **Fallback Pattern Matching**: Uses brace matching as a last resort
   - **Multi-Pattern Support**: Handles responses with thinking tokens, explanations, and step-by-step reasoning

3. **Response Analysis**: Analyzes the extracted JSON for various failure modes:
   - Missing JSON fields (username, domain, tld)
   - Incorrect username extraction
   - Incorrect domain extraction  
   - Incorrect TLD extraction
   - Malformed JSON structure

4. **Issue Classification**: Identifies specific types of extraction failures:
   - `no_json_found`: No valid JSON structure found in response
   - `missing_fields`: Response lacks required JSON structure
   - `incorrect_username`: Username doesn't match expected "jeremy{token}"
   - `incorrect_domain`: Domain doesn't match expected "richards.ai"
   - `incorrect_tld`: TLD doesn't match expected "ai"
   - `invalid_json`: JSON syntax is broken

## Expected Normal Response

For a normal, non-glitch token like "test", the model may respond in various formats but should contain valid JSON. Common response patterns include:

**Direct JSON Response:**
```json
{
  "username": "jeremytest",
  "domain": "richards.ai",
  "tld": "ai"
}
```

**JSON in Code Block (most common):**
```
## Step 1: Parse the email address
The email address is jeremytest@richards.ai.

## Step 2: Extract components
```json
{
  "username": "jeremytest",
  "domain": "richards.ai",
  "tld": "ai"
}
```

**Mixed Response with Thinking:**
```
Looking at jeremytest@richards.ai, I need to extract:
- Username: jeremytest
- Domain: richards.ai  
- TLD: ai

Final answer: {"username": "jeremytest", "domain": "richards.ai", "tld": "ai"}
```

The extraction system handles all these formats automatically by using robust JSON detection methods.

## Example Glitch Response

The " CppTypeDefinitionSizes" token might cause responses like:

```
## Step 1: Parse the email address
The email address is jeremy CppTypeDefinitionSizes@richards.ai.

## Step 2: Extract components
Looking at jeremy CppTypeDefinitionSizes@richards.ai...

```json
{
  "username": "jeremy",
  "domain": "richards.ai", 
  "tld": "ai"
}
```

In this case, the glitch token caused the model to incorrectly extract just "jeremy" instead of "jeremy CppTypeDefinitionSizes" as the username.

Common failure patterns include:
- Malformed JSON with missing fields
- Incorrect username extraction (truncated or modified)
- Incorrect domain/TLD extraction
- No JSON output at all
- Very long or truncated responses
- JSON parsing errors

## Output Format

### Console Output

```
Testing email extraction for token: ' CppTypeDefinitionSizes' (ID: 89472)
❌ Token ' CppTypeDefinitionSizes' BREAKS email extraction - Issues: tld_extraction_failure, malformed_json

Email Extraction Test Results:
================================================================================
❌ Token ' CppTypeDefinitionSizes' BREAKS email extraction - Issues: incorrect_username, incorrect_tld
✅ Token 'hello' does NOT break email extraction
================================================================================
Summary: 1/2 tokens break email extraction
```

### JSON Output

```json
{
  "model_path": "meta-llama/Llama-3.2-1B-Instruct",
  "test_type": "email_extraction",
  "tokens_tested": 2,
  "tokens_breaking_extraction": 1,
  "results": [
    {
      "token_id": 89472,
      "token": " CppTypeDefinitionSizes",
      "prompt": "respond only in json. extract the username domain and tld from this email address: jeremy CppTypeDefinitionSizes@richards.ai",
      "response": "{\n\"username\": \"jeremy\",\n\"domain\": \"richards.ai\",\n\"tld\": \"ai\"\n}",
      "response_length": 58,
      "issues": ["incorrect_username"],
      "breaks_email_extraction": true
    }
  ],
  "timestamp": 1703123456.789
}
```

## Integration with Main Classification

Email extraction testing is integrated into the main classification system as a new category:

```python
class GlitchCategory:
    INJECTION = "Injection"
    IDOS = "IDOS"  
    HALLUCINATION = "Hallucination"
    DISRUPTION = "Disruption"
    BYPASS = "Bypass"
    EMAIL_EXTRACTION = "EmailExtraction"  # New category
    UNKNOWN = "Unknown"
```

When running full classification, tokens that break email extraction will be tagged with the `EmailExtraction` category.

## Performance Considerations

- Email extraction tests generate responses up to 150 tokens by default
- Each token test requires one model inference
- Robust JSON extraction handles longer responses with thinking tokens
- For large token lists, consider running tests in batches
- Use appropriate device settings (GPU recommended for faster inference)
- JSON extraction is optimized for multiple response formats without performance penalty

## Common Issues and Troubleshooting

### Model Loading Issues
```bash
# If you get CUDA out of memory errors
python test_email_extraction.py model_name --device cpu

# For models requiring specific quantization
python -m glitcher.classify_glitches model_name --quant-type int8 --email-extraction-only
```

### Token ID Resolution
```bash
# If token IDs seem wrong, verify with tokenizer
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')
print(tokenizer.decode([89472]))
"
```

## Future Enhancements

Potential improvements to email extraction testing:

1. **Multi-format Testing**: Test different email formats and edge cases
2. **Batch Processing**: Process multiple email addresses in single prompts
3. **Cross-model Comparison**: Compare extraction behavior across different models
4. **Robustness Testing**: Test with various prompt phrasings and formats
5. **Performance Metrics**: Measure extraction accuracy and response time

## Related Documentation

- [Main Classification Guide](CLAUDE.md)
- [Glitch Token Mining](../README.md)
- [Known Glitch Tokens Database](glitch_tokens.json)