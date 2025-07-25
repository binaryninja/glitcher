# Enhanced Glitch Classification System Guide

## Overview

The enhanced glitch classification system has been completely rewritten to address the issues you encountered:

1. ✅ **Fixed "running twice" problem** - No more redundant model loading
2. ✅ **Detailed test failure information** - Shows exactly which tests failed and why
3. ✅ **Integrated email/domain analysis** - Full classification now includes the same detailed analysis as specialized modes

## Quick Start

### Using the New Enhanced Classification

```bash
# Enhanced classification with detailed test failure information
glitcher classify meta-llama/Llama-3.2-1B-Instruct --token-ids 89472,127438,85069

# With debug mode for response previews
glitcher classify meta-llama/Llama-3.2-1B-Instruct --token-ids 89472 --debug-responses

# From a file of token IDs
glitcher classify meta-llama/Llama-3.2-1B-Instruct --token-file glitch_tokens.json
```

### What's Different from the Old System

**Before (old system):**
```
Token               Injection  IDOS  Hallucination  ...  Notes
'sometoken'         ✅         ❌    ✅             ...  Injection, Hallucination
```

**After (enhanced system):**
```
1. Token: 'sometoken' (ID: 89472)
------------------------------------------------------------
   Categories: Injection, EmailExtraction

   Injection Tests:
     ✅ injection_test (triggered: glitch_injection_pattern)

   EmailExtraction Tests:
     ✅ email_extraction_test (triggered: detailed_email_analysis)
        Expected: jeremy<token>@richards.ai
        Issues: incorrect_username, missing_tld
        Response: {"username": "jeremy", "domain": "richards.ai"}...
```

## Commands Available

### 1. Full Enhanced Classification
```bash
glitcher classify meta-llama/Llama-3.2-1B-Instruct --token-ids 89472,127438,85069
```
**Features:**
- Shows detailed test failure information
- Includes integrated email/domain extraction analysis
- Displays which indicators triggered for each test
- No redundant model loading

### 2. Email Extraction Only
```bash
glitcher classify meta-llama/Llama-3.2-1B-Instruct --token-ids 89472 --email-extraction-only
```
**Features:**
- Focused on email extraction testing
- Detailed JSON parsing analysis
- Shows expected vs actual extraction results

### 3. Domain Extraction Only
```bash
glitcher classify meta-llama/Llama-3.2-1B-Instruct --token-ids 127438 --domain-extraction-only
```
**Features:**
- Focused on domain extraction from log files
- Detailed log parsing analysis
- Shows expected vs actual domain extraction

### 4. Behavioral Tests Only
```bash
glitcher classify meta-llama/Llama-3.2-1B-Instruct --token-ids 89472 --behavioral-only
```
**Features:**
- Only runs injection, IDOS, hallucination, disruption, bypass tests
- Faster execution for behavioral analysis

### 5. Functional Tests Only
```bash
glitcher classify meta-llama/Llama-3.2-1B-Instruct --token-ids 89472 --functional-only
```
**Features:**
- Only runs email/domain extraction tests
- Includes detailed extraction analysis

## Command Line Options

```bash
glitcher classify MODEL_PATH [OPTIONS]
```

### Required Arguments
- `MODEL_PATH`: Path or name of the model (e.g., `meta-llama/Llama-3.2-1B-Instruct`)
- One of:
  - `--token-ids`: Comma-separated token IDs (e.g., `89472,127438,85069`)
  - `--token-file`: JSON file with token IDs

### Optional Arguments

#### Model Configuration
- `--device`: Device to use (`cuda`, `cpu`) [default: `cuda`]
- `--quant-type`: Quantization (`bfloat16`, `float16`, `int8`, `int4`) [default: `bfloat16`]
- `--temperature`: Temperature for inference [default: `0.0`]
- `--max-tokens`: Max tokens to generate per test [default: `200`]

#### Test Modes
- `--email-extraction-only`: Only email extraction tests
- `--domain-extraction-only`: Only domain extraction tests
- `--behavioral-only`: Only behavioral tests (injection, IDOS, etc.)
- `--functional-only`: Only functional tests (email/domain)

#### Output Options
- `--output`: Output file [default: `classified_tokens.json`]
- `--debug-responses`: Show detailed response previews

## Enhanced Output Format

### Summary Table
The new summary shows detailed information for each token:

```
================================================================================
CLASSIFICATION SUMMARY
================================================================================

1. Token: 'SomeGlitchToken' (ID: 89472)
------------------------------------------------------------
   Categories: EmailExtraction, ValidEmailAddress

   EmailExtraction Tests:
     ✅ email_extraction_test (triggered: detailed_email_analysis)
        Expected: jeremySomeGlitchToken@richards.ai
        Issues: incorrect_username, missing_tld
        Response: {"username": "jeremy", "domain": "richards.ai"}...

   ValidEmailAddress Tests:
     ✅ valid_email_address_test (creates valid email addresses)

================================================================================
SUMMARY STATISTICS
================================================================================
Total tokens tested: 3
Tokens by category:
  EmailExtraction    :   2 ( 66.7%)
  ValidEmailAddress  :   1 ( 33.3%)
  No glitch detected :   1 ( 33.3%)
```

### JSON Output
The saved JSON file includes:

```json
{
  "model_path": "meta-llama/Llama-3.2-1B-Instruct",
  "total_tokens": 3,
  "category_counts": {
    "EmailExtraction": 2,
    "ValidEmailAddress": 1
  },
  "classifications": [
    {
      "token_id": 89472,
      "token": "SomeGlitchToken",
      "categories": ["EmailExtraction", "ValidEmailAddress"],
      "test_results": [
        {
          "test_name": "email_extraction_test",
          "category": "EmailExtraction",
          "is_positive": true,
          "indicators": {"detailed_email_analysis": true},
          "metadata": {
            "detailed_analysis": {
              "expected_email": "jeremySomeGlitchToken@richards.ai",
              "issues": ["incorrect_username", "missing_tld"],
              "response_preview": "{\"username\": \"jeremy\", \"domain\": \"richards.ai\"}..."
            }
          }
        }
      ]
    }
  ],
  "detailed_email_results": {
    "89472": {
      "is_valid": false,
      "issues": ["incorrect_username", "missing_tld"],
      "expected_email": "jeremySomeGlitchToken@richards.ai"
    }
  }
}
```

## Migration from Old System

### If you were using:
```bash
glitch-classify meta-llama/Llama-3.2-1B-Instruct --token-ids 89472
```

### Now use:
```bash
glitcher classify meta-llama/Llama-3.2-1B-Instruct --token-ids 89472
```

## Key Improvements

### 1. No More Redundant Loading
- ✅ Model loads only once
- ✅ Clean log output without duplicates
- ✅ Faster execution

### 2. Detailed Test Information
- ✅ See which specific tests failed
- ✅ Understand why tests failed
- ✅ View triggered indicators
- ✅ Access response previews in debug mode

### 3. Integrated Detailed Analysis
- ✅ Email extraction analysis in full classification
- ✅ Domain extraction analysis in full classification
- ✅ Same level of detail as specialized modes
- ✅ Stored in test metadata for programmatic access

### 4. Better Error Handling
- ✅ Graceful error handling in detailed analysis
- ✅ Fallback to basic checks if detailed analysis fails
- ✅ Clear error messages

## Examples

### Basic Classification
```bash
glitcher classify meta-llama/Llama-3.2-1B-Instruct --token-ids 89472,127438,85069
```

### Debug Mode with Response Previews
```bash
glitcher classify meta-llama/Llama-3.2-1B-Instruct --token-ids 89472 --debug-responses
```

### From Token File
```bash
# Create token file
echo '{"glitch_token_ids": [89472, 127438, 85069]}' > my_tokens.json

# Classify
glitcher classify meta-llama/Llama-3.2-1B-Instruct --token-file my_tokens.json
```

### Email Extraction Focus
```bash
glitcher classify meta-llama/Llama-3.2-1B-Instruct --token-ids 89472 --email-extraction-only
```

### Different Model Settings
```bash
glitcher classify meta-llama/Llama-3.2-1B-Instruct --token-ids 89472 \
  --temperature 0.1 --max-tokens 500 --device cuda --quant-type float16
```

## Troubleshooting

### Error: "Must specify either --token-ids or --token-file"
**Solution**: Provide token IDs to classify:
```bash
glitcher classify MODEL --token-ids 89472,127438
# or
glitcher classify MODEL --token-file tokens.json
```

### Error: "Token file not found"
**Solution**: Ensure your token file exists and has valid JSON:
```json
{"glitch_token_ids": [89472, 127438, 85069]}
```

### Error: "Model not found"
**Solution**: Check model path/name:
```bash
# For local models
glitcher classify ./path/to/model --token-ids 89472

# For HuggingFace models
glitcher classify meta-llama/Llama-3.2-1B-Instruct --token-ids 89472
```

### No Categories Detected
**Possible causes:**
1. Token IDs may not be glitch tokens
2. Model behavior may be different
3. Check debug output with `--debug-responses`

## Advanced Usage

### Custom Output Location
```bash
glitcher classify MODEL --token-ids 89472 --output /path/to/results.json
```

### Batch Processing Multiple Files
```bash
for file in token_files/*.json; do
    glitcher classify MODEL --token-file "$file" --output "results_$(basename $file)"
done
```

### Integration with Mining Results
```bash
# Mine tokens
glitcher mine MODEL --num-iterations 50 --output mined_tokens.json

# Classify mined tokens
glitcher classify MODEL --token-file mined_tokens.json --output classified_results.json
```

## Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster processing
2. **Adjust batch size**: For many tokens, consider processing in batches
3. **Choose appropriate quant-type**: `bfloat16` is usually best balance of speed/quality
4. **Use focused modes**: Use `--behavioral-only` or `--functional-only` for faster execution when you only need specific tests

## Support

If you encounter issues:
1. Check this guide for common solutions
2. Use `--debug-responses` to see detailed output
3. Check the saved JSON file for detailed analysis results
4. Review the classification log file for detailed debugging information