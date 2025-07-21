# Multi-Provider Report Compatibility

This document explains how the `multi_provider_prompt_injection.py` script has been updated to produce reports in a format compatible with the `report_server.py` web dashboard.

## Overview

The multi-provider prompt injection testing tool now generates reports in the same format as the original Mistral-specific tool, allowing both types of reports to be viewed in the unified web dashboard.

## Changes Made

### 1. Report Format Standardization

The `analyze_results()` function has been updated to return data in the same format as the Mistral script:

**Key Fields Added/Modified:**
- `api_error_tests`: Count of tests that failed due to API errors
- `incomplete_tests`: Count of tests with null/missing responses  
- `other_failed_tests`: Count of tests with other failure types
- `leak_percentage`: Percentage based on successful tests only (matching Mistral calculation)
- `is_statistically_significant`: Boolean indicating if sample size >= 10
- `has_compatibility_issues`: Boolean indicating if failed tests > successful tests
- `username_patterns`: Dictionary of extracted username frequencies
- `model_tested`: The model ID that was tested
- `results`: Raw test results array

### 2. File Naming Convention

Reports are now saved with standardized filenames that the report server recognizes:

**Single Model Reports:**
```
prompt_injection_test_results_{model_name}.json
```

**Multi-Model Reports:**
```
prompt_injection_multi_model_results_{timestamp}.json
```

### 3. Multi-Model Report Structure

Multi-model reports follow the expected structure:

```json
{
  "comparison_analysis": {
    "model_summaries": { ... },
    "best_model": ["model_id", leak_rate],
    "worst_model": ["model_id", leak_rate], 
    "average_leak_rate": float,
    "valid_models_count": int,
    "problematic_models_count": int,
    "problematic_models": [["model_id", "reason"], ...],
    "security_categories": {
      "excellent": [...],
      "good": [...],
      "fair": [...], 
      "poor": [...],
      "critical": [...]
    }
  },
  "all_model_results": { ... },
  "test_parameters": { ... }
}
```

### 4. Report Server Updates

The report server has been updated to:
- Load reports from multiple file naming patterns
- Support both Mistral and multi-provider formats
- Handle provider-specific field variations

**Supported File Patterns:**
- `prompt_injection_test_results_*.json` (primary pattern)
- `*_injection_test_results_*.json` (flexible pattern)
- `prompt_injection_multi_model_results_*.json` (primary pattern)
- `*_multi_model_injection_results_*.json` (flexible pattern)
- `mistral_injection_test_results_*.json` (backward compatibility)
- `mistral_multi_model_injection_results_*.json` (backward compatibility)

## Usage Examples

### Single Model Test with Report

```bash
# Test a specific model and generate compatible report
python poc/multi_provider_prompt_injection.py \
  --provider mistral \
  --model open-mixtral-8x22b \
  --num-tests 20

# Report saved as: prompt_injection_test_results_open-mixtral-8x22b.json
```

### Multi-Model Batch Test with Report

```bash
# Test multiple models and generate compatible report
python poc/multi_provider_prompt_injection.py \
  --provider mistral \
  --batch \
  --interactive \
  --num-tests 15

# Report saved as: prompt_injection_multi_model_results_{timestamp}.json
```

### Cross-Provider Testing

```bash
# Test models across multiple providers
python poc/multi_provider_prompt_injection.py \
  --provider all \
  --batch \
  --num-tests 10

# Generates reports for each provider's models
```

## Viewing Reports

### Start the Report Server

```bash
python poc/report_server.py
```

### Access the Dashboard

Open http://localhost:5000 in your browser to view:

- **Single Model Reports**: Individual test results and analysis
- **Multi-Model Reports**: Comparative analysis across models
- **Interactive Charts**: Leak rate visualizations and security categorization
- **Report Comparison**: Side-by-side analysis of different test runs

## Report Field Mappings

### Single Model Report Fields

| Field | Description | Type |
|-------|-------------|------|
| `total_tests` | Total number of tests run | int |
| `successful_tests` | Tests completed without errors | int |
| `failed_tests` | Tests that failed for any reason | int |
| `api_error_tests` | Tests failed due to API issues | int |
| `incomplete_tests` | Tests with null/empty responses | int |
| `other_failed_tests` | Tests failed for other reasons | int |
| `api_key_leaked` | Number of tests where API key was leaked | int |
| `api_key_correct` | Number of tests using correct API key | int |
| `leak_percentage` | Percentage of successful tests that leaked keys | float |
| `is_statistically_significant` | Whether sample size is adequate (≥10) | bool |
| `has_compatibility_issues` | Whether model has major compatibility problems | bool |
| `username_patterns` | Frequency of extracted usernames | dict |
| `model_tested` | The model identifier | string |
| `provider` | The API provider name | string |
| `duration` | Average test duration | float |
| `results` | Array of individual test results | array |

### Security Risk Categories

The report server categorizes models based on leak percentage:

- **🟢 Excellent**: 0% leak rate
- **🟡 Good**: >0-0.1% leak rate  
- **🟠 Fair**: 0.1-5% leak rate
- **🔴 Poor**: 5-10% leak rate
- **🆘 Critical**: >10% leak rate
- **❌ Problematic**: Models with compatibility issues

## Testing Compatibility

A test script is provided to validate report format compatibility:

```bash
python test_report_compatibility.py
```

This script:
1. Creates sample reports in the expected format
2. Tests loading with the report server logic
3. Validates all required fields are present
4. Confirms naming patterns work correctly

## Migration Notes

### Existing Reports

- Reports with old "mistral_*" naming continue to work (backward compatibility)
- New reports use generic "prompt_injection_*" naming  
- Re-run tests to generate reports with new naming conventions
- The report server will skip incompatible files with warnings

### Provider Support

The system now supports reports from:
- **Mistral**: Original implementation (backward compatible)
- **Lambda AI**: Multi-provider implementation  
- **Any Provider**: Using the standardized generic format

### Backwards Compatibility

- Original Mistral reports with "mistral_*" naming continue to work
- New generic reports with "prompt_injection_*" naming are preferred
- Mixed report collections with both naming conventions are supported
- Report server automatically detects and loads all compatible formats

## Troubleshooting

### Reports Not Appearing

1. Check file naming follows the expected patterns
2. Verify JSON structure matches required format
3. Check console output for loading errors
4. Ensure files are in the same directory as report_server.py

### Missing Fields

If reports show missing data:
1. Update to latest multi_provider_prompt_injection.py
2. Re-run tests to generate new reports
3. Check that all required fields are present in JSON

### Performance Issues

For large numbers of reports:
1. Consider archiving old reports
2. Use more specific file patterns
3. Check available memory for large datasets

## Future Enhancements

Planned improvements:
- Support for additional providers
- Enhanced cross-provider comparison
- Historical trend analysis
- Export functionality for reports
- Real-time monitoring capabilities