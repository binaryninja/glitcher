# ASR (Attack Success Rate) Implementation Guide

## Overview

The Glitcher now uses **Attack Success Rate (ASR)** to determine whether a token exhibits glitch behavior, replacing the previous strict "all-or-nothing" validation approach. This provides much more nuanced and configurable glitch detection.

## What is ASR?

**Attack Success Rate (ASR)** measures the percentage of validation attempts where a token exhibits glitch behavior:

```
ASR = (Number of attempts showing glitch behavior) / (Total attempts)
```

### Examples:
- Token tested 5 times, exhibits glitch in 3 attempts: **ASR = 60%**
- Token tested 3 times, exhibits glitch in 3 attempts: **ASR = 100%**
- Token tested 10 times, exhibits glitch in 2 attempts: **ASR = 20%**

## Key Changes from Previous Implementation

### Before (Strict Validation)
```python
# Token was only considered a glitch if ALL attempts showed glitch behavior
is_glitch = glitch_attempts == num_attempts
```

**Example**: Token with 2/3 glitch attempts → **NOT a glitch** (strict policy)

### After (ASR-Based Validation)
```python
# Token is considered a glitch if ASR meets or exceeds threshold
asr = glitch_attempts / num_attempts
is_glitch = asr >= asr_threshold
```

**Example**: Token with 2/3 glitch attempts (ASR = 66.7%)
- With `--asr-threshold 0.5`: **IS a glitch** ✓
- With `--asr-threshold 0.8`: **NOT a glitch** ✗

## ASR Threshold Guidelines

### Threshold Values and Use Cases

| ASR Threshold | Use Case | Description |
|---------------|----------|-------------|
| **1.0 (100%)** | Research/Academic | Only tokens that ALWAYS exhibit glitch behavior |
| **0.8 (80%)** | High Confidence | Tokens that are very consistently problematic |
| **0.5 (50%)** | Balanced (Default) | Tokens that exhibit glitch behavior more often than not |
| **0.3 (30%)** | Broad Detection | Tokens that occasionally exhibit glitch behavior |
| **0.0 (0%)** | Maximum Coverage | Any token that shows glitch behavior at least once |

### Recommended Settings by Model Type

#### Deterministic Models
```bash
# Use higher thresholds since results should be consistent
--asr-threshold 0.8 --num-attempts 3
```

#### Non-Deterministic Models
```bash
# Use lower thresholds with more attempts
--asr-threshold 0.5 --num-attempts 5
```

#### Research/Production Use
```bash
# High confidence detection
--asr-threshold 0.8 --num-attempts 5
```

## CLI Usage Examples

### Mining with ASR

```bash
# Default ASR threshold (50%)
glitcher mine meta-llama/Llama-3.2-1B-Instruct --num-iterations 50

# High-confidence detection (80% threshold)
glitcher mine meta-llama/Llama-3.2-1B-Instruct \
  --num-iterations 50 \
  --asr-threshold 0.8 \
  --num-attempts 5

# Broad detection (30% threshold)
glitcher mine meta-llama/Llama-3.2-1B-Instruct \
  --num-iterations 50 \
  --asr-threshold 0.3 \
  --num-attempts 10

# Strict detection (100% threshold)
glitcher mine meta-llama/Llama-3.2-1B-Instruct \
  --num-iterations 50 \
  --asr-threshold 1.0 \
  --num-attempts 3
```

### Testing Tokens with ASR

```bash
# Test specific tokens with custom ASR threshold
glitcher test meta-llama/Llama-3.2-1B-Instruct \
  --token-ids 89472,127438,85069 \
  --enhanced \
  --asr-threshold 0.7 \
  --num-attempts 5

# Compare different validation methods
glitcher compare meta-llama/Llama-3.2-1B-Instruct \
  --token-ids 89472,127438,85069 \
  --asr-threshold 0.8 \
  --num-attempts 5
```

## ASR Demonstration

Use the demonstration script to understand how ASR thresholds affect classification:

```bash
# Basic ASR demonstration
python demo_asr_thresholds.py meta-llama/Llama-3.2-1B-Instruct

# Test specific tokens with multiple attempts
python demo_asr_thresholds.py meta-llama/Llama-3.2-1B-Instruct \
  --token-ids 89472,127438,85069 \
  --num-attempts 10 \
  --max-tokens 100
```

## Real-World ASR Scenarios

### Scenario 1: Highly Consistent Glitch Token
```
Token: "SpecialToken" (ID: 12345)
Attempts: 5
Results: [GLITCH, GLITCH, GLITCH, GLITCH, GLITCH]
ASR: 100%

Classifications:
- threshold 0.3: GLITCH ✓
- threshold 0.5: GLITCH ✓
- threshold 0.8: GLITCH ✓
- threshold 1.0: GLITCH ✓
```

### Scenario 2: Intermittent Glitch Token
```
Token: "SometimesWeird" (ID: 67890)
Attempts: 5
Results: [GLITCH, normal, GLITCH, GLITCH, normal]
ASR: 60%

Classifications:
- threshold 0.3: GLITCH ✓
- threshold 0.5: GLITCH ✓
- threshold 0.8: normal ✗
- threshold 1.0: normal ✗
```

### Scenario 3: Rarely Problematic Token
```
Token: "RarelyBad" (ID: 54321)
Attempts: 10
Results: [GLITCH, normal, normal, normal, normal, normal, normal, GLITCH, normal, normal]
ASR: 20%

Classifications:
- threshold 0.3: normal ✗
- threshold 0.5: normal ✗
- threshold 0.8: normal ✗
- threshold 1.0: normal ✗
```

## Implementation Details

### Enhanced Validation Function Signature
```python
def enhanced_glitch_verify(
    model, tokenizer, token_id, 
    chat_template=None, log_file=None, 
    max_tokens=100, quiet=True, 
    num_attempts=1, asr_threshold=0.5
):
```

### Key Parameters
- `num_attempts`: Number of validation attempts (default: 1)
- `asr_threshold`: ASR threshold for glitch classification (default: 0.5)

### Mining Function Integration
```python
def mine_glitch_tokens(
    model, tokenizer, num_iterations=50, batch_size=8, k=32,
    verbose=True, language="ENG", checkpoint_callback=None,
    log_file="glitch_mining_log.jsonl",
    enhanced_validation=True, max_tokens=50, 
    num_attempts=1, asr_threshold=0.5
):
```

## Logging and Output

### ASR Data in Logs
Enhanced validation logs now include ASR information:

```json
{
  "event": "enhanced_token_verification",
  "token": "ExampleToken",
  "token_id": 12345,
  "num_attempts": 5,
  "glitch_attempts": 3,
  "asr": 0.6,
  "asr_threshold": 0.5,
  "is_glitch": true,
  "final_decision_reason": "ASR 60.00% >= threshold 50.00%"
}
```

### Mining Validation Logs
```json
{
  "event": "mining_validation",
  "iteration": 10,
  "token": "ExampleToken",
  "token_id": 12345,
  "validation_method": "enhanced",
  "validation_result": true,
  "max_tokens": 50,
  "num_attempts": 3,
  "asr_threshold": 0.8
}
```

## Benefits of ASR Implementation

### 1. Configurable Sensitivity
- **High thresholds (0.8-1.0)**: Focus on consistently problematic tokens
- **Medium thresholds (0.5-0.7)**: Balanced detection
- **Low thresholds (0.2-0.4)**: Broad screening for potential issues

### 2. Better Handling of Non-Deterministic Models
- Accounts for variability in model responses
- Multiple attempts provide statistical confidence
- ASR gives insight into token reliability

### 3. Research and Production Flexibility
- **Research**: Use high thresholds for reproducible results
- **Production screening**: Use medium thresholds for comprehensive detection
- **Exploratory analysis**: Use low thresholds to find edge cases

### 4. Transparent Decision Making
- Clear ASR percentage shows token behavior patterns
- Logs explain exactly why tokens are classified as glitches
- Threshold tuning based on empirical data

## Migration from Previous Version

### Automatic Migration
- **Default behavior**: ASR threshold of 0.5 with 1 attempt mimics balanced detection
- **Existing scripts**: Will work unchanged with new default parameters
- **Legacy strict mode**: Use `--asr-threshold 1.0` to require 100% consistency

### Parameter Mapping
```bash
# Old strict validation (all attempts must pass)
# Equivalent new command:
glitcher mine model --asr-threshold 1.0 --num-attempts 3

# More permissive than old strict validation:
glitcher mine model --asr-threshold 0.5 --num-attempts 3
```

## Performance Considerations

### Speed vs. Accuracy Trade-offs
- **More attempts**: Higher confidence but slower execution
- **Higher thresholds**: Fewer false positives but might miss edge cases
- **Lower thresholds**: More comprehensive detection but potential false positives

### Recommended Configurations

#### Fast Screening
```bash
--asr-threshold 0.8 --num-attempts 1 --validation-tokens 25
```

#### Balanced Detection
```bash
--asr-threshold 0.5 --num-attempts 3 --validation-tokens 50
```

#### Thorough Analysis
```bash
--asr-threshold 0.3 --num-attempts 5 --validation-tokens 100
```

## Troubleshooting

### Common Issues

#### Too Many False Positives
```bash
# Increase ASR threshold
--asr-threshold 0.8
```

#### Missing Known Glitch Tokens
```bash
# Decrease ASR threshold and increase attempts
--asr-threshold 0.3 --num-attempts 5
```

#### Inconsistent Results
```bash
# Increase number of attempts for better statistics
--num-attempts 10
```

### Debug Commands
```bash
# Test specific problematic token with detailed logging
glitcher test model --token-ids 12345 --enhanced --num-attempts 10 --asr-threshold 0.1

# Compare ASR behavior across different thresholds
python demo_asr_thresholds.py model --token-ids 12345 --num-attempts 10
```

## Future Enhancements

### Planned Features
- [ ] Adaptive ASR thresholds based on model behavior
- [ ] ASR-based token ranking and scoring
- [ ] Statistical confidence intervals for ASR measurements
- [ ] ASR trend analysis across model iterations
- [ ] Automated ASR threshold recommendation based on token distribution

### Advanced ASR Analytics
- [ ] ASR distribution visualization
- [ ] Correlation analysis between ASR and other token properties
- [ ] ASR-based clustering of similar tokens
- [ ] Time-series ASR tracking for model monitoring