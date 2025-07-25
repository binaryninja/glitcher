# Glitcher Validation System Guide

## Overview

The Glitcher validation system is designed to accurately identify glitch tokens while minimizing false positives. This guide explains how the validation mechanisms work, their differences, and how to configure them for optimal results.

## What Are Glitch Tokens?

Glitch tokens are vocabulary items in language models that cause unexpected behavior:
- **Incoherent Output**: Generate nonsensical or random text
- **Pattern Breaking**: Disrupt normal reasoning and instruction following
- **Repetitive Behavior**: Cause loops or repetitive generation
- **Context Ignoring**: Make the model ignore previous context

## Validation Methods

### 1. Standard Validation (`strictly_glitch_verify`)

**How it works:**
- Tests immediate next-token prediction probability
- Uses 3 different prompt formats for robustness
- Applies very strict probability thresholds
- Includes content-based filtering for known false positives

**Criteria for classification as glitch:**
```python
# Token is considered a glitch if ALL conditions are met:
1. Probability < threshold (0.00001 for Llama 3.2, 0.00005 for others)
2. Not the top predicted token
3. Doesn't match false positive patterns
4. Fails ALL 3 test formats
```

**Advantages:**
- Fast execution
- Low computational overhead
- Good for batch processing

**Limitations:**
- Single-shot testing may miss non-deterministic behavior
- Relies only on immediate next-token probability

### 2. Enhanced Validation (`enhanced_glitch_verify`)

**How it works:**
- Generates full text sequences (up to `max_tokens`)
- Searches for target token within generated content
- Runs multiple attempts to handle non-deterministic behavior
- Uses Attack Success Rate (ASR) for final classification

**Process:**
```python
for attempt in range(num_attempts):
    for test_format in [format1, format2, format3]:
        generated_text = model.generate(test_format, max_tokens=max_tokens)
        if target_token in generated_text:
            # Token behaves normally in this test
            test_passes += 1
        else:
            # Token exhibits glitch behavior
            test_fails += 1
    
    if all_tests_failed_this_attempt:
        glitch_attempts += 1

asr = glitch_attempts / num_attempts
is_glitch = asr >= asr_threshold
```

**Advantages:**
- More comprehensive testing
- Handles non-deterministic model behavior
- Provides confidence measure (ASR)
- Better detection of subtle glitches

**Trade-offs:**
- Slower execution
- Higher computational cost
- More complex configuration

## Attack Success Rate (ASR) System

### Definition
ASR = (Number of attempts showing glitch behavior) / (Total attempts) × 100%

### Interpretation
- **100% ASR**: Token ALWAYS exhibits glitch behavior (highest confidence)
- **80% ASR**: Token exhibits glitch behavior in 4/5 attempts (high confidence)
- **60% ASR**: Token exhibits glitch behavior in 3/5 attempts (moderate confidence)
- **40% ASR**: Token exhibits glitch behavior in 2/5 attempts (low confidence)
- **0% ASR**: Token NEVER exhibits glitch behavior (normal token)

### Threshold Guidelines

| Use Case | Recommended ASR Threshold | Description |
|----------|---------------------------|-------------|
| Research/Academic | 0.8 - 1.0 | High confidence, low false positives |
| General Discovery | 0.5 (default) | Balanced sensitivity and specificity |
| Broad Screening | 0.3 - 0.4 | Comprehensive detection, more false positives |
| Non-deterministic Models | 0.3 - 0.5 | Account for model variability |

### Example Scenarios

**Token tested 5 times, glitch behavior in 3 attempts (ASR = 60%):**
- `--asr-threshold 0.5`: Token classified as **GLITCH** ✓
- `--asr-threshold 0.7`: Token classified as **NORMAL** ✗
- `--asr-threshold 1.0`: Token classified as **NORMAL** ✗

## False Positive Detection

### Content-Based Filtering

Automatically filters tokens with known false positive patterns:

```python
false_positive_patterns = [
    '[', ']', '(', ')',  # Bracket tokens (formatting)
    '_',                 # Underscore (programming)
    'arg', 'prop',       # Common prefixes
    'char',              # Character references
    '$', '.'             # Special symbols
]
```

**Examples of filtered tokens:**
- `[INST]` - Instruction formatting
- `args` - Programming parameter
- `$variable` - Variable reference
- `.method` - Method calls

### Probability Thresholds

Ultra-strict thresholds for immediate next-token probability:
- **Llama 3.2 models**: 0.00001 (0.001%)
- **Other models**: 0.00005 (0.005%)

Tokens with higher probability are likely normal, even if they have low entropy.

### Multi-Format Testing

Three different prompt formats ensure robust validation:
1. **Direct completion**: Simple text continuation
2. **Conversational format**: Chat-style interaction
3. **Instruction format**: Task-oriented prompts

Token must fail ALL formats to be considered a glitch.

## Configuration Parameters

### Mining Configuration

```bash
# Enhanced validation (default)
glitcher mine model_path --enhanced-validation

# Configure ASR parameters
glitcher mine model_path --asr-threshold 0.7 --num-attempts 5 --validation-tokens 100

# Disable enhanced validation (use standard)
glitcher mine model_path --disable-enhanced-validation
```

### Testing Configuration

```bash
# Enhanced testing with ASR
glitcher test model_path --token-ids 1000,2000 --enhanced --asr-threshold 0.8

# Multiple attempts for non-deterministic models
glitcher test model_path --token-ids 1000,2000 --num-attempts 10 --asr-threshold 0.5
```

### Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--asr-threshold` | 0.5 | ASR threshold for glitch classification |
| `--num-attempts` | 1 | Number of validation attempts per token |
| `--validation-tokens` | 50 | Max tokens to generate during validation |
| `--enhanced-validation` | True | Use enhanced validation method |
| `--disable-enhanced-validation` | False | Force standard validation |

## Best Practices

### For Research and Academic Work
```bash
# High-confidence settings
glitcher mine model_path \
    --asr-threshold 0.8 \
    --num-attempts 5 \
    --validation-tokens 100
```

### For General Discovery
```bash
# Balanced settings (default)
glitcher mine model_path \
    --asr-threshold 0.5 \
    --num-attempts 3 \
    --validation-tokens 50
```

### For Non-deterministic Models
```bash
# More attempts, lower threshold
glitcher mine model_path \
    --asr-threshold 0.3 \
    --num-attempts 10 \
    --validation-tokens 75
```

### For Memory-Constrained Environments
```bash
# Standard validation only
glitcher mine model_path \
    --disable-enhanced-validation \
    --batch-size 4
```

## Output Interpretation

### Mining Output

```bash
✓ Validated glitch token: 'SomeToken' (ID: 12345, asr: 80.00%, entropy: 0.1234, target_prob: 0.001234, top_prob: 0.567890, method: enhanced)
✗ False positive: 'NormalToken' (ID: 67890, asr: 20.00%, failed enhanced validation)
```

**Key information:**
- **ASR**: Attack Success Rate percentage
- **method**: Validation method used (enhanced/standard)
- **entropy**: Original entropy score from mining
- **target_prob**: Probability of target token in next position
- **top_prob**: Probability of most likely next token

### Test Output

```bash
[1/3] Token: 'example', ID: 1234, Is glitch: True, ASR: 66.67%
[2/3] Token: 'normal', ID: 5678, Is glitch: False, ASR: 16.67%
```

## Troubleshooting

### High False Positive Rate
- Increase ASR threshold (`--asr-threshold 0.7` or higher)
- Increase number of attempts (`--num-attempts 5` or more)
- Check for model-specific patterns that should be filtered

### Missing True Glitches
- Decrease ASR threshold (`--asr-threshold 0.3` or lower)
- Increase validation tokens (`--validation-tokens 100` or more)
- Use enhanced validation if using standard

### Performance Issues
- Use standard validation for speed (`--disable-enhanced-validation`)
- Reduce number of attempts (`--num-attempts 1`)
- Reduce validation tokens (`--validation-tokens 20`)

## Technical Details

### Validation Logic Flow

```
Input: Token ID
├─ Content Filtering
│  ├─ Match false positive patterns? → NOT GLITCH
│  └─ No match → Continue
├─ Standard Validation
│  ├─ For each test format:
│  │  ├─ Calculate next-token probability
│  │  ├─ Check if above threshold → Test PASSES
│  │  └─ Check if top prediction → Test PASSES
│  └─ All tests fail? → GLITCH (standard)
└─ Enhanced Validation
   ├─ For each attempt:
   │  ├─ For each test format:
   │  │  ├─ Generate text sequence
   │  │  ├─ Search for target token
   │  │  └─ Token found? → Test PASSES
   │  └─ All tests fail? → Glitch attempt
   ├─ Calculate ASR = glitch_attempts / total_attempts
   └─ ASR ≥ threshold? → GLITCH (enhanced)
```

### Model-Specific Behavior

**Llama 3.2 Models:**
- Use stricter probability threshold (0.00001)
- Require ALL test formats to fail
- Special handling for instruction tokens

**Other Models:**
- Standard probability threshold (0.00005)
- Single test format failure can indicate normal token
- Standard false positive filtering

## Examples

### Command Examples

```bash
# Basic enhanced validation
glitcher mine meta-llama/Llama-3.2-1B-Instruct --num-iterations 50

# High-confidence research settings
glitcher mine meta-llama/Llama-3.2-1B-Instruct \
    --num-iterations 100 \
    --asr-threshold 0.8 \
    --num-attempts 5 \
    --validation-tokens 100

# Fast screening with standard validation
glitcher mine meta-llama/Llama-3.2-1B-Instruct \
    --num-iterations 200 \
    --disable-enhanced-validation \
    --batch-size 16

# Test specific tokens with enhanced validation
glitcher test meta-llama/Llama-3.2-1B-Instruct \
    --token-ids 89472,127438,85069 \
    --enhanced \
    --asr-threshold 0.7 \
    --num-attempts 5
```

This validation system provides robust, configurable glitch token detection while minimizing false positives through multiple complementary mechanisms.