# Glitcher Development Guide

## Installation and Setup
```bash
# Install package in development mode
pip install -e .
pip install accelerate  # Required for loading models with device_map
```

## Common Commands
```bash
# Run glitch token mining (enhanced validation enabled by default)
glitcher mine meta-llama/Llama-3.2-1B-Instruct --num-iterations 50 --batch-size 8 --k 32

# Enhanced mining with custom validation parameters
glitcher mine meta-llama/Llama-3.2-1B-Instruct --num-iterations 50 --validation-tokens 100 --num-attempts 3

# Enhanced mining with multiple attempts for non-deterministic models
glitcher mine meta-llama/Llama-3.2-1B-Instruct --num-iterations 30 --num-attempts 5 --validation-tokens 50

# Enhanced mining with custom ASR threshold (Attack Success Rate)
glitcher mine meta-llama/Llama-3.2-1B-Instruct --num-iterations 50 --asr-threshold 0.8 --num-attempts 3

# Strict ASR threshold for high-confidence glitch detection
glitcher mine meta-llama/Llama-3.2-1B-Instruct --num-iterations 50 --asr-threshold 1.0 --num-attempts 5

# Lenient ASR threshold for broader detection
glitcher mine meta-llama/Llama-3.2-1B-Instruct --num-iterations 50 --asr-threshold 0.3 --num-attempts 3

# Legacy mining (disable enhanced validation for speed)
glitcher mine meta-llama/Llama-3.2-1B-Instruct --num-iterations 50 --disable-enhanced-validation

# Test specific token IDs
glitcher test meta-llama/Llama-3.2-1B-Instruct --token-ids 89472,127438,85069

# Test with enhanced validation and multiple attempts for non-deterministic results
glitcher test meta-llama/Llama-3.2-1B-Instruct --token-ids 89472,127438,85069 --enhanced --num-attempts 3

# Test with enhanced validation and custom ASR threshold
glitcher test meta-llama/Llama-3.2-1B-Instruct --token-ids 89472,127438,85069 --enhanced --asr-threshold 0.7 --num-attempts 5

# Compare standard vs enhanced validation methods
glitcher compare meta-llama/Llama-3.2-1B-Instruct --token-ids 89472,127438,85069 --num-attempts 5

# Compare with custom ASR threshold
glitcher compare meta-llama/Llama-3.2-1B-Instruct --token-ids 89472,127438,85069 --num-attempts 5 --asr-threshold 0.8

# Chat test with a specific token
glitcher chat meta-llama/Llama-3.2-1B-Instruct 89472 --max-size 20

# Run validation tests
glitcher validate meta-llama/Llama-3.2-1B-Instruct --output-dir validation_results

# Test known glitch tokens
python test_known_glitches.py meta-llama/Llama-3.2-1B-Instruct --token-file glitch_tokens.json

# Run enhanced validation demo with multiple attempts
python test_enhanced_validation.py meta-llama/Llama-3.2-1B-Instruct --num-attempts 3 --max-tokens 100

# Compare validation methods with multiple attempts
python test_enhanced_validation.py meta-llama/Llama-3.2-1B-Instruct --compare --num-attempts 5

# Run token repetition test
python token_repetition_test.py meta-llama/Llama-3.2-1B-Instruct

# Test domain extraction from log files with known glitch token
glitcher domain meta-llama/Llama-3.2-1B-Instruct --test-cpptypes

# Test domain extraction with specific token IDs
glitcher domain meta-llama/Llama-3.2-1B-Instruct --token-ids 89472,127438,85069

# Test domain extraction with tokens from file
glitcher domain meta-llama/Llama-3.2-1B-Instruct --token-file glitch_tokens.json

# Run domain extraction test script directly
python test_domain_extraction.py meta-llama/Llama-3.2-1B-Instruct --test-cpptypes

# Test domain extraction with multiple tokens and control group
python test_domain_extraction.py meta-llama/Llama-3.2-1B-Instruct --token-ids 89472,127438,85069 --normal-count 10

# Test enhanced mining functionality
python test_enhanced_mining.py meta-llama/Llama-3.2-1B-Instruct --iterations 10

# Compare enhanced vs standard mining methods
python test_enhanced_mining.py meta-llama/Llama-3.2-1B-Instruct --compare --iterations 10

# Test mining with different ASR thresholds
python test_enhanced_mining.py meta-llama/Llama-3.2-1B-Instruct --iterations 10 --asr-threshold 0.8

# Demonstrate ASR threshold impact on token classification
python demo_asr_thresholds.py meta-llama/Llama-3.2-1B-Instruct --num-attempts 5

# ASR demo with specific tokens and custom parameters
python demo_asr_thresholds.py meta-llama/Llama-3.2-1B-Instruct --token-ids 89472,127438,85069 --num-attempts 10 --max-tokens 100
```

## ASR (Attack Success Rate) Guidelines

### Understanding ASR Thresholds
- **ASR = 1.0 (100%)**: Token exhibits glitch behavior in ALL attempts (strictest)
- **ASR = 0.8 (80%)**: Token exhibits glitch behavior in 80%+ of attempts (high confidence)
- **ASR = 0.5 (50%)**: Token exhibits glitch behavior in 50%+ of attempts (balanced, default)
- **ASR = 0.3 (30%)**: Token exhibits glitch behavior in 30%+ of attempts (lenient)
- **ASR = 0.0 (0%)**: Any glitch behavior detected (most permissive)

### Recommended ASR Thresholds
- **Research/Academic**: Use 0.8-1.0 for high confidence results
- **General Discovery**: Use 0.5 (default) for balanced detection
- **Broad Screening**: Use 0.3-0.4 for comprehensive token analysis
- **Non-deterministic Models**: Use lower thresholds (0.3-0.5) with more attempts (5-10)

### Example: Token with 3/5 attempts showing glitch behavior (ASR = 0.6)
- `--asr-threshold 0.5`: Token classified as GLITCH ✓
- `--asr-threshold 0.7`: Token classified as normal ✗
- `--asr-threshold 1.0`: Token classified as normal ✗

## Mining Output with ASR

### Enhanced Validation Output Format
When using enhanced validation (default), the mining process displays ASR information:

```
✓ Validated glitch token: 'SomeToken' (ID: 12345, asr: 75.00%, entropy: 0.1234, target_prob: 0.001234, top_prob: 0.567890, method: enhanced)
✗ False positive: 'AnotherToken' (ID: 67890, asr: 33.33%, failed enhanced validation)
```

### Standard Validation Output Format
When using standard validation (`--disable-enhanced-validation`):

```
✓ Validated glitch token: 'SomeToken' (ID: 12345, entropy: 0.1234, target_prob: 0.001234, top_prob: 0.567890, method: standard)
✗ False positive: 'AnotherToken' (ID: 67890, failed standard validation)
```

### ASR Display Interpretation
- **asr: 100.00%**: Token exhibits glitch behavior in ALL validation attempts
- **asr: 75.00%**: Token exhibits glitch behavior in 75% of attempts (e.g., 3/4)
- **asr: 66.67%**: Token exhibits glitch behavior in 67% of attempts (e.g., 2/3)
- **asr: 50.00%**: Token exhibits glitch behavior in 50% of attempts (e.g., 1/2)
- **asr: 33.33%**: Token exhibits glitch behavior in 33% of attempts (e.g., 1/3)

## Code Style Guidelines
- Follow PEP 8 conventions for Python code
- Import order: standard library, third-party packages, local modules
- Type hints required for function parameters and return values
- Naming: snake_case for functions/variables, PascalCase for classes
- Include bilingual docstrings (English and Chinese)
- Wrap code at 100 characters per line
- Use explicit error handling with informative messages
- Maintain clean module structure with clear separation of concerns