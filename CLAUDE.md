# Glitcher Development Guide

## Installation and Setup
```bash
# Install package in development mode
pip install -e .
pip install accelerate  # Required for loading models with device_map
pip install matplotlib   # Required for GUI animation (--gui flag)
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

# Range-based mining - Custom token ID range
glitcher mine meta-llama/Llama-3.2-1B-Instruct --mode range --range-start 0 --range-end 1000 --sample-rate 0.1

# Range-based mining - Unicode ranges (systematic exploration)
glitcher mine meta-llama/Llama-3.2-1B-Instruct --mode unicode --sample-rate 0.05 --max-tokens-per-range 50

# Range-based mining - Special token ranges
glitcher mine meta-llama/Llama-3.2-1B-Instruct --mode special --sample-rate 0.2 --max-tokens-per-range 100

# Range mining with enhanced validation
glitcher mine meta-llama/Llama-3.2-1B-Instruct --mode range --range-start 128000 --range-end 128256 --sample-rate 1.0 --num-attempts 3

# High-confidence range mining with strict ASR threshold
glitcher mine meta-llama/Llama-3.2-1B-Instruct --mode unicode --sample-rate 0.1 --asr-threshold 0.8 --num-attempts 5

# Genetic algorithm for breeding glitch token combinations
glitcher genetic meta-llama/Llama-3.2-1B-Instruct --base-text "The quick brown" --generations 50 --population-size 30

# Genetic algorithm with custom parameters
glitcher genetic meta-llama/Llama-3.2-1B-Instruct --base-text "Hello world" --generations 100 --population-size 50 --mutation-rate 0.15 --crossover-rate 0.8

# Genetic algorithm with larger token combinations
glitcher genetic meta-llama/Llama-3.2-1B-Instruct --base-text "The quick brown" --max-tokens 5 --generations 75 --elite-size 10

# Genetic algorithm targeting specific token
glitcher genetic meta-llama/Llama-3.2-1B-Instruct --base-text "The quick brown" --target-token "fox" --generations 50

# Genetic batch experiments across multiple scenarios
glitcher genetic meta-llama/Llama-3.2-1B-Instruct --batch --token-file glitch_tokens.json --generations 30 --population-size 25

# Genetic algorithm with custom token file and output
glitcher genetic meta-llama/Llama-3.2-1B-Instruct --token-file custom_glitch_tokens.json --output genetic_results_custom.json --generations 60

# Genetic algorithm with real-time GUI animation
glitcher genetic meta-llama/Llama-3.2-1B-Instruct --gui --base-text "The quick brown" --generations 50

# GUI with custom parameters and batch mode
glitcher genetic meta-llama/Llama-3.2-1B-Instruct --gui --batch --generations 30 --population-size 25

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

# Legacy standalone range mining (still available)
python range_mining.py meta-llama/Llama-3.2-1B-Instruct --range-start 0 --range-end 1000 --sample-rate 0.1
python range_mining.py meta-llama/Llama-3.2-1B-Instruct --unicode-ranges --sample-rate 0.05
python range_mining.py meta-llama/Llama-3.2-1B-Instruct --special-ranges --sample-rate 0.2

# Legacy standalone genetic algorithm (now integrated into main CLI as 'glitcher genetic')
python examples/genetic_probability_reducer.py meta-llama/Llama-3.2-1B-Instruct --base-text "The quick brown" --generations 50
python examples/genetic_batch_runner.py meta-llama/Llama-3.2-1B-Instruct --token-file glitch_tokens.json

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

## Range Mining Modes

### Mode: `entropy` (Default)
Standard entropy-based glitch token mining using the original algorithm.

### Mode: `range`
Systematic exploration of specific token ID ranges. Requires `--range-start` and `--range-end`.

### Mode: `unicode`
Tests tokens across Unicode character ranges (ASCII, Latin, Cyrillic, CJK, etc.).

### Mode: `special`
Tests tokens in vocabulary ranges likely to contain special tokens and artifacts.

### Range Mining Parameters
- `--sample-rate`: Fraction of tokens to test (0.0-1.0, default: 0.1)
- `--max-tokens-per-range`: Maximum tokens per range (default: 100)
- `--range-start`: Starting token ID (range mode only)
- `--range-end`: Ending token ID (range mode only)

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

### Range Mining Output
Range mining generates detailed logs and results:

```
✓ Found glitch token: 'SomeToken' (ID: 12345) in Unicode Block: Latin Extended-A
✗ Testing range: ASCII Control (0-127) - 5/127 tokens are glitches (3.9%)
```

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

## Genetic Algorithm Parameters

### Basic Parameters
- `--base-text`: Base text to test probability reduction on (default: "The quick brown")
- `--target-token`: Specific token to target (auto-detected if not provided)
- `--token-file`: JSON file containing glitch tokens (default: glitch_tokens.json)
- `--population-size`: Population size for genetic algorithm (default: 50)
- `--generations`: Maximum number of generations (default: 100)
- `--max-tokens`: Maximum tokens per individual combination (default: 3)

### Advanced Parameters
- `--mutation-rate`: Mutation rate (0.0-1.0, default: 0.1)
- `--crossover-rate`: Crossover rate (0.0-1.0, default: 0.7)
- `--elite-size`: Elite size for genetic algorithm (default: 5)
- `--batch`: Run batch experiments across multiple scenarios
- `--output`: Output file for results (default: genetic_results.json)
- `--gui`: Show real-time GUI animation of genetic algorithm evolution

### GUI Animation Parameters
- `--gui`: Enable real-time visualization of genetic algorithm evolution
- Requires matplotlib: `pip install matplotlib`
- Shows live fitness evolution, token combinations, and reduction statistics
- Interactive window that updates in real-time during evolution

### Genetic Algorithm Output Format
When using genetic algorithm, results include evolved token combinations:

```
🏆 Top Results:
1. Tokens: [12345, 67890] (['Token1', 'Token2'])
   Fitness: 0.856432
   Probability reduction: 85.64%

2. Tokens: [11111, 22222, 33333] (['TokenA', 'TokenB', 'TokenC'])
   Fitness: 0.823451
   Probability reduction: 82.35%
```

### Recommended Genetic Algorithm Settings
- **Quick Exploration**: `--generations 30 --population-size 20` (faster results)
- **Thorough Search**: `--generations 100 --population-size 50` (better results)
- **Large Combinations**: `--max-tokens 5 --generations 75` (complex token combinations)
- **Batch Analysis**: `--batch --generations 50` (multiple scenario testing)
- **Visual Monitoring**: `--gui --generations 50` (real-time progress visualization)

## GUI Animation Features

### Real-time Visualization
When using `--gui` flag, the genetic algorithm displays a live animation window showing:

- **Fitness Evolution Chart**: Live graph of best and average fitness over generations
- **Current Statistics Panel**: Real-time metrics including:
  - Current generation number
  - Best fitness score
  - Average population fitness
  - Probability reduction percentage
  - Evolution progress
- **Token Combination Display**: Current best token combination with:
  - Token IDs
  - Decoded token texts (when available)
  - Fitness score of current best individual
- **Base Information**: Target token and baseline probability

### GUI Usage Examples

```bash
# Basic GUI monitoring
glitcher genetic meta-llama/Llama-3.2-1B-Instruct --gui

# GUI with custom evolution parameters
glitcher genetic meta-llama/Llama-3.2-1B-Instruct --gui --generations 100 --population-size 50

# GUI batch experiments
glitcher genetic meta-llama/Llama-3.2-1B-Instruct --gui --batch --token-file glitch_tokens.json

# GUI with specific target
glitcher genetic meta-llama/Llama-3.2-1B-Instruct --gui --target-token "specific_target" --generations 75
```

### GUI Installation Requirements
```bash
# Required for GUI functionality
pip install matplotlib

# Optional: Better GUI backend support
pip install tkinter  # Usually pre-installed with Python
```

### GUI Features
- **Real-time Updates**: Animation updates every generation
- **Interactive Window**: Resizable, zoomable plots
- **Auto-scaling**: Axes automatically adjust to data range
- **Color-coded Status**: Visual indicators for performance levels
- **Stay-alive Mode**: Window remains open after evolution completes
- **Close-to-exit**: Simply close the window to terminate visualization

### GUI Performance Notes
- GUI adds minimal overhead to genetic algorithm execution
- Animation updates happen asynchronously
- Window can be closed at any time without stopping evolution
- Works with both single and batch experiment modes

## Code Style Guidelines
- Follow PEP 8 conventions for Python code
- Import order: standard library, third-party packages, local modules
- Type hints required for function parameters and return values
- Naming: snake_case for functions/variables, PascalCase for classes
- Include bilingual docstrings (English and Chinese)
- Wrap code at 100 characters per line
- Use explicit error handling with informative messages
- Maintain clean module structure with clear separation of concerns