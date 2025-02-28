# Glitcher

A command-line tool for mining, testing, and classifying glitch tokens in large language models.

## What are Glitch Tokens?

Glitch tokens are special tokens that language models struggle to repeat or process correctly. They can cause unexpected behaviors like:
- Inability to repeat the token when explicitly asked
- Producing nonsensical outputs or hallucinations
- Triggering repetitive loops or verbose outputs
- Disrupting the model's reasoning capabilities
- Potentially bypassing safety mechanisms

Finding and understanding glitch tokens helps improve model robustness and security.

## Installation

```bash
pip install git+https://github.com/binaryninja/glitcher.git
```

Or clone the repository and install locally:

```bash
git clone https://github.com/binaryninja/glitcher.git
cd glitcher
pip install -e .
pip install accelerate  # Required for loading models with device_map
```

## Comprehensive Guide to Finding and Evaluating Glitch Tokens

Glitcher provides a complete workflow for discovering, testing, and classifying glitch tokens:

1. **Find potential glitch tokens** using the mining tool
2. **Verify discovered tokens** with validation tests
3. **Analyze token behavior** in chat contexts
4. **Classify tokens** by their effects and behaviors
5. **Document findings** for further research

### Step 1: Mining Glitch Tokens

Find glitch tokens in a model using the gradient-guided search method:

```bash
# Quick scan (for initial testing)
glitcher mine meta-llama/Llama-3.2-1B-Instruct --num-iterations 50 --batch-size 4 --k 16

# More thorough scan
glitcher mine meta-llama/Llama-3.2-1B-Instruct --num-iterations 200 --batch-size 8 --k 32
```

For long-running scans, use the resume functionality:

```bash
# Initial scan with regular checkpoints
glitcher mine meta-llama/Llama-3.2-1B-Instruct --num-iterations 300 --save-interval 10 --progress-file llama_scan.json

# Resume after interruption
glitcher mine meta-llama/Llama-3.2-1B-Instruct --resume --progress-file llama_scan.json
```

### Step 2: Validating Discovered Tokens

After mining, verify which tokens are actually glitch tokens:

```bash
# Test tokens from mining results
glitcher test meta-llama/Llama-3.2-1B-Instruct --token-file glitch_tokens.json

# Test specific token IDs
glitcher test meta-llama/Llama-3.2-1B-Instruct --token-ids 89472,127438,107658
```

Run validation tests to understand model behavior with these tokens:

```bash
# Run all validation tests
glitcher validate meta-llama/Llama-3.2-1B-Instruct --output-dir validation_results

# Test token repetition specifically
python token_repetition_test.py meta-llama/Llama-3.2-1B-Instruct

# Test against known glitch tokens
python test_known_glitches.py meta-llama/Llama-3.2-1B-Instruct --token-file glitch_tokens.json
```

### Step 3: Analyzing Token Behavior

Analyze how the model responds to specific glitch tokens in conversation:

```bash
# Test a specific token with default parameters
glitcher chat meta-llama/Llama-3.2-1B-Instruct 89472

# Generate longer responses to observe full effects
glitcher chat meta-llama/Llama-3.2-1B-Instruct 127438 --max-size 50
```

Look for these typical behaviors in the output:
- The model fails to repeat the token accurately
- The model produces unexpected or nonsensical output
- The response is significantly different from what you'd expect with normal tokens

### Step 4: Classifying Glitch Tokens

Classify tokens by their effects using the classification tool:

```bash
# Classify tokens from a file
glitch-classify meta-llama/Llama-3.2-1B-Instruct --token-file validated_glitch_tokens.json

# Classify specific token IDs
glitch-classify meta-llama/Llama-3.2-1B-Instruct --token-ids 89472,127438,107658

# Use higher max-tokens for more accurate classification
glitch-classify meta-llama/Llama-3.2-1B-Instruct --token-file glitch_tokens.json --max-tokens 100
```

The classifier categorizes tokens into five main categories:

1. **Injection**: Tokens that enable bypassing instructions or filters
2. **IDOS (Infinite/Denial-of-Service)**: Tokens that trigger repetitive loops
3. **Hallucination**: Tokens causing nonsensical outputs
4. **Disruption**: Tokens that disrupt the model's internal reasoning
5. **Bypass**: Tokens that allow bypassing specific safety filters

### Step 5: Documenting Findings

Document your findings to enable further research:

1. Create a list of confirmed glitch tokens with their IDs and properties
2. Record which models are affected by each token
3. Note any patterns in token characteristics (e.g., character sets, encodings)
4. Document classification results and observed behaviors
5. Share findings with the research community when appropriate

This documentation helps track model vulnerabilities and contributes to improving model security.

## Conducting a Full Scan

To perform a comprehensive scan for glitch tokens, follow these guidelines:

### Scaling Considerations

1. **Memory Requirements**:
   - For larger models (>13B parameters), you'll need at least 16GB VRAM
   - Consider using quantization options like `--quant-type int8` or `--quant-type int4` for large models
   - For 70B+ models, use a multi-GPU setup or offload to CPU

2. **Time Requirements**:
   - A thorough scan might require 500-1000 iterations
   - Each iteration tests `batch_size` tokens (default: 8)
   - Expected duration: 1-4 hours depending on model size and hardware

3. **Efficient Parameter Settings**:
   ```bash
   # Large model scan (slower but thorough)
   glitcher mine meta-llama/Llama-3.2-70B --num-iterations 500 --batch-size 16 --k 64 --quant-type int4 --save-interval 10
   
   # Medium model scan (balanced)
   glitcher mine meta-llama/Llama-3.2-8B --num-iterations 250 --batch-size 8 --k 32
   
   # Quick scan (for initial testing)
   glitcher mine meta-llama/Llama-3.2-1B --num-iterations 50 --batch-size 4 --k 16
   ```

### Using Resume Effectively

For long-running scans, use the resume functionality:

1. **Start the initial scan with a reasonable save interval**:
   ```bash
   glitcher mine meta-llama/Llama-3.2-1B --num-iterations 300 --save-interval 10 --progress-file llama_scan.json
   ```

2. **If interrupted, resume the scan**:
   ```bash
   glitcher mine meta-llama/Llama-3.2-1B --resume --progress-file llama_scan.json
   ```

3. **Continue with more iterations if needed**:
   ```bash
   glitcher mine meta-llama/Llama-3.2-1B --resume --num-iterations 500 --progress-file llama_scan.json
   ```

### Scan Results Analysis

After completing a scan:

1. **Verify collected tokens**:
   ```bash
   glitcher test meta-llama/Llama-3.2-1B --token-file glitch_tokens.json
   ```

2. **Test individual tokens more deeply**:
   ```bash
   glitcher chat meta-llama/Llama-3.2-1B 89472 --max-size 50
   ```

### Limitations

- **Coverage**: Even long scans typically cover <1% of the model's vocabulary
- **False negatives**: Some glitch tokens may be missed due to the gradient-based search method
- **Model variability**: Different model versions may exhibit different glitch tokens
- **Quantization effects**: Using int4/int8 quantization may affect the detection accuracy

## New Classification Tool

The new `glitch-classify` command allows you to categorize glitch tokens by their effects:

```bash
# Classify glitch tokens from a file
glitch-classify meta-llama/Llama-3.2-1B --token-file validated_glitch_tokens.json

# Classify specific token IDs 
glitch-classify meta-llama/Llama-3.2-1B --token-ids 89472,127438

# Use a higher max-tokens value for better detection
glitch-classify meta-llama/Llama-3.2-1B --token-file tokens.json --max-tokens 100
```

### Classification of Glitch Tokens

The classification system helps researchers understand how different glitch tokens affect model behavior. Each token is systematically tested against multiple prompts to determine its effects.

### Classification Categories

Glitch tokens are classified into these categories based on specific behavioral tests:

1. **Injection**: Tokens that enable bypassing instructions or filters
   - Test: Checks if the token allows the model to ignore safety instructions
   - Example: The model produces harmful content when the token is inserted despite clear safety instructions
   
2. **IDOS (Infinite/Denial-of-Service)**: Tokens that trigger repetitive loops
   - Test: Checks if the token causes endless repetition or verbose output
   - Example: The model gets stuck in a loop repeating the same phrase or generates excessively long responses
   
3. **Hallucination**: Tokens causing nonsensical outputs
   - Test: Checks if the token makes the model produce incoherent responses
   - Example: The model generates random, nonsensical text that lacks logical coherence
   
4. **Disruption**: Tokens that disrupt the model's internal reasoning
   - Test: Checks if the token makes the model fail at simple tasks like math
   - Example: The model incorrectly answers that 2+2=5 when the token is included
   
5. **Bypass**: Tokens that allow bypassing specific safety filters
   - Test: Checks if the token makes the model ignore explicit instructions
   - Example: The model generates content it was specifically instructed not to produce

Each token is tested with multiple prompt templates, and can be classified into multiple categories if it exhibits several behaviors.

### Classification Output and Interpretation

The classifier produces a comprehensive JSON file with detailed test results and a summary table:

```
Classification Summary:
================================================================================
Token	Injection	IDOS	Hallucination	Disruption	Filter Bypass	Notes
--------------------------------------------------------------------------------
Token_XYZ	❌	✅	❌	❌	❌	IDOS
Token_ABC	✅	❌	✅	❌	✅	Injection, Hallucination, Bypass
================================================================================
```

How to interpret the results:
- ✅ indicates the token exhibits this behavior
- ❌ indicates the token does not exhibit this behavior
- The Notes column summarizes all observed behaviors

The detailed JSON output includes:
- Raw model responses for each test
- Confidence scores for each classification
- Specific patterns observed in the responses

## Understanding the Glitch Token Mining Algorithm

Glitcher uses a sophisticated gradient-guided search algorithm to efficiently find glitch tokens in large language models without having to test every token in the vocabulary (which would be prohibitively expensive).

### The Mining Process

The algorithm works through these key steps:

1. **Initialization**: Starts with a random token or one with low L2 norm
2. **Gradient Calculation**:
   - Constructs a prompt asking the model to repeat the current token
   - Calculates the entropy of the next token distribution
   - Computes the gradient of entropy with respect to the token embedding
   
3. **Token Exploration**:
   - Identifies nearby tokens in the embedding space using normalized L2 distance
   - Estimates the entropy change for each candidate token using the gradient
   - Selects a batch of tokens with the highest predicted entropy
   
4. **Glitch Verification**:
   - Tests each candidate token with multiple prompt formulations
   - A token is classified as a glitch if it has a low probability (<0.2 or <0.01 depending on model) of being correctly repeated AND is not the top predicted token
   
5. **Iteration**:
   - Selects the token with the highest actual entropy for the next iteration
   - Updates the mask to avoid re-testing tokens

This approach allows efficient discovery of glitch tokens by leveraging the model's own gradients to guide the search toward tokens likely to cause unusual behaviors.

### Why This Works

The gradient-guided approach is effective because:
- Tokens causing high entropy in the model's next-token distribution are more likely to be glitch tokens
- The gradient indicates which direction in embedding space will increase entropy
- Similar tokens in embedding space often have similar behaviors
- By iteratively exploring the embedding space, the algorithm can efficiently find clusters of glitch tokens

## Example Workflow

Here's a complete workflow for discovering, validating, and classifying glitch tokens in a Llama 3.2 model:

### 1. Initial Quick Scan (15-30 minutes)

```bash
# Run a quick initial scan to find potential glitch tokens
glitcher mine meta-llama/Llama-3.2-1B-Instruct --num-iterations 50 --batch-size 4 --k 16 --output initial_tokens.json
```

### 2. Validate Discovered Tokens (5-10 minutes)

```bash
# Test the discovered tokens to confirm they're actually glitches
glitcher test meta-llama/Llama-3.2-1B-Instruct --token-file initial_tokens.json --output validated_tokens.json
```

### 3. Analyze Token Behavior (5-10 minutes per token)

```bash
# Test a few of the most interesting tokens in detail
glitcher chat meta-llama/Llama-3.2-1B-Instruct 89472 --max-size 50
glitcher chat meta-llama/Llama-3.2-1B-Instruct 127438 --max-size 50
```

### 4. Classify and Document (30-60 minutes)

```bash
# Classify the validated tokens by their effects
glitch-classify meta-llama/Llama-3.2-1B-Instruct --token-file validated_tokens.json --max-tokens 100 --output classification_results.json
```

### 5. Expanded Search (optional, 1-4 hours)

```bash
# Run a more comprehensive scan based on findings
glitcher mine meta-llama/Llama-3.2-1B-Instruct --num-iterations 300 --batch-size 8 --k 32 --save-interval 10 --progress-file comprehensive_scan.json
```

## List of Known Glitch Tokens

Below are examples of tokens that have been confirmed as glitch tokens in various models:

| Token ID | Token Text | Affected Models | Classification |
|----------|------------|----------------|----------------|
| 89472 | useRalative | Llama-3.2-1B-Instruct | Disruption |
| 127438 | ▍▍▍▍▍▍▍▍▍▍▍▍▍▍▍▍ | Llama-3.2-1B-Instruct | IDOS |
| 107658 | итися | Llama-3.2-1B-Instruct | Hallucination |
| 112162 | уватися | Llama-3.2-1B-Instruct | Disruption |
| 178 | � | Llama-3.2-1B-Instruct | IDOS, Hallucination |

Note: This is not an exhaustive list. Different models may have different glitch tokens, and model updates can change which tokens exhibit glitch behavior.

## Security and Ethical Considerations

The study of glitch tokens serves to improve model robustness and security, but requires responsible handling:

- **Responsible Disclosure**: Report significant findings to model providers before public disclosure
- **Research Purpose**: This tool is intended for legitimate research and model improvement
- **No Exploitation**: Do not use glitch tokens to harm systems or bypass safety measures
- **Documentation**: Keep clear records of testing procedures and results

## License

MIT