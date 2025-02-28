# Glitcher

A command-line tool for mining, testing, and classifying glitch tokens in large language models.

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

## Usage

Glitcher provides several CLI tools for working with glitch tokens:

### Mining Glitch Tokens

Find glitch tokens in a model:

```bash
glitcher mine meta-llama/Llama-3.2-1B --num-iterations 50 --batch-size 8 --k 32
```

Resume a mining session:

```bash
glitcher mine meta-llama/Llama-3.2-1B --resume --progress-file glitch_progress.json
```

### Testing Specific Tokens

Verify if specific tokens are glitch tokens:

```bash
glitcher test meta-llama/Llama-3.2-1B --token-ids 89472,127438,85069
```

Or test tokens from a JSON file:

```bash
glitcher test meta-llama/Llama-3.2-1B --token-file glitch_tokens.json
```

### Chat Testing with a Token

Test how a model responds to a specific token in a chat context:

```bash
glitcher chat meta-llama/Llama-3.2-1B 89472 --max-size 20
```

## Options

### Common Options

- `--device`: Specify the device to use (default: cuda)
- `--quant-type`: Quantization type (bfloat16, float16, int8, int4)

### Mining Options

- `--num-iterations`: Number of iterations to run (default: 50)
- `--batch-size`: Batch size for token testing (default: 8)
- `--k`: Number of nearest tokens to consider (default: 32)
- `--output`: Output file for results (default: glitch_tokens.json)
- `--save-interval`: Save progress every N iterations (default: 5)
- `--resume`: Resume from previous progress file
- `--progress-file`: File to save/load progress (default: glitch_progress.json)

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
```

### Classification Categories

Glitch tokens are classified into these categories:

1. **Injection**: Tokens that enable bypassing instructions or filters
   - Test: Checks if the token allows the model to ignore safety instructions
   
2. **IDOS (Infinite/Denial-of-Service)**: Tokens that trigger repetitive loops
   - Test: Checks if the token causes endless repetition or verbose output
   
3. **Hallucination**: Tokens causing nonsensical outputs
   - Test: Checks if the token makes the model produce incoherent responses
   
4. **Disruption**: Tokens that disrupt the model's internal reasoning
   - Test: Checks if the token makes the model fail at simple tasks like math
   
5. **Bypass**: Tokens that allow bypassing specific safety filters
   - Test: Checks if the token makes the model ignore explicit instructions

Each token is tested with these specific test prompts, and can be classified into multiple categories.

### Classification Output

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

## How It Works

Glitcher discovers and tests "glitch tokens" - tokens that language models are unable to properly repeat when asked. These tokens can be used for various purposes including better understanding model behavior and limitations.

The mining process works by:

1. Starting with a random token
2. Calculating entropy and gradient for the token
3. Finding similar tokens with higher estimated entropy
4. Testing if these tokens are "glitch tokens"
5. Using the highest entropy token for the next iteration

## Examples

### Finding glitch tokens in Llama 3 (1B model)

```bash
glitcher mine meta-llama/Llama-3.2-1B --num-iterations 20
```

### Verifying a list of known glitch tokens

```bash
glitcher test meta-llama/Llama-3.2-1B --token-ids 89472,127438,85069,126523,80370
```

### Testing a specific token in a chat context

```bash
glitcher chat meta-llama/Llama-3.2-1B 89472
```

### Scanning for glitch tokens in a single step

```bash
glitch-scan meta-llama/Llama-3.2-1B --num-iterations 50 --batch-size 8
```

### Classifying discovered glitch tokens

```bash
glitch-classify meta-llama/Llama-3.2-1B --token-file validated_glitch_tokens.json
```

## License

MIT