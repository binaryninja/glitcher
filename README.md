# Glitcher

A command-line tool for mining and testing glitch tokens in large language models.

## Installation

```bash
pip install git+https://github.com/binaryninja/glitcher.git
```

Or clone the repository and install locally:

```bash
git clone https://github.com/binaryninja/glitcher.git
cd glitcher
pip install -e .
```

## Usage

Glitcher provides a CLI tool with three main commands:

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

## License

MIT