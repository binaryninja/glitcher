# Glitcher First Run Guide

Complete reference for all glitcher features. Assumes you have a GPU with CUDA and an instruct-tuned model.

## Installation

```bash
git clone https://github.com/binaryninja/glitcher.git
cd glitcher
pip install -e .
pip install accelerate bitsandbytes matplotlib
```

## Three Entry Points

| Command | Purpose |
|---------|---------|
| `glitcher` | Main CLI with all subcommands |
| `glitch-scan` | Combined mine + validate pipeline |
| `glitch-classify` | Standalone classification tool |

---

## 1. Mining Glitch Tokens

Discover glitch tokens using entropy-guided embedding search. Tokens with low L2 norms in embedding space cluster together and exhibit anomalous behavior.

### Basic mining

```bash
glitcher mine meta-llama/Llama-3.2-1B-Instruct \
  --num-iterations 50 \
  --batch-size 8 \
  --k 32
```

Output: `glitch_tokens.json` containing discovered token IDs and metadata.

### Mining modes

```bash
# Default: entropy-guided gradient search through embedding space
glitcher mine MODEL --mode entropy

# Scan a specific token ID range
glitcher mine MODEL --mode range --range-start 128000 --range-end 128256 --sample-rate 0.5

# Target Unicode token ranges
glitcher mine MODEL --mode unicode --sample-rate 0.15

# Search special/reserved token ranges
glitcher mine MODEL --mode special
```

### Resume interrupted mining

```bash
# Start with checkpointing
glitcher mine MODEL --num-iterations 500 --save-interval 10 --progress-file scan_progress.json

# Resume later
glitcher mine MODEL --num-iterations 500 --resume --progress-file scan_progress.json
```

### Enhanced validation during mining

Enhanced validation generates 30-100+ tokens per test and searches for the target token in the output, reducing false positives from ~30-50% down to ~5-15%.

```bash
# Enabled by default. To disable:
glitcher mine MODEL --disable-enhanced-validation

# Configure validation depth
glitcher mine MODEL --validation-tokens 100 --num-attempts 3 --asr-threshold 0.5
```

### All mining options

```
--num-iterations N       Iterations to run (default: 50)
--batch-size N           Batch size (default: 8)
--k N                    Nearest neighbors in embedding space (default: 32)
--mode {entropy|range|unicode|special}  Mining strategy (default: entropy)
--output FILE            Output file (default: glitch_tokens.json)
--save-interval N        Checkpoint every N iterations (default: 5)
--progress-file FILE     Checkpoint file (default: glitch_progress.json)
--resume                 Resume from progress file
--enhanced-validation    Use enhanced validation (default: on)
--disable-enhanced-validation  Use standard validation only
--validation-tokens N    Tokens to generate per validation (default: 50)
--num-attempts N         Validation attempts (default: 1)
--asr-threshold FLOAT    Attack success rate threshold (default: 0.5)
--range-start ID         Start of token range (range mode)
--range-end ID           End of token range (range mode)
--sample-rate FLOAT      Sampling fraction for range modes (default: 0.1)
--max-tokens-per-range N Max tokens per range (default: 100)
--reasoning-level {none|medium|deep}  For Harmony/gpt-oss models (default: medium)
--device {cuda|cpu}      Device (default: cuda)
--quant-type {auto|bfloat16|float16|int8|int4}  Quantization (default: bfloat16)
```

---

## 2. Testing Tokens

Validate specific tokens for glitch behavior.

```bash
# Test specific token IDs
glitcher test MODEL --token-ids 89472,127438,85069

# Test from a file
glitcher test MODEL --token-file glitch_tokens.json

# Enhanced validation (recommended)
glitcher test MODEL --token-ids 89472 --enhanced --max-tokens 50 --num-attempts 3

# With quantization for large models
glitcher test MODEL --token-ids 89472 --quant-type int4
```

### All test options

```
--token-ids "ID1,ID2"   Comma-separated token IDs
--token-file FILE        JSON file with token IDs
--enhanced               Use enhanced multi-token validation
--max-tokens N           Max tokens to generate (default: 100)
--num-attempts N         Validation attempts (default: 1)
--asr-threshold FLOAT    ASR threshold (default: 0.5)
--output FILE            Output file (default: test_results.json)
--quiet                  Suppress warnings
--device {cuda|cpu}      Device (default: cuda)
--quant-type {auto|bfloat16|float16|int8|int4}  Quantization (default: bfloat16)
```

---

## 3. Comparing Validation Methods

Side-by-side comparison of standard vs enhanced validation.

```bash
glitcher compare MODEL --token-ids 89472,100,300 --num-attempts 2 --max-tokens 50
```

Shows which tokens are identified differently by each method, with agreement/disagreement statistics.

---

## 4. Interactive Chat

Chat with a model while injecting a specific token to observe its behavior in real time.

```bash
glitcher chat MODEL 89472 --max-size 20
```

Pipe empty input for non-interactive testing:

```bash
echo '' | timeout 120 glitcher chat MODEL 89472 --max-size 20
```

---

## 5. Domain Extraction Testing

Test whether glitch tokens break domain extraction from email headers. Inserts each token into a domain name and asks the model to extract it as JSON.

```bash
# Test specific tokens
glitcher domain MODEL --token-ids 8673,10939

# Test known CppTypeDefinitionSizes glitch token
glitcher domain MODEL --test-cpptypes

# Load tokens from file
glitcher domain MODEL --token-file glitch_tokens.json

# Skip normal token control group
glitcher domain MODEL --token-ids 8673 --skip-normal
```

### All domain options

```
--token-ids "ID1,ID2"   Comma-separated token IDs
--token-file FILE        JSON file with token IDs
--test-cpptypes          Test known CppTypeDefinitionSizes token (ID 89472)
--skip-normal            Skip normal token control group
--normal-count N         Number of control tokens (default: 5)
--output FILE            Output file (default: domain_extraction_results.json)
--device {cuda|cpu}      Device (default: cuda)
--quant-type {bfloat16|float16|int8|int4}  Quantization (default: bfloat16)
```

---

## 6. Classification

Classify glitch tokens into categories by running them through behavioral tests: email extraction, domain extraction, prompt injection, infinite loops, hallucination, and more.

### Classification modes

```bash
# Full classification (all tests)
glitcher classify MODEL --token-ids 89472,127438

# Email extraction tests only
glitcher classify MODEL --token-file glitch_tokens.json --email-extraction-only

# Domain extraction tests only
glitcher classify MODEL --token-ids 8673,10939 --domain-extraction-only

# Behavioral tests only
glitcher classify MODEL --token-file glitch_tokens.json --behavioral-only

# Functional tests only
glitcher classify MODEL --token-file glitch_tokens.json --functional-only
```

### Standalone classifier entry point

```bash
glitch-classify MODEL --token-ids 8673,10939 --email-extraction-only --max-tokens 100
```

### Classification categories

| Category | Description |
|----------|-------------|
| INJECTION | Prompt injection / jailbreaking |
| IDOS | Infinite loops / denial-of-service |
| HALLUCINATION | Nonsensical output generation |
| DISRUPTION | Internal reasoning disruption |
| BYPASS | Filter or guardrail bypass |
| EMAIL_EXTRACTION | Breaks email parsing |
| VALID_EMAIL_ADDRESS | Token creates a syntactically valid email |
| DOMAIN_EXTRACTION | Breaks domain parsing |
| VALID_DOMAIN_NAME | Token creates a syntactically valid domain |
| UNKNOWN | Unable to categorize |

### All classify options

```
--token-ids "ID1,ID2"      Comma-separated token IDs
--token-file FILE           JSON file with token IDs
--email-extraction-only     Only email extraction tests
--domain-extraction-only    Only domain extraction tests
--behavioral-only           Only behavioral tests
--functional-only           Only functional tests
--max-tokens N              Max tokens per test (default: 200)
--temperature FLOAT         Sampling temperature (default: 0.0)
--output FILE               Output file (default: classified_tokens.json)
--debug-responses           Verbose response logging
--device {cuda|cpu}         Device (default: cuda)
--quant-type {bfloat16|float16|int8|int4}  Quantization (default: bfloat16)
```

---

## 7. Genetic Algorithm

Evolutionary search for token combinations that maximize probability reduction (or increase a wanted token's probability). Breeds populations of token combos across generations.

### Standard evolution

```bash
glitcher genetic MODEL \
  --base-text "The quick brown" \
  --token-file glitch_tokens.json \
  --generations 100 \
  --population-size 50 \
  --max-tokens 3
```

### Wanted token optimization

Find token combinations that increase the probability of a specific output token:

```bash
glitcher genetic MODEL \
  --base-text "The quick brown" \
  --wanted-token cat \
  --token-file glitch_tokens.json \
  --generations 50
```

### Baseline-only analysis

Rank individual tokens by their impact without running the full genetic algorithm:

```bash
glitcher genetic MODEL \
  --base-text "The quick brown" \
  --baseline-only \
  --token-file glitch_tokens.json
```

### ASCII-only mode

Restrict the token pool to ASCII-only tokens:

```bash
glitcher genetic MODEL \
  --base-text "The quick brown" \
  --ascii-only \
  --token-file glitch_tokens.json
```

### Comprehensive vocabulary search

Scan the full model vocabulary for wanted token optimization (slower but thorough):

```bash
glitcher genetic MODEL \
  --base-text "The quick brown" \
  --wanted-token fox \
  --comprehensive-search \
  --token-file glitch_tokens.json
```

Results are cached in `cache/comprehensive_search/`. Use `--clear-cache` to reset or `--no-cache` to skip caching.

### Batch experiments

Run multiple scenarios in one go:

```bash
glitcher genetic MODEL --batch --token-file glitch_tokens.json
```

### Real-time GUI visualization

Watch the evolution in real time with matplotlib:

```bash
glitcher genetic MODEL \
  --base-text "The quick brown" \
  --token-file glitch_tokens.json \
  --gui
```

### All genetic options

```
--base-text TEXT             Text to test against (default: "The quick brown")
--target-token TOKEN         Token to reduce (auto-detected if omitted)
--wanted-token TOKEN         Token to increase probability of
--token-file FILE            Glitch tokens file (default: glitch_tokens.json)
--ascii-only                 ASCII-only tokens
--include-normal-tokens      Include normal vocab tokens in pool
--comprehensive-search       Full vocabulary scan for wanted token
--no-cache                   Disable search cache
--clear-cache                Clear search cache before running

--population-size N          Population size (default: 50)
--generations N              Max generations (default: 100)
--mutation-rate FLOAT        Mutation rate (default: 0.1)
--crossover-rate FLOAT       Crossover rate (default: 0.7)
--elite-size N               Elite preserved across generations (default: 5)
--max-tokens N               Max tokens per individual (default: 3)
--early-stopping-threshold F Stop at this reduction (default: 0.999)

--exact-token-count          Fixed token count per individual (default)
--variable-token-count       Allow 1 to max-tokens per individual

--adaptive-mutation          Dynamic mutation rate (high to low)
--initial-mutation-rate F    Starting mutation rate (default: 0.8)
--final-mutation-rate F      Ending mutation rate (default: 0.1)

--baseline-seeding           Seed population from baseline results (default)
--no-baseline-seeding        Random-only seeding
--baseline-seeding-ratio F   Fraction seeded from baseline (default: 0.7)
--baseline-only              Only run baseline analysis, skip evolution
--skip-baseline              Skip baseline, go straight to evolution
--baseline-top-n N           Top tokens to display (default: 10)

--sequence-aware-diversity   Prevent duplicate sequences (default)
--no-sequence-diversity      Traditional diversity only
--sequence-diversity-ratio F Sequence diversity fraction (default: 0.3)

--enable-shuffle-mutation    Allow token reordering
--gui                        Real-time matplotlib visualization
--batch                      Run batch experiments
--output FILE                Results file (default: genetic_results.json)
--device {cuda|cpu}          Device (default: cuda)
--quant-type {bfloat16|float16|int8|int4}  Quantization (default: bfloat16)
```

---

## 8. Validate

Quick smoke test that loads the model, runs a few tokens through `strictly_glitch_verify`, and saves results to a directory.

```bash
glitcher validate MODEL --output-dir /tmp/validation_results
```

---

## 9. Scan and Validate

Combined pipeline: mine tokens, then validate them. Single command.

```bash
glitch-scan MODEL \
  --num-iterations 50 \
  --batch-size 8 \
  --k 32 \
  --chat-sample-size 3 \
  --output validated_glitch_tokens.json
```

---

## 10. GUI

Full tkinter GUI for configuring and running the genetic algorithm interactively.

```bash
glitcher gui
glitcher gui --config my_config.json
```

Tabs: Config, Control (start/pause/stop), Progress (real-time metrics), Results (best individuals and fitness scores).

---

## Quantization

All subcommands accept `--quant-type`:

| Type | VRAM | Speed | Quality | When to use |
|------|------|-------|---------|-------------|
| `bfloat16` | ~2x model | Fast | Best | Default, most GPUs |
| `float16` | ~2x model | Fast | Good | Llama 3.2-1B (avoids generation bugs) |
| `int8` | ~1x model | Medium | Good | 8-16GB VRAM |
| `int4` | ~0.5x model | Slower | Fair | <8GB VRAM |
| `auto` | Auto | Auto | Auto | Let the library decide |

```bash
# Large model on limited VRAM
glitcher mine meta-llama/Llama-3.2-7B-Instruct --quant-type int8
glitcher test meta-llama/Llama-3.2-7B-Instruct --token-ids 89472 --quant-type int4
```

---

## Key Concepts

**Glitch tokens** are tokens with low prediction probability that cause unexpected model behavior: hallucinations, loops, safety bypasses, or garbled output. They cluster in low-norm regions of embedding space.

**ASR (Attack Success Rate)** is the fraction of validation attempts where a token exhibits glitch behavior. Default threshold is 0.5. Set to 0.0 to flag any glitch, 1.0 to require consistent failure.

**Enhanced validation** generates 30-100+ tokens per test and searches for the target in the output sequence, rather than just checking immediate next-token probability. This accounts for models that say "Sure! Here is..." before producing the token.

**Harmony support** handles gpt-oss models with structured channel-based output (analysis/commentary/final) and configurable reasoning effort levels.

---

## Output Files

| File | Produced by | Contents |
|------|-------------|----------|
| `glitch_tokens.json` | `mine` | Token IDs, text, L2 norms, entropy, ASR, method |
| `glitch_progress.json` | `mine` | Resumable checkpoint |
| `test_results.json` | `test` | Per-token validation results |
| `comparison_results.json` | `compare` | Standard vs enhanced comparison |
| `domain_extraction_results.json` | `domain` | Domain extraction test results |
| `classified_tokens.json` | `classify` | Categories and behavioral test results |
| `genetic_results.json` | `genetic` | Top evolved token combinations |
| `token_impact_baseline.json` | `genetic --baseline-only` | Individual token effectiveness ranking |

---

## Typical Workflow

```bash
MODEL=meta-llama/Llama-3.2-1B-Instruct

# 1. Discover glitch tokens
glitcher mine $MODEL --num-iterations 50 --batch-size 8 --k 32

# 2. Validate with enhanced method
glitcher test $MODEL --token-file glitch_tokens.json --enhanced --max-tokens 50

# 3. Classify by behavior
glitcher classify $MODEL --token-file glitch_tokens.json --output classified.json

# 4. Test email/domain extraction impact
glitcher classify $MODEL --token-file glitch_tokens.json --email-extraction-only --output email_results.json
glitcher domain $MODEL --token-file glitch_tokens.json --output domain_results.json

# 5. Find optimal attack combinations
glitcher genetic $MODEL --base-text "The quick brown" --token-file glitch_tokens.json --generations 50

# 6. Explore interactively
glitcher chat $MODEL 89472 --max-size 30

# 7. Generate a report
python scripts/generate_enhanced_report.py email_results.json report.html
```

---

## Provider System (poc/providers/)

For testing across different LLM APIs:

```python
from poc.providers import get_provider, list_available_providers

# Available: openai, anthropic, mistral, openrouter, lambda, transformers
provider = get_provider('transformers',
    model_path='meta-llama/Llama-3.2-1B-Instruct',
    quant_type='int4')

available = list_available_providers()
```

---

## Running Tests

Tests are standalone scripts (not pytest). They require GPU + model.

```bash
# Unit tests
python tests/unit/test_genetic_probabilities.py meta-llama/Llama-3.2-1B-Instruct
python tests/unit/test_glitch_error_detection.py
python tests/unit/test_next_token_probability.py meta-llama/Llama-3.2-1B-Instruct
python tests/unit/test_proper_context.py meta-llama/Llama-3.2-1B-Instruct

# Integration tests
python tests/integration/test_enhanced_classification.py
python tests/integration/test_genetic_integration.py
python tests/integration/test_report_generator.py
python tests/integration/test_reports.py

# Full functional test suite
cd glitch_data/glitcher_test
./run_tests.sh
```

### Linting

```bash
flake8 glitcher/
black glitcher/ --check
black glitcher/  # auto-format
```

---

## Troubleshooting

**Out of memory**: Use `--quant-type int8` or `--quant-type int4` and reduce `--batch-size`.

**Llama 3.2-1B garbled output**: Use `--quant-type float16` instead of the default bfloat16.

**High false positive rate**: Use `--enhanced` validation with `--num-attempts 3`.

**Temperature/top_p warnings**: Cosmetic, safe to ignore. Use `--quiet` to suppress.

**Model not supported**: Glitcher requires instruct-tuned models with chat templates. Base models will not work correctly. Look for "Instruct" or "Chat" in the model name.
